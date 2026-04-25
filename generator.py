"""
ADOFAI -> 太鼓达人风格视频生成器 - 视频生成核心  [v3]
- TaikoAdofaiGenerator：全局距离预计算 + 二分查找加速的视频生成器
- GPU 检测工具：自动检测 AMD / NVIDIA / 无GPU，智能选择编码器
- 音频工具：扫描与选择背景音乐文件

v3 变更 (相比 v2):
  [核心] speed_segments 去重 —— 只在BPM变化时建断点，997K段→可能几百段
  [核心] _fast_calc_appear_time 替换为 _time_at_distance 解析解 —— 50轮二分→1次searchsorted
  [核心] 向量化批量计算音符位置 —— 逐个循环→numpy批处理，O(n)→O(1)次searchsorted
  [体验] tqdm 进度条 —— __init__ 预计算全程可见
  [保留] bisect 时间窗口裁剪 (来自v2)
"""

import math
import os
import subprocess
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from config import CONFIG
from chart import Tile, load_adofai_tiles


# ═══════════════════════════════════════════════════
#  GPU 检测工具
# ═══════════════════════════════════════════════════

def ffmpeg_available() -> bool:
    """检测系统是否安装了 FFmpeg"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


def detect_encoder() -> Tuple[str, list]:
    """
    自动检测最佳视频编码器及参数
    优先级：AMD AMF > NVIDIA NVENC > CPU libx264
    返回：(编码器名称, 编码参数列表)
    """
    # AMD AMF
    try:
        result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
        if 'h264_amf' in result.stdout:
            return ('h264_amf', [
                '-quality', 'balanced',
                '-usage', 'lowlatency',
                '-rc', 'cqp',
                '-qp_i', '23',
                '-qp_p', '23',
                '-g', '30',
                '-keyint_min', '30',
                '-bf', '0',
                '-refs', '1',
                '-pix_fmt', 'yuv420p'
            ])
    except FileNotFoundError:
        pass

    # NVIDIA NVENC
    try:
        subprocess.run(['nvidia-smi'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return ('h264_nvenc', [
            '-preset', 'p1',
            '-rc', 'cqp',
            '-qp', '23',
            '-g', '30',
            '-bf', '0',
            '-pix_fmt', 'yuv420p'
        ])
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # CPU 回退
    return ('libx264', [
        '-preset', 'veryfast',
        '-tune', 'zerolatency',
        '-crf', '23',
        '-g', '30',
        '-bf', '0',
        '-pix_fmt', 'yuv420p'
    ])


# ═══════════════════════════════════════════════════
#  背景音乐工具
# ═══════════════════════════════════════════════════

AUDIO_EXTENSIONS = {'.ogg', '.wav', '.aiff', '.mp3', '.flac'}


def scan_audio_files(directory: str) -> List[str]:
    """扫描目录下的常见音频文件"""
    files = []
    for f in os.listdir(directory):
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS:
            files.append(os.path.join(directory, f))
    return sorted(files)


def select_audio_file(audio_files: List[str]) -> str | None:
    """交互式选择一个音频文件"""
    if not audio_files:
        print('  未找到任何音频文件。')
        return None
    print('\n  找到以下音频文件：')
    for i, f in enumerate(audio_files, 1):
        print(f'    [{i}] {os.path.basename(f)}')
    print(f'    [0] 不使用背景音乐')
    while True:
        choice = input('  请选择: ').strip()
        if choice == '0':
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(audio_files):
                return audio_files[idx]
            else:
                print('  无效序号，请重新输入')
        except ValueError:
            print('  无效输入')


# ═══════════════════════════════════════════════════
#  视频生成器
# ═══════════════════════════════════════════════════

class TaikoAdofaiGenerator:
    def __init__(self, adofai_path: str,
                 base_speed: float = 300.0,
                 speed_mult: float = 1.0,
                 screen_w: int = 1920,
                 fps: float = 60.0,
                 track_height: int = 160,
                 use_gpu: bool = True,
                 bgm_path: str = None,
                 fixed_speed: bool = False,
                 min_note_interval: float = 0.0):
        self.fixed_speed = fixed_speed
        self.min_note_interval = min_note_interval
        # 当固定速度模式启用最小音符间隔时，需要变速，内部转为非固定速度模式
        self._use_min_interval = fixed_speed and min_note_interval > 0
        self.tiles, self.offset_ms = load_adofai_tiles(adofai_path)
        self.base_speed = base_speed
        self.speed_mult = speed_mult
        self.width = screen_w
        self.height = track_height
        self.fps = fps
        self.track_height = track_height
        self.use_gpu = use_gpu
        self.bgm_path = bgm_path

        self.scroll_distance = self.width - CONFIG['hit_x']

        note_dia = int(self.track_height * CONFIG['note_dia_ratio'])
        self.note_radius = note_dia // 2
        self.judge_radius = self.note_radius

        # 过滤 midspin
        self.notes: List[Tile] = [t for t in self.tiles[1:] if not t.midspin]
        if not self.notes:
            raise ValueError('没有可用的音符 (可能全部为 midspin)')

        last_offset = self.notes[-1].offset
        self.total_duration = last_offset + CONFIG['after_black']

        # ── 预计算（带进度条） ──
        n_notes = len(self.notes)
        print(f'⚙️  预计算滚动数据 ({n_notes} 个音符)...')

        with tqdm(total=3, desc='   预计算', bar_format='{desc} |{bar}| {n_fmt}/{total_fmt} 步') as pbar:
            # 步骤1: 构建速度段
            pbar.set_postfix_str('构建速度段')
            if self._use_min_interval:
                # 最小音符间隔模式：内部使用非固定速度，用虚拟BPM实现变速
                saved_fixed = self.fixed_speed
                self.fixed_speed = False
                self.speed_segments = self._build_min_interval_segments()
            else:
                self.speed_segments = self._build_dedup_speed_segments()
            pbar.update(1)

            # 步骤2: 构建距离函数
            pbar.set_postfix_str('距离函数')
            self._build_global_distance_function()
            if self._use_min_interval:
                self.fixed_speed = saved_fixed
            pbar.update(1)

            # 步骤3: 向量化批量计算所有音符位置 —— 一次numpy批处理替代逐个50轮二分
            pbar.set_postfix_str('音符位置')
            self._batch_compute_note_data()
            pbar.update(1)

        n_segments = len(self._knot_times) - 1
        print(f'   ✅ 预计算完成: {n_segments} 个速度段 (去重后), {n_notes} 个音符')

    # ── 速度段去重 ─────────────────────────────

    def _build_dedup_speed_segments(self) -> List[Tuple[float, float]]:
        """
        [v3核心] 合并连续相同BPM的速度段
        原代码: 每个音符建一个段 → 997K段 → _knot_times 巨大
        优化后: 只在BPM变化时建段 → 通常几百~几千段
        距离函数结果完全不变，因为相同BPM的区间斜率一样，不需要分割
        """
        segments: List[Tuple[float, float]] = []
        prev_bpm = None
        for note in self.notes:
            bpm = note.bpm if note.bpm > 0 else 100.0
            if bpm != prev_bpm:
                segments.append((note.offset, bpm))
                prev_bpm = bpm
        return segments

    def _build_min_interval_segments(self) -> List[Tuple[float, float]]:
        """
        [最小音符间隔] 在固定速度模式下，为距离过近的音符对构建变速段。

        原理：
          固定速度下，相邻音符的屏幕间距 = dt × base_vel。
          若该间距 < min_note_interval，则在两个音符之间临时加速滚动，
          使屏幕间距恰好等于 min_note_interval，加速后回到原速度。

        实现方式：
          内部切换为非固定速度模式，用"虚拟BPM"编码所需速度，
          使得 _velocity(bpm) 能返回正确的变速值。
        """
        base_vel = self.base_speed * self.speed_mult
        base_bpm = CONFIG['base_bpm']
        virtual_base_bpm = base_bpm  # _velocity(base_bpm) = base_vel

        segments: List[Tuple[float, float]] = [(0.0, virtual_base_bpm)]

        for i in range(len(self.notes) - 1):
            dt = self.notes[i + 1].offset - self.notes[i].offset
            if dt > 0:
                normal_dist = dt * base_vel
                if normal_dist < self.min_note_interval:
                    needed_vel = self.min_note_interval / dt
                    virtual_bpm = base_bpm * needed_vel / base_vel
                    # 在当前音符处加速
                    segments.append((self.notes[i].offset, virtual_bpm))
                    # 在下一个音符处回到原速度
                    segments.append((self.notes[i + 1].offset, virtual_base_bpm))

        return segments

    # ── 向量化计算方法 ─────────────────────────

    def _velocity(self, bpm: float) -> float:
        """计算滚动速度，fixed_speed模式下忽略BPM变化"""
        if self.fixed_speed:
            return self.base_speed * self.speed_mult
        return self.base_speed * (bpm / CONFIG['base_bpm']) * self.speed_mult

    def _velocity_array(self, bpms: np.ndarray) -> np.ndarray:
        """向量化版本：批量计算速度"""
        if self.fixed_speed:
            return np.full_like(bpms, self.base_speed * self.speed_mult)
        return self.base_speed * (bpms / CONFIG['base_bpm']) * self.speed_mult

    def _build_global_distance_function(self):
        if not self.speed_segments:
            self._knot_times = np.array([0.0, self.total_duration])
            self._knot_dists = np.array([0.0, 0.0])
            self._knot_bpms_after = np.array([100.0, 100.0])
            return

        breakpoints = sorted({0.0, self.total_duration} | {t for t, _ in self.speed_segments})

        knots = []
        first_bpm = self.speed_segments[0][1] if self.speed_segments else 100.0
        knots.append((0.0, 0.0, first_bpm))

        seg_idx = 0
        current_bpm = first_bpm
        for i in range(len(breakpoints) - 1):
            t_start = breakpoints[i]
            t_end = breakpoints[i + 1]
            while seg_idx < len(self.speed_segments) and self.speed_segments[seg_idx][0] <= t_start:
                current_bpm = self.speed_segments[seg_idx][1]
                seg_idx += 1
            v = self._velocity(current_bpm)
            dist_add = v * (t_end - t_start)
            prev_dist = knots[-1][1]
            knots.append((t_end, prev_dist + dist_add, current_bpm))

        self._knot_times = np.array([k[0] for k in knots], dtype=np.float64)
        self._knot_dists = np.array([k[1] for k in knots], dtype=np.float64)
        self._knot_bpms_after = np.array([k[2] for k in knots], dtype=np.float64)

    def _distance_at_time(self, t: float) -> float:
        """标量版本：计算时刻 t 的全局距离（用于 _draw_frame 每帧1次调用）"""
        if t <= self._knot_times[0]:
            dt = t - self._knot_times[0]
            v = self._velocity(self._knot_bpms_after[0])
            return self._knot_dists[0] + v * dt
        if t >= self._knot_times[-1]:
            dt = t - self._knot_times[-1]
            v = self._velocity(self._knot_bpms_after[-1])
            return self._knot_dists[-1] + v * dt
        idx = np.searchsorted(self._knot_times, t, side='right') - 1
        t0 = self._knot_times[idx]
        dist0 = self._knot_dists[idx]
        bpm = self._knot_bpms_after[idx]
        v = self._velocity(bpm)
        return dist0 + v * (t - t0)

    def _distance_at_times(self, t_arr: np.ndarray) -> np.ndarray:
        """[v3] 向量化版本：一次计算多个时刻的全局距离"""
        n_knots = len(self._knot_times)
        indices = np.searchsorted(self._knot_times, t_arr, side='right') - 1
        indices = np.clip(indices, 0, n_knots - 2)

        t0s = self._knot_times[indices]
        dist0s = self._knot_dists[indices]
        bpms = self._knot_bpms_after[indices]
        velocities = self._velocity_array(bpms)

        return dist0s + velocities * (t_arr - t0s)

    def _time_at_distances(self, d_arr: np.ndarray) -> np.ndarray:
        """
        [v3核心] 向量化距离→时间 反函数
        替代原来的 _fast_calc_appear_time (50轮二分搜索)
        距离函数是分段线性的，反函数可直接解析求解：
          d = dist0 + v * (t - t0)  →  t = t0 + (d - dist0) / v
        只需一次 searchsorted 找到 d 落在哪个区间，然后直接计算
        """
        n_knots = len(self._knot_dists)
        indices = np.searchsorted(self._knot_dists, d_arr, side='right') - 1
        indices = np.clip(indices, 0, n_knots - 2)

        t0s = self._knot_times[indices]
        dist0s = self._knot_dists[indices]
        bpms = self._knot_bpms_after[indices]
        velocities = self._velocity_array(bpms)

        # 安全除法：速度不可能为0（base_speed > 0, bpm > 0），但做防护
        safe_v = np.where(velocities > 0, velocities, 1.0)
        return t0s + (d_arr - dist0s) / safe_v

    def _batch_compute_note_data(self):
        """
        [v3核心] 向量化批量计算所有音符的 t_appear / dist_at_hit / dist_at_appear
        原代码: 逐个音符循环 + 每个音符50轮二分 → 5000万次 searchsorted
        优化后: 3次向量化 searchsorted → 总共3次批处理
        """
        n_notes = len(self.notes)

        # 构建输入数组
        t_hits = np.array([note.offset for note in self.notes], dtype=np.float64)
        bpms = np.array([note.bpm if note.bpm > 0 else 100.0 for note in self.notes], dtype=np.float64)

        # 批量计算 dist_at_hit
        dist_at_hits = self._distance_at_times(t_hits)

        # 批量计算 target_dist = dist_at_hit - scroll_distance
        target_dists = dist_at_hits - self.scroll_distance

        # 批量计算 t_appear —— 核心优化：解析解替代50轮二分
        t_appears = self._time_at_distances(target_dists)

        # 批量计算 dist_at_appear
        dist_at_appears = self._distance_at_times(t_appears)

        # 存储为并行 numpy 数组（比 list[dict] 更快、更省内存）
        self._nd_t_hit = t_hits
        self._nd_t_appear = t_appears
        self._nd_bpm = bpms
        self._nd_dist_at_hit = dist_at_hits
        self._nd_dist_at_appear = dist_at_appears

    # ── 绘制 ─────────────────────────────────────

    def _draw_frame(self, t: float) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (self.width, self.track_height), (128, 128, 128), -1)

        center_y = self.track_height // 2
        cv2.circle(img, (CONFIG['hit_x'], center_y), self.judge_radius, (50, 50, 50), -1)

        # ── 时间窗口裁剪（v2 的 bisect 优化，改用 np.searchsorted） ──
        left_idx = int(np.searchsorted(self._nd_t_appear, t, side='right'))
        right_idx = int(np.searchsorted(self._nd_t_hit, t, side='left'))

        if right_idx >= left_idx:
            return img

        # 每帧只算1次全局距离
        dist_t = self._distance_at_time(t)

        last_x = None
        for i in range(right_idx, left_idx):
            x = self.width - (dist_t - self._nd_dist_at_appear[i])
            if x + self.note_radius < 0 or x - self.note_radius > self.width:
                continue
            ix = int(x)
            # 跳过完全重叠的音符（屏幕坐标差为 0px）
            if last_x is not None and ix == last_x:
                continue
            last_x = ix
            center = (ix, center_y)
            cv2.circle(img, center, self.note_radius + 3, (255, 255, 255), -1)
            cv2.circle(img, center, self.note_radius, (0, 0, 255), -1)
        return img

    # ── 视频输出 ─────────────────────────────────

    def generate_video(self, output_path: str):
        """根据配置选择 FFmpeg 或 OpenCV 编码器生成视频"""
        if self.bgm_path:
            if not ffmpeg_available():
                raise RuntimeError('背景音乐需要 FFmpeg，但未找到 FFmpeg')
            encoder, preset_params = detect_encoder()
            base, _ = os.path.splitext(output_path)
            output_path = base + '.mp4'
            self._generate_with_ffmpeg(output_path, encoder, preset_params)
        elif self.use_gpu and ffmpeg_available():
            encoder, preset_params = detect_encoder()
            base, _ = os.path.splitext(output_path)
            output_path = base + '.mp4'
            self._generate_with_ffmpeg(output_path, encoder, preset_params)
        else:
            base, _ = os.path.splitext(output_path)
            output_path = base + '.avi'
            self._generate_with_opencv(output_path)

    def _generate_with_ffmpeg(self, output_path: str, encoder: str, preset_params: list):
        n_notes = len(self._nd_t_hit)
        print(f'\n⏳ 正在生成视频 (🎮 GPU加速: {encoder})...')
        if self.bgm_path:
            print(f'   🔈 背景音乐: {os.path.basename(self.bgm_path)}')
            if self.offset_ms != 0:
                print(f'   ⏱️  音频延迟: {self.offset_ms:.0f} 毫秒 (offset 对齐)')
        print(f'   分辨率: {self.width}×{self.height}  |  帧率: {self.fps} fps')
        print(f'   音符数: {n_notes}  |  时长: {self.total_duration:.1f} 秒')
        print(f'   速度: {self.base_speed} px/s (100BPM)  |  变速倍率: {self.speed_mult}')
        print(f'   输出: {output_path}\n')

        # 构建基础命令
        if self.bgm_path:
            t_hit = self.notes[0].offset
            offset_s = self.offset_ms / 1000.0
            start_video = t_hit - offset_s

            if start_video >= 0:
                delay_ms = int(start_video * 1000)
                audio_filters = f'adelay=delays={delay_ms}:all=1'
            else:
                trim_start = -start_video
                audio_filters = f'atrim=start={trim_start},asetpts=PTS-STARTPTS'

            command = [
                'ffmpeg', '-y',
                '-i', self.bgm_path,
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{self.width}x{self.height}',
                '-pix_fmt', 'bgr24',
                '-r', str(self.fps),
                '-use_wallclock_as_timestamps', '1',
                '-i', '-',
                '-filter_complex', f'[0:a]{audio_filters}[aout]',
                '-map', '[aout]',
                '-map', '1:v:0',
                '-c:v', encoder,
                *preset_params,
                '-c:a', 'aac', '-b:a', '192k',
                '-shortest',
                output_path
            ]
        else:
            command = [
                'ffmpeg', '-y',
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{self.width}x{self.height}',
                '-pix_fmt', 'bgr24',
                '-r', str(self.fps),
                '-use_wallclock_as_timestamps', '1',
                '-i', '-',
                '-c:v', encoder,
                *preset_params,
                output_path
            ]

        proc = subprocess.Popen(command, stdin=subprocess.PIPE, bufsize=0)
        total_frames = int(math.ceil(self.total_duration * self.fps))
        last_hit = self.notes[-1].offset if self.notes else 0.0

        try:
            for frame_idx in range(total_frames):
                t = frame_idx / self.fps
                if t > last_hit + CONFIG['after_black']:
                    break
                frame = self._draw_frame(t)
                proc.stdin.write(frame.tobytes())

                if frame_idx % 30 == 0 or frame_idx == total_frames - 1:
                    pct = (t / last_hit) * 100 if last_hit > 0 else 0
                    bar_len = 30
                    filled = int(bar_len * min(pct / 100, 1.0))
                    bar = '█' * filled + '░' * (bar_len - filled)
                    print(f'\r   进度: [{bar}] {min(pct, 100):.0f}%', end='')
        except BrokenPipeError:
            print('\n⚠️  FFmpeg 进程意外退出，请检查编码参数或显卡驱动。')
        finally:
            proc.stdin.close()
            proc.wait()

        print(f'\r   进度: [{"█" * 30}] 100%')
        print(f'\n✅ 视频已保存至: {os.path.abspath(output_path)}')

    def _generate_with_opencv(self, output_path: str):
        n_notes = len(self._nd_t_hit)
        print(f'\n⏳ 正在生成视频 (🐢 CPU 软件编码)...')
        print(f'   分辨率: {self.width}×{self.height}  |  帧率: {self.fps} fps')
        print(f'   音符数: {n_notes}  |  时长: {self.total_duration:.1f} 秒')
        print(f'   速度: {self.base_speed} px/s (100BPM)  |  变速倍率: {self.speed_mult}')
        print(f'   输出: {output_path}\n')

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height), True)
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height), True)
        if not out.isOpened():
            raise RuntimeError('无法创建视频文件，请安装 K-Lite Codec Pack 或检查 opencv 安装')

        total_frames = int(math.ceil(self.total_duration * self.fps))
        last_hit = self.notes[-1].offset if self.notes else 0.0

        for frame_idx in range(total_frames):
            t = frame_idx / self.fps
            if t > last_hit + CONFIG['after_black']:
                break
            frame = self._draw_frame(t)
            out.write(frame)

            if frame_idx % 30 == 0 or frame_idx == total_frames - 1:
                pct = (t / last_hit) * 100 if last_hit > 0 else 0
                bar_len = 30
                filled = int(bar_len * min(pct / 100, 1.0))
                bar = '█' * filled + '░' * (bar_len - filled)
                print(f'\r   进度: [{bar}] {min(pct, 100):.0f}%', end='')

        out.release()
        print(f'\r   进度: [{"█" * 30}] 100%')
        print(f'\n✅ 视频已保存至: {os.path.abspath(output_path)}')
