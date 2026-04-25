#!/usr/bin/env python3
"""
ADOFAI -> 太鼓达人风格视频生成器 - 交互式入口
- 用户交互界面：文件输入、参数调整、背景音乐选择
- 调用 generator 完成视频生成

项目结构：
  config.py    - 全局配置与角度映射
  chart.py     - Tile 数据模型 + 谱面解析
  generator.py - 视频生成器 + GPU检测 + 音频工具
  main.py      - 交互式入口 (本文件)
"""

import os
import sys

from chart import load_adofai_tiles
from generator import (
    TaikoAdofaiGenerator,
    ffmpeg_available,
    detect_encoder,
    scan_audio_files,
    select_audio_file,
)


# ═══════════════════════════════════════════════════
#  交互工具函数
# ═══════════════════════════════════════════════════

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def input_path(prompt: str, allow_empty: bool = False) -> str | None:
    while True:
        raw = input(prompt).strip().strip('"').strip("'")
        if raw:
            return raw
        if allow_empty:
            return ""
        print('⚠️  路径不能为空，请重新输入')


def input_float(prompt: str, default: float) -> float:
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f'⚠️  输入无效，使用默认值: {default}')
        return default


def input_int(prompt: str, default: int) -> int:
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f'⚠️  输入无效，使用默认值: {default}')
        return default


# ═══════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════

def main():
    clear_screen()
    print('═' * 60)
    print('    🥁 ADOFAI → 太鼓达人 视频生成器 (AMD GPU 加速)')
    print('═' * 60)

    gpu_info = ''
    if ffmpeg_available():
        encoder, _ = detect_encoder()
        if encoder == 'h264_amf':
            gpu_info = '✅ 检测到 AMD GPU，将启用 AMF 硬件加速 (优化低延迟)'
        elif encoder == 'h265_nvenc':
            gpu_info = '✅ 检测到 NVIDIA GPU，将启用 NVENC 硬件加速'
        else:
            gpu_info = '⚠️  未检测到支持的硬件编码器，将使用 CPU 软编码 (libx264)'
    else:
        gpu_info = '⚠️  未找到 FFmpeg，将使用 OpenCV 软件编码 (AVI)'

    print(gpu_info)
    print()

    # ── 1. 输入文件 ──
    print('📂 请拖入 ADOFAI 谱面文件 (.adofai)，然后按回车：')
    adofai_path = input_path('   → ', allow_empty=False)
    if not os.path.isfile(adofai_path):
        print(f'\n❌ 文件不存在: {adofai_path}')
        input('\n按回车退出...')
        sys.exit(1)
    print(f'\n✅ 已加载: {os.path.basename(adofai_path)}')

    # 快速显示 offset
    try:
        _, offset_ms = load_adofai_tiles(adofai_path)
        if offset_ms != 0:
            print(f'   ℹ️  谱面 offset: {offset_ms:.0f} 毫秒')
    except:
        pass

    # ── 2. 输出路径 ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_name = os.path.splitext(os.path.basename(adofai_path))[0]
    default_output = os.path.join(script_dir, f"{base_name}_taiko.mp4" if ffmpeg_available() else f"{base_name}_taiko.avi")
    print(f'\n💾 输出视频路径 (直接回车使用默认):')
    print(f'   默认: {default_output}')
    user_out = input_path('   → ', allow_empty=True)
    output_path = user_out if user_out else default_output

    # ── 3. 参数调整菜单 ──
    params = {
        'base_speed': 300.0,
        'speed_mult': 1.0,
        'fps': 60.0,
        'width': 1920,
        'track_height': 160,
        'bgm_enabled': False,
        'bgm_path': '',
        'fixed_speed': False,
        'min_note_interval': 0.0,
    }

    while True:
        clear_screen()
        print('═' * 60)
        print('    ⚙️  参数调整 (直接回车 = 使用当前值)')
        print('═' * 60)
        print()
        print(f'  [1] 基准速度       : {params["base_speed"]:.0f} px/s')
        print(f'  [2] 变速倍率       : {params["speed_mult"]:.1f}')
        print(f'  [3] 帧率           : {params["fps"]:.0f} fps')
        print(f'  [4] 视频宽度       : {params["width"]} px')
        print(f'  [5] 轨道高度       : {params["track_height"]} px')
        bgm_status = f"开启 ({os.path.basename(params['bgm_path'])})" if params['bgm_enabled'] and params['bgm_path'] else '关闭'
        print(f'  [B] 背景音乐       : {bgm_status}')
        speed_mode = '固定速度' if params['fixed_speed'] else '随BPM变速'
        print(f'  [S] 速度模式       : {speed_mode}')
        if params['fixed_speed']:
            min_interval_str = f'{params["min_note_interval"]:.0f} px' if params['min_note_interval'] > 0 else '关闭'
            print(f'  [M] 最小音符间隔   : {min_interval_str}  (仅固定速度模式)')
        print()
        print('  [0] 开始生成视频')
        print('  [Q] 退出')
        print()

        choice = input('请选择 → ').strip().lower()
        if choice == '0':
            break
        elif choice == 'q':
            print('\n👋 已取消')
            sys.exit(0)
        elif choice == '1':
            params['base_speed'] = input_float('   基准速度 (px/s): ', params['base_speed'])
        elif choice == '2':
            params['speed_mult'] = input_float('   变速倍率: ', params['speed_mult'])
        elif choice == '3':
            params['fps'] = input_float('   帧率 (fps): ', params['fps'])
        elif choice == '4':
            params['width'] = input_int('   视频宽度 (px): ', params['width'])
        elif choice == '5':
            params['track_height'] = input_int('   轨道高度 (px): ', params['track_height'])
        elif choice == 'b':
            # 背景音乐设置
            if params['bgm_enabled']:
                ans = input('  背景音乐已开启，是否关闭并重新选择？ (Y/n): ').strip().lower()
                if ans in ('', 'y', 'yes'):
                    params['bgm_enabled'] = False
                    params['bgm_path'] = ''
                    print('  已关闭背景音乐。')
                else:
                    audio_files = scan_audio_files(os.path.dirname(adofai_path))
                    if audio_files:
                        selected = select_audio_file(audio_files)
                        if selected:
                            if not ffmpeg_available():
                                print('  ⚠️  FFmpeg 不可用，无法添加背景音乐。背景音乐已关闭。')
                                params['bgm_enabled'] = False
                                params['bgm_path'] = ''
                            else:
                                params['bgm_path'] = selected
                                print(f'  已选择: {os.path.basename(selected)}')
            else:
                ans = input('  是否启用背景音乐？ (y/N): ').strip().lower()
                if ans in ('y', 'yes'):
                    if not ffmpeg_available():
                        print('  ⚠️  FFmpeg 不可用，无法添加背景音乐。')
                        input('按回车继续...')
                        continue
                    audio_files = scan_audio_files(os.path.dirname(adofai_path))
                    if not audio_files:
                        print('  未找到任何音频文件 (ogg, wav, aiff, mp3, flac)。')
                        input('按回车继续...')
                        continue
                    selected = select_audio_file(audio_files)
                    if selected:
                        params['bgm_enabled'] = True
                        params['bgm_path'] = selected
                        print(f'  已启用背景音乐: {os.path.basename(selected)}')
            input('按回车继续...')
        elif choice == 's':
            params['fixed_speed'] = not params['fixed_speed']
            mode = '固定速度' if params['fixed_speed'] else '随BPM变速'
            print(f'  速度模式已切换为: {mode}')
            if not params['fixed_speed']:
                params['min_note_interval'] = 0.0
            input('按回车继续...')
        elif choice == 'm':
            if params['fixed_speed']:
                val = input_float('   最小音符间隔 (px, 0=关闭): ', params['min_note_interval'])
                params['min_note_interval'] = max(0.0, val)
                if params['min_note_interval'] > 0:
                    print(f'  最小音符间隔已设为: {params["min_note_interval"]:.0f} px')
                else:
                    print('  最小音符间隔已关闭')
            else:
                print('  ⚠️  此选项仅在固定速度模式下可用')
            input('按回车继续...')
        else:
            print('⚠️  无效选项')
            input('按回车继续...')


    # ── 4. 生成视频 ──
    try:
        gen = TaikoAdofaiGenerator(
            adofai_path=adofai_path,
            base_speed=params['base_speed'],
            speed_mult=params['speed_mult'],
            screen_w=params['width'],
            fps=params['fps'],
            track_height=params['track_height'],
            use_gpu=True,
            bgm_path=params['bgm_path'] if params['bgm_enabled'] else None,
            fixed_speed=params['fixed_speed'],
            min_note_interval=params['min_note_interval'],
        )
        gen.generate_video(output_path)
    except Exception as e:
        print(f'\n❌ 错误: {e}')
        import traceback
        traceback.print_exc()
        input('\n按回车退出...')
        sys.exit(1)

    input('\n按回车退出...')


if __name__ == '__main__':
    main()
