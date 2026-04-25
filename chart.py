"""
ADOFAI -> 太鼓达人风格视频生成器 - 谱面数据模型与解析  [v3]
- Tile 数据模型：音符/瓦片的核心属性与状态更新
- load_adofai_tiles：读取 .adofai 谱面文件并构建 Tile 序列

v3 变更:
  [P0-1] 强制 orjson —— 去掉回退判断，直接 import orjson
  [P0-2] str.translate 替代 re.sub 清洗控制字符
  [新增] tqdm 进度条 —— 文件读取、清洗、解析、构建均有进度反馈
"""

import re
from pathlib import Path
from typing import List, Tuple

import orjson
from tqdm import tqdm

from config import CONFIG, PATH_MAP


# [P0-2] 预构建控制字符翻译表（模块级常量，只创建一次）
# 等价于 re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)
# 保留: \x09(tab) \x0a(lf) \x0d(cr)
_CTRL_CHARS_TABLE = str.maketrans(
    '', '',
    '\x00\x01\x02\x03\x04\x05\x06\x07\x08'
    '\x0b\x0c'
    '\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f'
)


class Tile:
    """谱面中单个瓦片/音符的数据模型"""
    __slots__ = ('angle', 'bpm', 'stdbpm', 'bpmangle', 'pause',
                 'offset', 'beat', 'volume', 'twirl', 'midspin', 'hold', 'cw')

    def __init__(self, angle: float = 0.0):
        self.angle = angle
        self.bpm = -1.0
        self.stdbpm = -1.0
        self.bpmangle = 0.0
        self.pause = 0.0
        self.offset = 0.0
        self.beat = 0.0
        self.volume = -1.0
        self.twirl = False
        self.midspin = False
        self.hold = False
        self.cw = 1

    def update(self, prev: 'Tile' = None) -> None:
        """根据前一个 Tile 计算当前 Tile 的完整状态（BPM、偏移、方向等）"""
        if prev is None:
            if self.stdbpm < 0: self.stdbpm = 100.0
            if self.bpm < 0: self.bpm = self.stdbpm
            self.cw = 1 ^ (1 if self.twirl else 0)
            self.offset = 0.0
            self.beat = 0.0
            if self.volume < 0: self.volume = 100.0
            return

        if self.angle == 999.0:
            self.midspin = True
            self.angle = prev.angle - 180.0

        da = 180.0 - self.angle + prev.angle
        if da >= 360: da -= 360
        elif da < 0: da += 360

        self.cw = prev.cw ^ (1 if self.twirl else 0)
        if self.cw:
            ao = 360.0 if (da == 0 and not self.midspin) else da
        else:
            ao = 0.0 if self.midspin else (360.0 - da)

        if self.stdbpm < 0 and prev.stdbpm > 0:
            self.stdbpm = -self.stdbpm * prev.stdbpm
        elif self.stdbpm < 0:
            self.stdbpm = prev.stdbpm

        if self.bpmangle > 0 and ao > 0:
            self.bpm = (self.stdbpm * (ao - self.bpmangle) + prev.stdbpm * self.bpmangle) / ao
        else:
            self.bpm = self.stdbpm

        beat_len = 60.0 / self.bpm
        self.offset = prev.offset + (ao / 180.0 + self.pause) * beat_len
        self.beat = prev.beat + ao / 180.0 + self.pause

        if self.volume < 0: self.volume = prev.volume


def _format_size(size_bytes: float) -> str:
    """字节数转人类可读格式"""
    if size_bytes < 1024:
        return f'{size_bytes:.0f} B'
    elif size_bytes < 1024 * 1024:
        return f'{size_bytes / 1024:.1f} KB'
    else:
        return f'{size_bytes / (1024 * 1024):.1f} MB'


def load_adofai_tiles(path: str) -> Tuple[List[Tile], float]:
    """
    读取谱面文件，返回 (tiles列表, offset_ms)
    其中 offset_ms 是谱面 settings 中的 offset，单位毫秒
    """
    file_size = Path(path).stat().st_size
    print(f'📂 读取谱面: {Path(path).name} ({_format_size(file_size)})')

    # ── 步骤1: 读取文件 ──
    with tqdm(total=4, desc='   解析谱面', bar_format='{desc} |{bar}| {n_fmt}/{total_fmt} 步') as pbar:
        pbar.set_postfix_str('读取文件')
        raw = Path(path).read_text(encoding='utf-8-sig')
        pbar.update(1)

        # ── 步骤2: 清洗控制字符 + 尾部逗号 ──
        pbar.set_postfix_str('清洗字符')
        cleaned = raw.translate(_CTRL_CHARS_TABLE)
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
        pbar.update(1)
        del raw  # 释放原始字符串内存

        # ── 步骤3: orjson 解析 ──
        pbar.set_postfix_str('JSON 解析')
        try:
            doc = orjson.loads(cleaned)
        except Exception as e:
            print(f'\n❌ JSON 解析失败！(orjson)')
            print(f'   错误描述: {e}')
            raise
        pbar.update(1)
        del cleaned  # 释放清洗后字符串内存

        # ── 步骤4: 构建 Tile 序列 ──
        pbar.set_postfix_str('构建音符')
        if 'angleData' in doc and isinstance(doc['angleData'], list):
            angles = [float(v) for v in doc['angleData']]
        elif 'pathData' in doc and isinstance(doc['pathData'], str):
            angles = [PATH_MAP.get(c, 0.0) for c in doc['pathData']]
        else:
            raise ValueError('谱面缺少 angleData 或 pathData')

        tiles = [Tile()] + [Tile(a) for a in angles]

        settings = doc.get('settings', {})
        if settings:
            tiles[0].stdbpm = float(settings.get('bpm', 100))
            tiles[0].volume = float(settings.get('volume', 100))

        offset_ms = float(settings.get('offset', 0))

        for act in doc.get('actions', []):
            floor = act.get('floor')
            event = act.get('eventType')
            if floor is None or event is None:
                continue
            idx = floor + 1
            if idx < 1 or idx >= len(tiles):
                continue
            tile = tiles[idx]

            if event == 'SetSpeed':
                st = act.get('speedType', '')
                if st == 'Bpm':
                    tile.stdbpm = float(act['beatsPerMinute'])
                elif 'bpmMultiplier' in act:
                    tile.stdbpm = -float(act['bpmMultiplier'])
                if 'angleOffset' in act:
                    tile.bpmangle = float(act['angleOffset'])
            elif event == 'Twirl':
                tile.twirl = True
            elif event == 'Pause':
                if 'duration' in act:
                    tile.pause = float(act['duration'])
            elif event == 'Hold':
                tile.hold = True
                if 'duration' in act:
                    tile.pause += float(act['duration']) * 2

        prev = None
        for t in tiles:
            t.update(prev)
            prev = t

        pbar.update(1)

    print(f'   ✅ 解析完成: {len(tiles)} 个瓦片, {len(angles)} 个音符, offset={offset_ms:.0f}ms')
    return tiles, offset_ms
"""
ADOFAI -> 太鼓达人风格视频生成器 - 谱面数据模型与解析  [v3]
- Tile 数据模型：音符/瓦片的核心属性与状态更新
- load_adofai_tiles：读取 .adofai 谱面文件并构建 Tile 序列

v3 变更:
  [P0-1] 强制 orjson —— 去掉回退判断，直接 import orjson
  [P0-2] str.translate 替代 re.sub 清洗控制字符
  [新增] tqdm 进度条 —— 文件读取、清洗、解析、构建均有进度反馈
"""

import re
from pathlib import Path
from typing import List, Tuple

import orjson
from tqdm import tqdm

from config import CONFIG, PATH_MAP


# [P0-2] 预构建控制字符翻译表（模块级常量，只创建一次）
# 等价于 re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)
# 保留: \x09(tab) \x0a(lf) \x0d(cr)
_CTRL_CHARS_TABLE = str.maketrans(
    '', '',
    '\x00\x01\x02\x03\x04\x05\x06\x07\x08'
    '\x0b\x0c'
    '\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f'
)


class Tile:
    """谱面中单个瓦片/音符的数据模型"""
    __slots__ = ('angle', 'bpm', 'stdbpm', 'bpmangle', 'pause',
                 'offset', 'beat', 'volume', 'twirl', 'midspin', 'hold', 'cw')

    def __init__(self, angle: float = 0.0):
        self.angle = angle
        self.bpm = -1.0
        self.stdbpm = -1.0
        self.bpmangle = 0.0
        self.pause = 0.0
        self.offset = 0.0
        self.beat = 0.0
        self.volume = -1.0
        self.twirl = False
        self.midspin = False
        self.hold = False
        self.cw = 1

    def update(self, prev: 'Tile' = None) -> None:
        """根据前一个 Tile 计算当前 Tile 的完整状态（BPM、偏移、方向等）"""
        if prev is None:
            if self.stdbpm < 0: self.stdbpm = 100.0
            if self.bpm < 0: self.bpm = self.stdbpm
            self.cw = 1 ^ (1 if self.twirl else 0)
            self.offset = 0.0
            self.beat = 0.0
            if self.volume < 0: self.volume = 100.0
            return

        if self.angle == 999.0:
            self.midspin = True
            self.angle = prev.angle - 180.0

        da = 180.0 - self.angle + prev.angle
        if da >= 360: da -= 360
        elif da < 0: da += 360

        self.cw = prev.cw ^ (1 if self.twirl else 0)
        if self.cw:
            ao = 360.0 if (da == 0 and not self.midspin) else da
        else:
            ao = 0.0 if self.midspin else (360.0 - da)

        if self.stdbpm < 0 and prev.stdbpm > 0:
            self.stdbpm = -self.stdbpm * prev.stdbpm
        elif self.stdbpm < 0:
            self.stdbpm = prev.stdbpm

        if self.bpmangle > 0 and ao > 0:
            self.bpm = (self.stdbpm * (ao - self.bpmangle) + prev.stdbpm * self.bpmangle) / ao
        else:
            self.bpm = self.stdbpm

        beat_len = 60.0 / self.bpm
        self.offset = prev.offset + (ao / 180.0 + self.pause) * beat_len
        self.beat = prev.beat + ao / 180.0 + self.pause

        if self.volume < 0: self.volume = prev.volume


def _format_size(size_bytes: float) -> str:
    """字节数转人类可读格式"""
    if size_bytes < 1024:
        return f'{size_bytes:.0f} B'
    elif size_bytes < 1024 * 1024:
        return f'{size_bytes / 1024:.1f} KB'
    else:
        return f'{size_bytes / (1024 * 1024):.1f} MB'


def load_adofai_tiles(path: str) -> Tuple[List[Tile], float]:
    """
    读取谱面文件，返回 (tiles列表, offset_ms)
    其中 offset_ms 是谱面 settings 中的 offset，单位毫秒
    """
    file_size = Path(path).stat().st_size
    print(f'📂 读取谱面: {Path(path).name} ({_format_size(file_size)})')

    # ── 步骤1: 读取文件 ──
    with tqdm(total=4, desc='   解析谱面', bar_format='{desc} |{bar}| {n_fmt}/{total_fmt} 步') as pbar:
        pbar.set_postfix_str('读取文件')
        raw = Path(path).read_text(encoding='utf-8-sig')
        pbar.update(1)

        # ── 步骤2: 清洗控制字符 + 尾部逗号 ──
        pbar.set_postfix_str('清洗字符')
        cleaned = raw.translate(_CTRL_CHARS_TABLE)
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
        pbar.update(1)
        del raw  # 释放原始字符串内存

        # ── 步骤3: orjson 解析 ──
        pbar.set_postfix_str('JSON 解析')
        try:
            doc = orjson.loads(cleaned)
        except Exception as e:
            print(f'\n❌ JSON 解析失败！(orjson)')
            print(f'   错误描述: {e}')
            raise
        pbar.update(1)
        del cleaned  # 释放清洗后字符串内存

        # ── 步骤4: 构建 Tile 序列 ──
        pbar.set_postfix_str('构建音符')
        if 'angleData' in doc and isinstance(doc['angleData'], list):
            angles = [float(v) for v in doc['angleData']]
        elif 'pathData' in doc and isinstance(doc['pathData'], str):
            angles = [PATH_MAP.get(c, 0.0) for c in doc['pathData']]
        else:
            raise ValueError('谱面缺少 angleData 或 pathData')

        tiles = [Tile()] + [Tile(a) for a in angles]

        settings = doc.get('settings', {})
        if settings:
            tiles[0].stdbpm = float(settings.get('bpm', 100))
            tiles[0].volume = float(settings.get('volume', 100))

        offset_ms = float(settings.get('offset', 0))

        for act in doc.get('actions', []):
            floor = act.get('floor')
            event = act.get('eventType')
            if floor is None or event is None:
                continue
            idx = floor + 1
            if idx < 1 or idx >= len(tiles):
                continue
            tile = tiles[idx]

            if event == 'SetSpeed':
                st = act.get('speedType', '')
                if st == 'Bpm':
                    tile.stdbpm = float(act['beatsPerMinute'])
                elif 'bpmMultiplier' in act:
                    tile.stdbpm = -float(act['bpmMultiplier'])
                if 'angleOffset' in act:
                    tile.bpmangle = float(act['angleOffset'])
            elif event == 'Twirl':
                tile.twirl = True
            elif event == 'Pause':
                if 'duration' in act:
                    tile.pause = float(act['duration'])
            elif event == 'Hold':
                tile.hold = True
                if 'duration' in act:
                    tile.pause += float(act['duration']) * 2

        prev = None
        for t in tiles:
            t.update(prev)
            prev = t

        pbar.update(1)

    print(f'   ✅ 解析完成: {len(tiles)} 个瓦片, {len(angles)} 个音符, offset={offset_ms:.0f}ms')
    return tiles, offset_ms
