#!/usr/bin/env python3
"""
EPUB → XTC 変換ツール
XTEINK X4/X3 リーダー向けに最適化された XTC 形式に変換

XTC 形式仕様：
- ヘッダー：56 バイト
- マジック：0x00435458 (XTC), 0x0048435458 (XTCH)
- リトルエンディアン
- ページ：480x800, 1 ビット (XTC) または 2 ビット (XTCH)
"""

import os
import sys
import struct
import zipfile
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image, ImageOps
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

console = Console()

# XTEINK X4/X3 ディスプレイ仕様
DISPLAY_WIDTH = 528
DISPLAY_HEIGHT = 792

# XTC フォーマット定数
# ファイル上に並ぶバイト順序で定義（リトルエンディアンで pack される）
# "XTC\0" = 'X','T','C',0x00
# "XTCH" = 'H','T','C','X' (little-endian: 0x48435458)
# "XTG\0" = 'X','T','G',0x00
# "XTH\0" = 'H','T','H','X' (little-endian: 0x00485458)
XTC_MAGIC = 0x00435458      # "XTC\0"
XTCH_MAGIC = 0x48435458     # "XTCH"
XTG_MAGIC = 0x00475458      # "XTG\0"
XTH_MAGIC = 0x00485458      # "XTH\0"

# ヘッダー構造
HEADER_SIZE = 48      # サンプルファイルに基づく
METADATA_SIZE = 240   # サンプルに基づく (128+112)
PAGE_INDEX_ENTRY_SIZE = 16


def fnv1a_hash(data: bytes) -> int:
    """FNV-1a 32-bit hash"""
    h = 0x811c9dc5  # FNV offset basis
    for byte in data:
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF  # FNV prime
    return h
PAGE_HEADER_SIZE = 22


def extract_epub(epub_path: str, extract_dir: str) -> str:
    """EPUB ファイルを解凍"""
    epub_path = Path(epub_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(epub_path, 'r') as zf:
        zf.extractall(extract_dir)

    return str(extract_dir)


def parse_opf(opf_path: str) -> dict:
    """content.opf からメタデータを抽出"""
    import xml.etree.ElementTree as ET

    tree = ET.parse(opf_path)
    root = tree.getroot()

    metadata = {
        'title': 'Unknown',
        'author': 'Unknown',
        'language': 'ja',
        'identifier': ''
    }

    for elem in root.iter():
        tag = elem.tag.split('}')[-1].lower() if '}' in elem.tag else elem.tag.lower()
        if 'title' in tag:
            metadata['title'] = elem.text or 'Unknown'
        if 'creator' in tag or 'author' in tag:
            metadata['author'] = elem.text or 'Unknown'
        if 'identifier' in tag:
            metadata['identifier'] = elem.text or ''

    return metadata


def apply_extreme_contrast(img, pixels, width, height, t1, t2, t3, brightness=100):
    """
    極限コントラスト処理（共通関数）

    Parameters:
    - t1: 黒閾値（以下は黒）
    - t2: 暗中間閾値（t1-t2 は暗中間）
    - t3: 明中間閾値（t2-t3 は明中間、以上は白）
    - brightness: 明るさ調整（60-100%、100 が標準）
    """
    from PIL import ImageEnhance

    # 明るさ調整を閾値変換前に適用
    if brightness != 100:
        factor = brightness / 100
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)

    pixels = img.load()

    # 閾値で変換
    lut = []
    for x in range(256):
        if x < t1:
            val = 0
        elif x < t2:
            val = int((x - t1) * 85 / (t2 - t1)) if t2 > t1 else 0
        elif x < t3:
            val = 85 + int((x - t2) * 85 / (t3 - t2)) if t3 > t2 else 85
        else:
            val = 170 + int((x - t3) * 85 / (255 - t3)) if t3 < 255 else 255
        lut.append(val)
    img = img.point(lut)

    # 2. 誤差拡散（Floyd-Steinberg）
    pixels = img.load()
    levels = [0, 85, 170, 255]

    def quantize_2bit(val):
        min_dist = 256
        result = 0
        for i, level in enumerate(levels):
            dist = abs(val - level)
            if dist < min_dist:
                min_dist = dist
                result = i * 85
        return result

    for y in range(height):
        for x in range(width):
            old = pixels[x, y]
            new = quantize_2bit(old)
            pixels[x, y] = new
            error = old - new

            # 誤差拡散（Floyd-Steinberg）
            if x + 1 < width:
                pixels[x + 1, y] = max(0, min(255, pixels[x + 1, y] + error * 7 // 16))
            if y + 1 < height:
                if x > 0:
                    pixels[x - 1, y + 1] = max(0, min(255, pixels[x - 1, y + 1] + error * 3 // 16))
                pixels[x, y + 1] = max(0, min(255, pixels[x, y + 1] + error * 5 // 16))
                if x + 1 < width:
                    pixels[x + 1, y + 1] = max(0, min(255, pixels[x + 1, y + 1] + error * 1 // 16))

    return img


def apply_dithering_2bit(img: Image.Image, method: str = 'none',
                          enhance_contrast: bool = True,
                          text_mode: bool = False,
                          morph_dilation: int = 0,
                          dark_mode: bool = False,
                          brightness: int = 90) -> Image.Image:
    """
    2 ビット（4 階調）用ディザリング

    Methods:
    - 'none': なし（単純量子化）
    - 'floyd': Floyd-Steinberg（誤差拡散・高画質）
    - 'sierra_lite': Sierra Lite（軽量・高速）
    - 'atkinson': Atkinson（自然な仕上がり）
    - 'text': テキスト最適化（文字を濃く表示）
    - 'manga': 漫画最適化（黒髪は黒く、中間トーン維持）
    - 'sharp': 輪郭強調＋コントラスト＋誤差拡散制限（文字くっきり高画質）
    - 'light': 白寄せモード（肌色・明るい色を明るく維持）
    - 'extreme': 極限コントラスト（40/60/70・デフォルト）
    - 'extreme_40_45_50': 極限コントラスト（40/45/50・肌色・白寄せ）
    - 'extreme_45_55_65': 極限コントラスト（45/55/65・狭め）
    - 'extreme_50_55_60': 極限コントラスト（50/55/60・さらに狭め）
    - 'extreme_55_65_75': 極限コントラスト（55/65/75・広め）

    Parameters:
    - morph_dilation: モルフォロジー膨張処理の半径（0=なし、1-2=推奨）
    - dark_mode: 画像を暗くする（-15% 輝度、少年漫画雑誌風）
    - brightness: 明るさ調整（60-100%、100 が標準、90 がおすすめ）
    """
    from PIL import ImageFilter

    img = img.convert('L').copy()
    pixels = img.load()
    width, height = img.size

    # 極限コントラストモード：中間を黒白に強制分割
    if method == 'extreme':
        # デフォルト閾値：40/60/70
        return apply_extreme_contrast(img, pixels, width, height, 40, 60, 70, brightness)

    # 極限コントラストモード・閾値バリエーション
    if method.startswith('extreme_'):
        # extreme_45_55_65, extreme_50_60_70 形式
        parts = method.split('_')
        if len(parts) == 3:
            t1 = int(parts[1])
            t2 = int(parts[2])
            t3 = int(parts[3]) if len(parts) > 3 else t2 + 10
            return apply_extreme_contrast(img, pixels, width, height, t1, t2, t3, brightness)
        else:
            # 旧形式：extreme_45 → 45/65/75
            threshold = int(parts[1])
            return apply_extreme_contrast(img, pixels, width, height, threshold, threshold+20, threshold+30, brightness)

    # 白寄せモード：明るい色を明るく維持
    if method == 'light':
        # 1. 明るい部分を引き上げる
        lut = []
        for x in range(256):
            if x < 64:
                # 暗い部分：そのまま
                val = x
            elif x < 192:
                # 中間：少し明るく
                val = 64 + int(128 * ((x - 64) / 128) ** 0.85)
            else:
                # 明るい部分：もっと明るく（255 に近づける）
                val = 192 + int(63 * ((x - 192) / 63) ** 0.7)
            lut.append(val)
        img = img.point(lut)

        # 2. 標準的な誤差拡散
        pixels = img.load()
        levels = [0, 85, 170, 255]

        def quantize_2bit(val):
            min_dist = 256
            result = 0
            for i, level in enumerate(levels):
                dist = abs(val - level)
                if dist < min_dist:
                    min_dist = dist
                    result = i * 85
            return result

        for y in range(height):
            for x in range(width):
                old = pixels[x, y]
                new = quantize_2bit(old)
                pixels[x, y] = new
                error = old - new

                # 誤差拡散（Atkinson）
                if x + 1 < width:
                    pixels[x + 1, y] = max(0, min(255, pixels[x + 1, y] + error // 8))
                if x + 2 < width:
                    pixels[x + 2, y] = max(0, min(255, pixels[x + 2, y] + error // 8))
                if y + 1 < height:
                    if x > 0:
                        pixels[x - 1, y + 1] = max(0, min(255, pixels[x - 1, y + 1] + error // 8))
                    pixels[x, y + 1] = max(0, min(255, pixels[x, y + 1] + error // 8))
                    if x + 1 < width:
                        pixels[x + 1, y + 1] = max(0, min(255, pixels[x + 1, y + 1] + error // 8))

        return img

    # 漫画モード：黒を強調して髪をちゃんと黒く表示
    if method == 'manga':
        # 1. 単純なコントラスト調整（線形）
        # 暗い部分は黒く、明るい部分は維持
        lut = []
        for x in range(256):
            if x < 64:
                # 暗い部分：少し黒に近づける（64→0 に）
                val = int(x * 64 / 64) - int((64 - x) * 0.3)
                val = max(0, val)
            elif x < 192:
                # 中間：そのまま
                val = x
            else:
                # 明るい部分：そのまま
                val = x
            lut.append(val)
        img = img.point(lut)

        # 2. 誤差拡散付き量子化（標準的な Atkinson）
        pixels = img.load()
        levels = [0, 85, 170, 255]  # 標準レベル

        def quantize_2bit(val):
            min_dist = 256
            result = 0
            for i, level in enumerate(levels):
                dist = abs(val - level)
                if dist < min_dist:
                    min_dist = dist
                    result = i * 85
            return result

        for y in range(height):
            for x in range(width):
                old = pixels[x, y]
                new = quantize_2bit(old)
                pixels[x, y] = new
                error = old - new

                # 誤差拡散（Atkinson と同じ）
                if x + 1 < width:
                    pixels[x + 1, y] = max(0, min(255, pixels[x + 1, y] + error // 8))
                if x + 2 < width:
                    pixels[x + 2, y] = max(0, min(255, pixels[x + 2, y] + error // 8))
                if y + 1 < height:
                    if x > 0:
                        pixels[x - 1, y + 1] = max(0, min(255, pixels[x - 1, y + 1] + error // 8))
                    pixels[x, y + 1] = max(0, min(255, pixels[x, y + 1] + error // 8))
                    if x + 1 < width:
                        pixels[x + 1, y + 1] = max(0, min(255, pixels[x + 1, y + 1] + error // 8))

        return img

    # sharp モード：コントラスト調整＋誤差拡散制限（シャープなし）
    if method == 'sharp':
        # 1. 緩やかなコントラストカーブ（潰れ防止）
        # 暗いは少し暗く、明るいところは少し明るく
        lut = []
        for x in range(256):
            if x < 80:
                # 暗い側：少し暗くするだけ（潰れ防止）
                val = int(80 * (x / 80) ** 1.1)
            elif x > 180:
                # 明るい側：少し明るく
                val = 180 + int(75 * ((x - 180) / 75) ** 0.9)
            else:
                # 中間：線形
                val = x
            lut.append(val)
        img = img.point(lut)

        # 2. 誤差拡散制限付き量子化
        pixels = img.load()
        levels = [0, 85, 170, 255]  # 標準レベル

        def quantize_2bit(val):
            min_dist = 256
            result = 0
            for i, level in enumerate(levels):
                dist = abs(val - level)
                if dist < min_dist:
                    min_dist = dist
                    result = i * 85
            return result

        for y in range(height):
            for x in range(width):
                old = pixels[x, y]
                new = quantize_2bit(old)
                pixels[x, y] = new
                error = old - new

                # 誤差拡散制限：暗いピクセル（新値が 0）からの拡散を 50% に抑制
                diffusion_rate = 0.5 if new == 0 else 1.0

                if x + 1 < width:
                    pixels[x + 1, y] = max(0, min(255, pixels[x + 1, y] + int(error // 8 * diffusion_rate)))
                if x + 2 < width:
                    pixels[x + 2, y] = max(0, min(255, pixels[x + 2, y] + int(error // 8 * diffusion_rate)))
                if y + 1 < height:
                    if x > 0:
                        pixels[x - 1, y + 1] = max(0, min(255, pixels[x - 1, y + 1] + int(error // 8 * diffusion_rate)))
                    pixels[x, y + 1] = max(0, min(255, pixels[x, y + 1] + int(error // 8 * diffusion_rate)))
                    if x + 1 < width:
                        pixels[x + 1, y + 1] = max(0, min(255, pixels[x + 1, y + 1] + int(error // 8 * diffusion_rate)))

        return img

    # テキストモード：コントラスト強調
    if text_mode:
        # アンシャープマスクで輪郭強調
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=0))

    # モルフォロジー膨張処理（文字を太くする）
    if morph_dilation > 0:
        # グレースケールで膨張処理（明部を拡大）
        for _ in range(morph_dilation):
            img = img.filter(ImageFilter.MaxFilter(3))

    # コントラスト強調（テキストモードはさらに強調）
    if enhance_contrast or text_mode:
        img = img.filter(ImageFilter.EDGE_ENHANCE)

    # 最適化された 4 階調 LUT（e-ink 向けに補正）
    if text_mode:
        levels = [0, 60, 140, 255]  # テキスト用・暗めシフト
    elif enhance_contrast:
        levels = [0, 85, 170, 255]  # デフォルト
    else:
        levels = [0, 85, 170, 255]  # デフォルト

    def quantize_2bit(val):
        """最も近い 4 階調レベルに量子化"""
        min_dist = 256
        result = 0
        for i, level in enumerate(levels):
            dist = abs(val - level)
            if dist < min_dist:
                min_dist = dist
                result = i * 85  # 出力は 0, 85, 170, 255
        return result

    if method == 'none':
        for y in range(height):
            for x in range(width):
                pixels[x, y] = quantize_2bit(pixels[x, y])

    elif method == 'floyd':
        # Floyd-Steinberg for 4-level (same matrix, different quantization)
        for y in range(height):
            for x in range(width):
                old = pixels[x, y]
                new = quantize_2bit(old)
                pixels[x, y] = new
                error = old - new
                if x + 1 < width:
                    pixels[x + 1, y] = max(0, min(255, pixels[x + 1, y] + error * 7 // 16))
                if y + 1 < height:
                    if x > 0:
                        pixels[x - 1, y + 1] = max(0, min(255, pixels[x - 1, y + 1] + error * 3 // 16))
                    pixels[x, y + 1] = max(0, min(255, pixels[x, y + 1] + error * 5 // 16))
                    if x + 1 < width:
                        pixels[x + 1, y + 1] = max(0, min(255, pixels[x + 1, y + 1] + error * 1 // 16))

    elif method == 'sierra_lite':
        # Sierra Lite: 軽量誤差拡散 (1/4)
        # 右：2/4, 左下：1/4, 下：1/4
        for y in range(height):
            for x in range(width):
                old = pixels[x, y]
                new = quantize_2bit(old)
                pixels[x, y] = new
                error = old - new
                if x + 1 < width:
                    pixels[x + 1, y] = max(0, min(255, pixels[x + 1, y] + error * 2 // 4))
                if y + 1 < height:
                    if x > 0:
                        pixels[x - 1, y + 1] = max(0, min(255, pixels[x - 1, y + 1] + error // 4))
                    pixels[x, y + 1] = max(0, min(255, pixels[x, y + 1] + error // 4))

    elif method == 'atkinson':
        # Atkinson for 4-level
        for y in range(height):
            for x in range(width):
                old = pixels[x, y]
                new = quantize_2bit(old)
                pixels[x, y] = new
                error = old - new
                if x + 1 < width:
                    pixels[x + 1, y] = max(0, min(255, pixels[x + 1, y] + error // 8))
                if x + 2 < width:
                    pixels[x + 2, y] = max(0, min(255, pixels[x + 2, y] + error // 8))
                if y + 1 < height:
                    if x > 0:
                        pixels[x - 1, y + 1] = max(0, min(255, pixels[x - 1, y + 1] + error // 8))
                    pixels[x, y + 1] = max(0, min(255, pixels[x, y + 1] + error // 8))
                    if x + 1 < width:
                        pixels[x + 1, y + 1] = max(0, min(255, pixels[x + 1, y + 1] + error // 8))

    elif method == 'text':
        # Text optimized: minimal error diffusion, darker output
        # 誤差拡散を最小限にして文字の輪郭を維持
        for y in range(height):
            for x in range(width):
                old = pixels[x, y]
                new = quantize_2bit(old)
                pixels[x, y] = new
                error = old - new
                # 誤差拡散は 1/8 のみ（輪郭ぼかし防止）
                if x + 1 < width:
                    pixels[x + 1, y] = max(0, min(255, pixels[x + 1, y] + error // 8))
                if y + 1 < height:
                    pixels[x, y + 1] = max(0, min(255, pixels[x, y + 1] + error // 8))

    # dark_mode: 画像を暗くする（少年漫画雑誌風）
    if dark_mode:
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.85)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)

    return img


def apply_dithering(img: Image.Image, method: str = 'none') -> Image.Image:
    """
    ディザリングを適用

    Methods:
    - 'none': なし（単純量子化）
    - 'floyd': Floyd-Steinberg（誤差拡散）
    - 'stucki': Stucki（誤差拡散）
    - 'atkinson': Atkinson（誤差拡散）
    - 'bayer': Bayer 順序ディザ
    """
    if method == 'none':
        return img

    img = img.convert('L').copy()
    pixels = img.load()
    width, height = img.size

    if method == 'floyd':
        # Floyd-Steinberg: 右・左下・下・右下 = 7/16, 3/16, 5/16, 1/16
        for y in range(height):
            for x in range(width):
                old = pixels[x, y]
                new = 0 if old < 128 else 255
                pixels[x, y] = new
                error = old - new
                if x + 1 < width:
                    pixels[x + 1, y] = max(0, min(255, pixels[x + 1, y] + error * 7 // 16))
                if y + 1 < height:
                    if x > 0:
                        pixels[x - 1, y + 1] = max(0, min(255, pixels[x - 1, y + 1] + error * 3 // 16))
                    pixels[x, y + 1] = max(0, min(255, pixels[x, y + 1] + error * 5 // 16))
                    if x + 1 < width:
                        pixels[x + 1, y + 1] = max(0, min(255, pixels[x + 1, y + 1] + error * 1 // 16))

    elif method == 'stucki':
        # Stucki: 右=8, 右右=4, 左下=2, 下=4, 右下=2, 左下下=1, 下下=2, 右下下=1 /32
        for y in range(height):
            for x in range(width):
                old = pixels[x, y]
                new = 0 if old < 128 else 255
                pixels[x, y] = new
                error = old - new
                if x + 1 < width:
                    pixels[x + 1, y] = max(0, min(255, pixels[x + 1, y] + error * 8 // 32))
                if x + 2 < width:
                    pixels[x + 2, y] = max(0, min(255, pixels[x + 2, y] + error * 4 // 32))
                if y + 1 < height:
                    if x > 0:
                        pixels[x - 1, y + 1] = max(0, min(255, pixels[x - 1, y + 1] + error * 2 // 32))
                    pixels[x, y + 1] = max(0, min(255, pixels[x, y + 1] + error * 4 // 32))
                    if x + 1 < width:
                        pixels[x + 1, y + 1] = max(0, min(255, pixels[x + 1, y + 1] + error * 2 // 32))
                if y + 2 < height:
                    if x > 0:
                        pixels[x - 1, y + 2] = max(0, min(255, pixels[x - 1, y + 2] + error * 1 // 32))
                    pixels[x, y + 2] = max(0, min(255, pixels[x, y + 2] + error * 2 // 32))
                    if x + 1 < width:
                        pixels[x + 1, y + 2] = max(0, min(255, pixels[x + 1, y + 2] + error * 1 // 32))

    elif method == 'atkinson':
        # Atkinson: 右・左下・下・右下・右右 = 1/8
        for y in range(height):
            for x in range(width):
                old = pixels[x, y]
                new = 0 if old < 128 else 255
                pixels[x, y] = new
                error = old - new
                if x + 1 < width:
                    pixels[x + 1, y] = max(0, min(255, pixels[x + 1, y] + error // 8))
                if x + 2 < width:
                    pixels[x + 2, y] = max(0, min(255, pixels[x + 2, y] + error // 8))
                if y + 1 < height:
                    if x > 0:
                        pixels[x - 1, y + 1] = max(0, min(255, pixels[x - 1, y + 1] + error // 8))
                    pixels[x, y + 1] = max(0, min(255, pixels[x, y + 1] + error // 8))
                    if x + 1 < width:
                        pixels[x + 1, y + 1] = max(0, min(255, pixels[x + 1, y + 1] + error // 8))

    elif method == 'bayer':
        # Bayer 4x4 行列
        bayer = [
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5]
        ]
        for y in range(height):
            for x in range(width):
                threshold = (bayer[y % 4][x % 4] + 0.5) * 16
                pixels[x, y] = 0 if pixels[x, y] < threshold else 255

    return img


def render_page_to_image(img_path: str, width: int = DISPLAY_WIDTH,
                         height: int = DISPLAY_HEIGHT, bits: int = 2,
                         dither: str = 'none', dither_2bit: str = None,
                         enhance_contrast: bool = True, text_mode: bool = False,
                         morph_dilation: int = 0, brightness: int = 90) -> Image.Image:
    """
    画像ファイルを XTEINK 向けにレンダリング
    bits=1: モノクロ (1 ビット), bits=2: 4 階調グレースケール (2 ビット)
    dither: 1 ビット用ディザリング
    dither_2bit: 2 ビット用ディザリング（None なら dither と同じ）
    enhance_contrast: コントラスト強調（2 ビット用）
    text_mode: テキスト最適化モード（文字を濃く表示）
    brightness: 明るさ調整（60-100%、100 が標準、90 がおすすめ）
    """
    with Image.open(img_path) as img:
        # アスペクト比を維持してリサイズ
        img.thumbnail((width, height), Image.Resampling.LANCZOS)

        # グレースケール変換
        img = ImageOps.grayscale(img)

        # 明るさ調整（変換前に適用）
        if brightness != 100:
            from PIL import ImageEnhance
            factor = brightness / 100
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.1)

        # ディザリング適用
        if bits == 1:
            img = apply_dithering(img, dither)
            img = img.convert('1')

        elif bits == 2:
            # 2 ビット用ディザリング（指定なければデフォルトは Sierra Lite）
            if dither_2bit is None:
                dither_2bit = dither if dither != 'none' else 'sierra_lite'
            img = apply_dithering_2bit(img, dither_2bit, enhance_contrast, text_mode, morph_dilation, brightness=100)

        # 指定サイズに貼り付け（中央揃え、背景は白）
        result = Image.new('L' if bits == 2 else '1', (width, height), 255 if bits == 2 else 1)
        offset = ((width - img.width) // 2, (height - img.height) // 2)
        result.paste(img, offset)

        return result


def process_page_to_xtg(args) -> tuple:
    """
    個別ページを XTG/XTH フォーマットに処理（並列化用）

    Args:
        args: (i, img_path, bits, dither, dither_2bit, enhance_contrast, text_mode, morph_dilation, brightness)

    Returns:
        (i, page_data, success, error_message)
    """
    i, img_path_str, bits, dither, dither_2bit, enhance_contrast, text_mode, morph_dilation, brightness = args

    try:
        img = render_page_to_image(
            img_path_str,
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
            bits,
            dither,
            dither_2bit=dither_2bit,
            enhance_contrast=enhance_contrast,
            text_mode=text_mode,
            morph_dilation=morph_dilation,
            brightness=brightness
        )
        page_data = encode_xtg_page(img, bits)
        return (i, page_data, True, None)
    except Exception as e:
        return (i, None, False, str(e))


def encode_xtg_page(img: Image.Image, bits: int = 2) -> bytes:
    """
    画像を XTG/XTH フォーマットのページデータにエンコード

    XTG (1-bit): 行優先、8 ピクセル/バイト、MSB = 左ピクセル
    XTH (2-bit): 列優先、2 ビットプレーン構造、8 ピクセル/バイト

    XTEINK はビット解釈が逆：0=白，1=黒 (1-bit)、0=白，3=黒 (2-bit)

    戻り値：ページヘッダー (22 バイト) + ページデータ
    """
    width, height = img.size

    # ページデータ本体を生成
    image_data = bytearray()

    if bits == 1:
        # 1 ビット：行優先、8 ピクセル/バイト
        # XTEINK: 0=黒，1=白（標準ビットマップと同じ）
        for y in range(height):
            for x in range(0, width, 8):
                byte = 0
                for bit in range(8):
                    if x + bit < width:
                        pixel = img.getpixel((x + bit, y))
                        # pixel: 0=黒，255=白 → XTEINK: 0=黒，1=白
                        if pixel >= 128:  # 白
                            byte |= (0x80 >> bit)
                image_data.append(byte)

    elif bits == 2:
        # 2 ビット：列優先、2 ビットプレーン構造
        # 各ビットプレーン：8 ピクセル/バイト（縦方向）
        # p1 = 上位ビット，p2 = 下位ビット
        # 最終データ：p1 + p2（連結）
        bytes_per_col = (height + 7) // 8  # 800/8 = 100
        p1 = bytearray(width * bytes_per_col)
        p2 = bytearray(width * bytes_per_col)

        for x_idx, x in enumerate(range(width - 1, -1, -1)):  # 右から左
            for y in range(height):
                pixel = img.getpixel((x, y))
                # 0-255 -> 0-3 に変換後、反転（XTEINK は 0=白，3=黒）
                raw_value = pixel // 64  # 0-3
                bit_value = 3 - raw_value  # 反転

                bit1 = (bit_value >> 1) & 0x01  # 上位ビット
                bit2 = bit_value & 0x01          # 下位ビット

                target_byte = (x_idx * bytes_per_col) + (y // 8)
                bit_pos = 7 - (y % 8)

                if bit1:
                    p1[target_byte] |= (1 << bit_pos)
                if bit2:
                    p2[target_byte] |= (1 << bit_pos)

        image_data.extend(p1)
        image_data.extend(p2)

    # ページヘッダー (22 バイト)
    page_magic = 0x00475458 if bits == 1 else 0x00485458  # "XTG\0" or "XTH\0"
    total_size = 22 + len(image_data)  # ページヘッダーを含む合計サイズ

    page_header = bytearray()
    page_header.extend(struct.pack('<I', page_magic))  # 4: magic
    page_header.extend(struct.pack('<H', width))        # 2: width
    page_header.extend(struct.pack('<H', height))       # 2: height
    # [8:12]: sample.xtch は 0x00000077 (119 = 480/4 - 1)
    page_header.extend(b'\x00\x00\x00')
    page_header.append(width // 4 - 1)  # 119 for 480 width
    page_header.extend(struct.pack('<I', 1))            # 4: flags? (sample=1)
    page_header.extend(b'\x00' * 6)                     # 6: padding

    return bytes(page_header) + bytes(image_data)


def create_xtc_file(output_path: str, pages: list, title: str, author: str, bits: int = 2):
    """
    XTC ファイルを作成

    構造:
    - ヘッダー (48 バイト)
    - メタデータ (240 バイト): タイトル (128) + 著者 (112)
    - チャプター情報 (page_count * 112 バイト)
    - インデックステーブル (page_count * 16 バイト)
    - データエリア (各ページの XTG/XTH データ)
    """
    page_count = len(pages)

    # オフセット計算
    header_size = HEADER_SIZE              # 48
    metadata_offset = header_size           # 48
    metadata_size = 240                     # サンプルに基づく (128+112)
    chapter_offset = metadata_offset + metadata_size  # 288 = 0x120
    chapter_size = page_count * 112
    index_table_offset = chapter_offset + chapter_size
    index_table_size = page_count * PAGE_INDEX_ENTRY_SIZE
    data_offset = index_table_offset + index_table_size

    # 各ページのオフセットとサイズを計算
    page_offsets = []
    page_sizes = []
    current_offset = data_offset

    for page_data in pages:
        page_offsets.append(current_offset)
        page_sizes.append(len(page_data))
        current_offset += len(page_data)

    with open(output_path, 'wb') as f:
        # === ヘッダー (48 バイト) ===
        magic = XTCH_MAGIC if bits == 2 else XTC_MAGIC
        f.write(struct.pack('<I', magic))
        f.write(struct.pack('<H', 0x0100))  # version
        f.write(struct.pack('<H', page_count))
        f.write(struct.pack('<B', 0))  # readDirection
        f.write(struct.pack('<B', 1))  # hasMetadata
        f.write(struct.pack('<B', 0))  # hasThumbnails
        f.write(struct.pack('<B', 1))  # hasChapters
        f.write(struct.pack('<I', 0))  # currentPage
        f.write(struct.pack('<Q', metadata_offset))
        f.write(struct.pack('<Q', index_table_offset))
        f.write(struct.pack('<Q', data_offset))
        # 最後の 8 バイトは padding (thumbOffset/chapterOffset 相当)
        f.write(b'\x00' * 8)

        # === メタデータ (240 バイト) ===
        # タイトル (128 バイト)
        title_bytes = title.encode('utf-8')[:128]
        f.write(title_bytes)
        f.write(b'\x00' * (128 - len(title_bytes)))
        # 著者 (112 バイト)
        author_bytes = author.encode('utf-8')[:112]
        f.write(author_bytes)
        f.write(b'\x00' * (112 - len(author_bytes)))

        # === チャプター情報 (各ページ 112 バイト) ===
        for i in range(page_count):
            # hash/ID (4 バイト) - FNV-1a ハッシュ
            chapter_name = f"Page {i+1}"
            name_hash = fnv1a_hash(chapter_name.encode('utf-8'))
            f.write(struct.pack('<I', name_hash))
            # reserved (2 バイト)
            f.write(struct.pack('<H', 0))
            # page_count (2 バイト)
            f.write(struct.pack('<H', page_count))
            # reserved (8 バイト)
            f.write(b'\x00' * 8)
            # name (96 バイト)
            chapter_name_bytes = chapter_name.encode('utf-8')[:96]
            f.write(chapter_name_bytes)
            f.write(b'\x00' * (96 - len(chapter_name_bytes)))

        # === インデックステーブル (各ページ 16 バイト) ===
        for i, (offset, size) in enumerate(zip(page_offsets, page_sizes)):
            f.write(struct.pack('<Q', offset))
            f.write(struct.pack('<I', size))
            f.write(struct.pack('<H', DISPLAY_WIDTH))
            f.write(struct.pack('<H', DISPLAY_HEIGHT))

        # === データエリア ===
        for page_data in pages:
            f.write(page_data)

    console.print(f"[green]✓ ファイル作成完了：{output_path}[/green]")


def epub_to_xtc(epub_path: str, output_dir: str = None, bits: int = 2,
                dither: str = 'none', dither_2bit: str = None,
                enhance_contrast: bool = True, text_mode: bool = False,
                morph_dilation: int = 0, brightness: int = 90,
                max_workers: int = None) -> str:
    """
    EPUB を XTC 形式に変換

    Parameters:
    - brightness: 明るさ調整（60-100%、100 が標準、90 がおすすめ）
    """
    epub_path = Path(epub_path).resolve()

    if not epub_path.exists():
        raise FileNotFoundError(f"EPUB ファイルが見つかりません：{epub_path}")

    if output_dir is None:
        output_dir = Path.cwd() / "xtc"
    else:
        output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    ext = '.xtch' if bits == 2 else '.xtc'
    output_path = output_dir / f"{epub_path.stem}{ext}"

    console.print(f"[cyan]XTC 変換中：{epub_path.name}[/cyan]")

    # 一時ディレクトリに EPUB を解凍
    temp_dir = epub_path.parent / f".xtc_temp_{epub_path.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # EPUB 解凍
        extract_dir = extract_epub(epub_path, temp_dir / "epub_contents")

        # メタデータ抽出
        opf_files = list(Path(extract_dir).rglob("*.opf"))
        if opf_files:
            metadata = parse_opf(str(opf_files[0]))
        else:
            metadata = {'title': epub_path.stem, 'author': 'Unknown'}

        # 画像ディレクトリを探す
        # 1. 直下の images
        images_dir = Path(extract_dir) / "images"
        # 2. EPUB/images (標準的な EPUB 構造)
        if not images_dir.exists():
            epub_images = Path(extract_dir) / "EPUB" / "images"
            if epub_images.exists():
                images_dir = epub_images
        # 3. 再帰的に images を探す
        if not images_dir.exists():
            for d in Path(extract_dir).rglob("images"):
                if d.is_dir():
                    images_dir = d
                    break

        if not images_dir.exists():
            raise ValueError("画像ディレクトリが見つかりません")

        # 画像ファイル一覧を取得
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
        image_files = sorted([
            f for f in images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ])

        if not image_files:
            raise ValueError("画像ファイルが見つかりません")

        # CPU コア数に合わせて並列処理（デフォルト：コア数の 80%）
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)

        page_count = len(image_files)
        console.print(f"  [dim]{page_count} ページを{max_workers}スレッドで処理[/dim]")

        # 並列処理タスクを準備
        tasks = [
            (i, str(img_path), bits, dither, dither_2bit, enhance_contrast, text_mode, morph_dilation, brightness)
            for i, img_path in enumerate(image_files)
        ]

        # 結果格納用
        pages = [None] * page_count

        # 並列処理実行
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_page_to_xtg, task): task[0] for task in tasks}

            for future in as_completed(futures):
                i, page_data, success, error = future.result()

                if success:
                    pages[i] = page_data
                else:
                    console.print(f"  [red]ページ {i+1} 失敗：{error}[/red]")

        # スキップされたページがあれば除去
        pages = [p for p in pages if p is not None]

        # XTC ファイル作成
        create_xtc_file(
            str(output_path),
            pages,
            metadata.get('title', 'Unknown'),
            metadata.get('author', 'Unknown'),
            bits
        )

        return str(output_path)

    finally:
        # 一時ディレクトリ清掃
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def select_mode() -> str:
    """変換モード選択"""
    console.print(Panel.fit(
        "[bold]XTC 変換モードを選んでね[/bold]\n",
        border_style="cyan"
    ))

    modes = [
        ("1", "EPUB → XTC 変換（モノクロ・1 ビット）"),
        ("2", "EPUB → XTCH 変換（4 階調・2 ビット）"),
        ("3", "まとめて変換（モノクロ）"),
        ("4", "まとめて変換（4 階調）"),
        ("5", "戻る"),
    ]

    for key, desc in modes:
        console.print(f"  [yellow]{key}[/yellow]: {desc}")

    return Prompt.ask("\n選択", choices=["1", "2", "3", "4", "5"], default="1")


def select_dither_2bit() -> tuple:
    """2 ビット用ディザリング選択"""
    console.print(Panel.fit(
        "[bold]2 ビット用ディザリングを選んでね[/bold]\n",
        border_style="yellow"
    ))

    dithers = [
        ("1", "なし（単純量子化）"),
        ("2", "Floyd-Steinberg（誤差拡散・高画質）"),
        ("3", "Sierra Lite（軽量・高速）"),
        ("4", "Atkinson（自然な仕上がり）"),
        ("5", "Text（文字最適化・濃め）"),
        ("6", "Sharp（輪郭強調＋コントラスト・文字くっきり）"),
        ("7", "Manga（漫画最適化・黒髪は黒く）"),
        ("8", "Extreme（極限コントラスト・40/60/70）"),
        ("9", "Extreme40（40/45/50・肌色・白寄せ）"),
        ("10", "Extreme45（45/55/65・狭め）"),
        ("11", "Extreme50（50/55/60・さらに狭め）"),
        ("12", "Extreme55（55/65/75・広め）"),
    ]

    for key, desc in dithers:
        console.print(f"  [yellow]{key}[/yellow]: {desc}")

    dither_map = {
        "1": "none", "2": "floyd", "3": "sierra_lite", "4": "atkinson",
        "5": "text", "6": "sharp", "7": "manga", "8": "extreme",
        "9": "extreme_40_45_50", "10": "extreme_45_55_65", "11": "extreme_50_55_60", "12": "extreme_55_65_75"
    }
    choice = Prompt.ask("\n選択", choices=[str(i) for i in range(1, 13)], default="9")
    return dither_map[choice]


def select_dither() -> str:
    """ディザリング選択"""
    console.print(Panel.fit(
        "[bold]ディザリングを選んでね[/bold]（デフォルト：Floyd）\n",
        border_style="yellow"
    ))

    dithers = [
        ("1", "なし（単純量子化）"),
        ("2", "Floyd-Steinberg（誤差拡散・高画質）←おすすめ"),
        ("3", "Stucki（誤差拡散・鮮明）"),
        ("4", "Atkinson（誤差拡散・自然）"),
        ("5", "Bayer（順序ディザ）"),
    ]

    for key, desc in dithers:
        console.print(f"  [yellow]{key}[/yellow]: {desc}")

    dither_map = {"1": "none", "2": "floyd", "3": "stucki", "4": "atkinson", "5": "bayer"}
    choice = Prompt.ask("\n選択", choices=["1", "2", "3", "4", "5"], default="2")
    return dither_map[choice]


def preview_dither_patterns(epub_path: str, max_pages: int = 3):
    """
    複数ページの画像を various ディザリングでレンダリングしてプレビュー保存
    1 ビットと 2 ビットの両方に対応
    """
    import subprocess
    from pathlib import Path

    preview_dir = Path("/tmp/xtc_preview")
    preview_dir.mkdir(exist_ok=True)

    # EPUB を解凍
    temp_dir = Path(epub_path).parent / f".xtc_preview_{Path(epub_path).stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        extract_dir = extract_epub(epub_path, temp_dir / "epub_contents")

        # 画像ディレクトリを探す
        images_dir = Path(extract_dir) / "images"
        if not images_dir.exists():
            for d in Path(extract_dir).rglob("images"):
                images_dir = d
                break

        if not images_dir.exists():
            console.print("[red]画像ディレクトリが見つかりません[/red]")
            return

        # 画像ファイル一覧を取得
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
        image_files = sorted([
            f for f in images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ])

        if not image_files:
            console.print("[red]画像ファイルが見つかりません[/red]")
            return

        # 最初の max_pages 枚だけプレビュー
        preview_images = image_files[:max_pages]

        dither_options_1bit = [
            ("none", "なし（1 ビット）"),
            ("floyd", "Floyd-Steinberg（1 ビット）"),
            ("stucki", "Stucki（1 ビット）"),
            ("atkinson", "Atkinson（1 ビット）"),
            ("bayer", "Bayer（1 ビット）"),
        ]

        # 2 ビット用ディザリングオプション
        dither_options_2bit = [
            ("none", "なし（2 ビット）"),
            ("floyd", "Floyd-Steinberg（2 ビット）"),
            ("sierra_lite", "Sierra Lite（2 ビット）"),
            ("atkinson", "Atkinson（2 ビット）"),
            ("text", "Text（文字最適化）"),
            ("sharp", "Sharp（輪郭強調＋コントラスト）"),
            ("manga", "Manga（漫画最適化）"),
            ("extreme", "Extreme（40/60/70）"),
            ("extreme_45_55_65", "Extreme45（45/55/65・狭め）"),
            ("extreme_50_55_60", "Extreme50（50/55/60・さらに狭め）"),
            ("extreme_55_65_75", "Extreme55（55/65/75・広め）"),
        ]

        console.print(f"\n[cyan]1 ビットプレビュー画像を生成中…（{len(preview_images)} ページ × {len(dither_options_1bit)} パターン）[/cyan]")

        for page_idx, img_path in enumerate(preview_images):
            with Image.open(img_path) as base_img:
                base_img = ImageOps.grayscale(base_img)
                base_img.thumbnail((DISPLAY_WIDTH, DISPLAY_HEIGHT), Image.Resampling.LANCZOS)

                # 1 ビット用プレビュー
                for dither_name, dither_label in dither_options_1bit:
                    if dither_name == "none":
                        proc_img = base_img.point(lambda x: 0 if x < 128 else 255, mode='1')
                    else:
                        proc_img = apply_dithering(base_img.copy(), dither_name)
                        proc_img = proc_img.convert('1')

                    out_path = preview_dir / f"page{page_idx+1}_1bit_{dither_name}.png"
                    proc_img.save(out_path)

                # 2 ビット用プレビュー
                for dither_name, dither_label in dither_options_2bit:
                    text_mode = dither_name == 'text'
                    sharp_mode = dither_name == 'sharp'
                    manga_mode = dither_name == 'manga'
                    # sharp/manga/extreme モードは内部で全処理を行うため、追加パラメータは不要
                    if sharp_mode or manga_mode or dither_name.startswith('extreme'):
                        proc_img = apply_dithering_2bit(base_img.copy(), dither_name)
                    else:
                        proc_img = apply_dithering_2bit(base_img.copy(), dither_name, enhance_contrast=True, text_mode=text_mode, morph_dilation=0)
                    out_path = preview_dir / f"page{page_idx+1}_2bit_{dither_name}.png"
                    proc_img.save(out_path)

        # Finder で開く
        console.print(f"\n[green]プレビュー完了：{preview_dir}[/green]")
        console.print("Finder で開きます…")
        subprocess.run(["open", str(preview_dir)])

        # ユーザーに選択を促す
        console.print("\n[dim]/tmp/xtc_preview フォルダで比較してください[/dim]")
        console.print("[dim]1bit: 5 パターン，2bit: 5 パターンを生成しました[/dim]")

    finally:
        # 一時ディレクトリ清掃
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    """メインループ"""
    console.print(Panel.fit(
        "[bold magenta]📖 EPUB → XTC 変換ツール[/bold magenta]\n"
        "XTEINK X4/X3 向けに変換するよ！",
        border_style="magenta"
    ))
    console.print()

    work_dir = Path.cwd()
    epub_dir = work_dir / "epub"
    output_dir = work_dir / "xtc"

    epub_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    while True:
        mode = select_mode()

        if mode == "1":
            console.print(f"\n[yellow]EPUB フォルダからファイルを選んでね（モノクロ）[/yellow]")
            console.print(f"  参照フォルダ：{epub_dir}")

            epub_files = list(epub_dir.glob("*.epub"))
            if not epub_files:
                console.print("[red]EPUB フォルダにファイルがありません[/red]")
                continue

            for i, epub in enumerate(epub_files[:10], 1):
                console.print(f"    [{i}] {epub.name}")

            epub_input = Prompt.ask("\nEPUB ファイル名（または番号）")

            if epub_input.isdigit():
                idx = int(epub_input) - 1
                if 0 <= idx < len(epub_files):
                    epub_path = epub_files[idx]
                else:
                    console.print("[red]無効な選択[/red]")
                    continue
            else:
                epub_path = epub_dir / epub_input

            # EPUB を解凍して画像リストを取得（プレビュー用）
            temp_dir = epub_path.parent / f".xtc_temp_{epub_path.stem}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            try:
                extract_dir = extract_epub(epub_path, temp_dir / "epub_contents")
                images_dir = Path(extract_dir) / "images"
                if not images_dir.exists():
                    for d in Path(extract_dir).rglob("images"):
                        images_dir = d
                        break
                image_extensions = {".png", ".jpg", ".jpeg"}
                image_files = sorted([
                    f for f in images_dir.iterdir()
                    if f.suffix.lower() in image_extensions
                ])

                # プレビュー表示
                if image_files:
                    preview_choice = Prompt.ask(
                        "\nディザリングのプレビューを見ますか？",
                        choices=["y", "n"],
                        default="y"
                    )
                    if preview_choice.lower() == "y":
                        # プレビュー用に画像パスだけを渡す
                        preview_dither_patterns(str(epub_path))
                        # 続けて変換するか確認
                        cont = Prompt.ask("\n変換を続けますか？", choices=["y", "n"], default="y")
                        if cont.lower() != "y":
                            continue

                # ディザリング選択
                dither = select_dither()

                try:
                    output_path = epub_to_xtc(epub_path, output_dir, bits=1, dither=dither)
                    console.print(f"[green]✓ 変換完了：{output_path}[/green]")
                except Exception as e:
                    console.print(f"[red]エラー：{e}[/red]")
            finally:
                import shutil
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        elif mode == "2":
            # 2 ビット用ディザリング選択
            dither_2bit = select_dither_2bit()

            # 明るさ調整（60-100%）
            console.print("\n[yellow]明るさ調整（60-100%、100 が標準、90 がおすすめ）[/yellow]")
            brightness_input = Prompt.ask("  明るさ %", default="90")
            try:
                brightness = int(brightness_input)
                if brightness < 60 or brightness > 100:
                    console.print("[red]60-100 の範囲で入力してね[/red]")
                    continue
            except ValueError:
                console.print("[red]数値で入力してね[/red]")
                continue

            # テキストモードかどうか判定
            text_mode = dither_2bit == 'text'

            console.print(f"\n[yellow]EPUB フォルダからファイルを選んでね（4 階調グレースケール）[/yellow]")
            console.print(f"  参照フォルダ：{epub_dir}")

            epub_files = list(epub_dir.glob("*.epub"))
            if not epub_files:
                console.print("[red]EPUB フォルダにファイルがありません[/red]")
                continue

            for i, epub in enumerate(epub_files[:10], 1):
                console.print(f"    [{i}] {epub.name}")

            epub_input = Prompt.ask("\nEPUB ファイル名（または番号）")

            if epub_input.isdigit():
                idx = int(epub_input) - 1
                if 0 <= idx < len(epub_files):
                    epub_path = epub_files[idx]
                else:
                    console.print("[red]無効な選択[/red]")
                    continue
            else:
                epub_path = epub_dir / epub_input

            try:
                output_path = epub_to_xtc(epub_path, output_dir, bits=2, dither_2bit=dither_2bit, text_mode=text_mode, brightness=brightness)
            except Exception as e:
                console.print(f"[red]エラー：{e}[/red]")

        elif mode == "3":
            console.print(f"\n[yellow]EPUB フォルダをまとめて変換するよ（モノクロ）[/yellow]")

            # ディザリング選択
            dither = select_dither()

            epub_files = list(epub_dir.glob("*.epub"))
            if not epub_files:
                console.print("[red]EPUB フォルダにファイルがありません[/red]")
                continue

            for i, epub in enumerate(epub_files, 1):
                console.print(f"    [{i}] {epub.name}")

            selection = Prompt.ask(
                "\n変換するファイルを選択（例：1,3,5 または 1-3、すべては all）",
                default="all"
            )

            selected_files = []
            if selection.lower() == "all":
                selected_files = epub_files
            else:
                for part in selection.split(","):
                    part = part.strip()
                    if "-" in part:
                        start, end = map(int, part.split("-"))
                        selected_files.extend(epub_files[max(0,start-1):end])
                    elif part.isdigit():
                        idx = int(part) - 1
                        if 0 <= idx < len(epub_files):
                            selected_files.append(epub_files[idx])

            if not selected_files:
                console.print("[red]有効な選択がありません[/red]")
                continue

            console.print(f"\n[cyan]{len(selected_files)} 個の EPUB を変換します[/cyan]")

            success_count = 0
            error_count = 0

            for epub_path in selected_files:
                try:
                    output_path = epub_to_xtc(epub_path, output_dir, bits=1, dither=dither)
                    console.print(f"  [green]✓ {epub_path.name}[/green]")
                    success_count += 1
                except Exception as e:
                    console.print(f"  [red]✗ {epub_path.name}: {e}[/red]")
                    error_count += 1

            console.print(f"\n[green]完了：{success_count} 個成功，{error_count} 個失敗[/green]")

        elif mode == "4":
            # 2 ビット用ディザリング選択
            dither_2bit = select_dither_2bit()

            # 明るさ調整（60-100%）
            console.print("\n[yellow]明るさ調整（60-100%、100 が標準、90 がおすすめ）[/yellow]")
            brightness_input = Prompt.ask("  明るさ %", default="90")
            try:
                brightness = int(brightness_input)
                if brightness < 60 or brightness > 100:
                    console.print("[red]60-100 の範囲で入力してね[/red]")
                    continue
            except ValueError:
                console.print("[red]数値で入力してね[/red]")
                continue

            # テキストモードかどうか判定
            text_mode = dither_2bit == 'text'

            console.print(f"\n[yellow]EPUB フォルダをまとめて変換するよ（4 階調グレースケール）[/yellow]")

            epub_files = list(epub_dir.glob("*.epub"))
            if not epub_files:
                console.print("[red]EPUB フォルダにファイルがありません[/red]")
                continue

            for i, epub in enumerate(epub_files, 1):
                console.print(f"    [{i}] {epub.name}")

            selection = Prompt.ask(
                "\n変換するファイルを選択（例：1,3,5 または 1-3、すべては all）",
                default="all"
            )

            selected_files = []
            if selection.lower() == "all":
                selected_files = epub_files
            else:
                for part in selection.split(","):
                    part = part.strip()
                    if "-" in part:
                        start, end = map(int, part.split("-"))
                        selected_files.extend(epub_files[max(0,start-1):end])
                    elif part.isdigit():
                        idx = int(part) - 1
                        if 0 <= idx < len(epub_files):
                            selected_files.append(epub_files[idx])

            if not selected_files:
                console.print("[red]有効な選択がありません[/red]")
                continue

            console.print(f"\n[cyan]{len(selected_files)} 個の EPUB を変換します[/cyan]")

            success_count = 0
            error_count = 0

            for epub_path in selected_files:
                try:
                    output_path = epub_to_xtc(epub_path, output_dir, bits=2, dither_2bit=dither_2bit, text_mode=text_mode, brightness=brightness)
                    console.print(f"  [green]✓ {epub_path.name}[/green]")
                    success_count += 1
                except Exception as e:
                    console.print(f"  [red]✗ {epub_path.name}: {e}[/red]")
                    error_count += 1

            console.print(f"\n[green]完了：{success_count} 個成功，{error_count} 個失敗[/green]")

        elif mode == "5":
            console.print("\n[magenta]bye[/magenta]")
            break


if __name__ == "__main__":
    main()
