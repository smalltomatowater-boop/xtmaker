#!/usr/bin/env python3
"""
PDF/ZIP → EPUB → XTC 変換ツール
XTEINK X4/X3 リーダー向けに最適化

フロー：
- ZIP → PDF → EPUB → XTCH
- PDF → EPUB → XTCH
"""

import os
import sys
import struct
import zipfile
import shutil
import tempfile
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont
import fitz  # PyMuPDF
from ebooklib import epub

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

console = Console()

# XTEINK X4/X3 ディスプレイ仕様
DISPLAY_WIDTH = 528
DISPLAY_HEIGHT = 792

# 4階調量子化プリセット (t1, t2, t3) → 出力レベル 0, 85, 170, 255
QUANTIZE_PRESETS = {
    'soft':   (56, 128, 184),   # 中間域広め、自然な階調
    'normal': (64, 128, 192),   # デフォルト（均一量子化標準）
    'hard':   (72, 128, 200),   # コントラスト強め
    'sharp':  (80, 128, 208),   # 輪郭重視
    'crisp':  (88, 128, 216),   # 最大コントラスト
    'lloyd':  (42, 128, 213),   # Lloyd-Max最小歪み量子化
}

# XTCH ファイル形式定数（epub2xtc.py 準拠）
XTC_MAGIC = 0x00435458      # "XTC\0"
XTCH_MAGIC = 0x48435458     # "XTCH"
HEADER_SIZE = 48
METADATA_SIZE = 240          # title(128) + author(112)
PAGE_INDEX_ENTRY_SIZE = 16
CHAPTER_ENTRY_SIZE = 112


def _fnv1a_hash(data: bytes) -> int:
    """FNV-1a 32-bit hash"""
    h = 0x811c9dc5
    for byte in data:
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def high_quality_downscale(img, target_size, sharpen=True):
    """高品質な多段階リサイズ"""
    from PIL import ImageFilter, ImageOps

    target_width, target_height = target_size
    orig_width, orig_height = img.size
    scale = min(target_width / orig_width, target_height / orig_height)

    # 元の平均輝度を記録
    orig_gray = img.convert('L')
    orig_pixels = list(orig_gray.get_flattened_data())
    orig_avg = sum(orig_pixels) / len(orig_pixels) if orig_pixels else 128

    # 多段階リサイズ（各ステップ最大 0.5 倍）
    current = img.copy()
    if scale < 0.5:
        remaining_scale = scale
        while True:
            if remaining_scale >= 0.5:
                new_width = int(current.width * remaining_scale)
                new_height = int(current.height * remaining_scale)
                current = current.resize((new_width, new_height), Image.Resampling.LANCZOS)
                break
            else:
                new_width = current.width // 2
                new_height = current.height // 2
                current = current.resize((new_width, new_height), Image.Resampling.LANCZOS)
                remaining_scale /= 0.5

    if current.size != target_size:
        current = current.resize(target_size, Image.Resampling.LANCZOS)

    # グレースケール化
    current = ImageOps.grayscale(current)

    # シャープ（控えめに）
    if sharpen:
        current = current.filter(ImageFilter.UnsharpMask(radius=0.3, percent=50, threshold=3))

    # 明るさ維持：元の輝度を保つように調整
    current_pixels = list(current.getdata())
    current_avg = sum(current_pixels) / len(current_pixels) if current_pixels else 128

    # 輝度差を補正
    if current_avg > 0 and orig_avg > current_avg:
        brightness_factor = orig_avg / current_avg
        # 飛び抜け防止
        brightness_factor = min(brightness_factor, 1.3)
        current = ImageOps.autocontrast(current, cutoff=0.5)
        # 輝度調整
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(current)
        current = enhancer.enhance(brightness_factor * 0.9)

    return current


# =============================================================================
# ZIP → PDF
# =============================================================================

def extract_zip_images(zip_path: str, extract_dir: str = None) -> str:
    """ZIP ファイルから画像を抽出"""
    zip_path = Path(zip_path).resolve()

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP ファイルが見つかりません：{zip_path}")

    if extract_dir is None:
        extract_dir = zip_path.parent / f".zip_extract_{zip_path.stem}"

    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.infolist()

        # ルートディレクトリ構造を解析
        root_dirs = set()
        for m in members:
            filename = m.filename.rstrip('/')
            if filename and not m.is_dir():
                parts = filename.split("/")
                if len(parts) > 1:
                    root_dirs.add(parts[0])

        if len(root_dirs) == 1:
            common_root = list(root_dirs)[0]
            for m in members:
                filename = m.filename
                if filename.startswith(common_root) and not m.is_dir():
                    if not filename.startswith("__MACOSX") and not filename.startswith("."):
                        rel_path = filename[len(common_root):].lstrip("/")
                        if rel_path:
                            target = extract_dir / rel_path
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with zf.open(m) as src, open(target, "wb") as dst:
                                dst.write(src.read())
        else:
            for m in members:
                filename = m.filename
                if not filename.startswith("__MACOSX") and not filename.startswith("."):
                    target = extract_dir / filename
                    if not m.is_dir():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(m) as src, open(target, "wb") as dst:
                            dst.write(src.read())

    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    image_count = sum(1 for f in extract_dir.rglob("*")
                     if f.suffix.lower() in image_extensions)

    console.print(f"[green]✓ {image_count} 枚の画像を展開[/green]")
    return str(extract_dir)


def images_to_pdf(images_dir: str, output_path: str = None) -> str:
    """画像フォルダを PDF に変換"""
    images_dir = Path(images_dir).resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"画像フォルダが見つかりません：{images_dir}")

    if output_path is None:
        output_path = images_dir.parent / f"{images_dir.name}.pdf"

    output_path = Path(output_path)

    # 画像ファイル一覧取得（再帰的）
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    image_files = sorted([
        f for f in images_dir.rglob("*")
        if f.suffix.lower() in image_extensions
    ])

    if not image_files:
        raise ValueError(f"画像ファイルが見つかりません：{images_dir}")

    console.print(f"[cyan] 画像 {len(image_files)} 枚を PDF 化：{images_dir.name}[/cyan]")

    # 画像を開いて PDF として保存
    images = []
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
        except Exception as e:
            console.print(f"[yellow] 警告：{img_path.name} をスキップ ({e})[/yellow]")

    if not images:
        raise ValueError("有効な画像がありません")

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        resolution=150.0,
        format='PDF'
    )

    console.print(f"[green]✓ PDF 作成完了：{output_path}[/green]")
    return str(output_path)


def zip_to_pdf(zip_path: str, output_dir: str = None) -> str:
    """ZIP を PDF に変換"""
    zip_path = Path(zip_path).resolve()

    if output_dir is None:
        output_dir = zip_path.parent

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 一時ディレクトリに画像を展開
    temp_dir = zip_path.parent / f".zip_temp_{zip_path.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        extract_dir = extract_zip_images(str(zip_path), str(temp_dir))
        output_path = images_to_pdf(extract_dir, output_dir / f"{zip_path.stem}.pdf")
        return output_path
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


# =============================================================================
# PDF → EPUB
# =============================================================================

def pdf_to_epub_images(pdf_path: str, output_dir: str = None,
                       device: str = "x4", sharpen: bool = True,
                       vectorize: bool = False) -> str:
    """PDF を画像モードで EPUB 化（pdf2epub.py ベース）"""
    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF ファイルが見つかりません：{pdf_path}")

    if output_dir is None:
        output_dir = Path.cwd() / "epub"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{pdf_path.stem}.epub"

    console.print(f"[cyan]PDF を画像 EPUB 化：{pdf_path.name}[/cyan]")

    # デバイス解像度
    if device == "x4":
        target_size = (480, 800)
    elif device == "x3":
        target_size = (528, 792)
    else:
        target_size = None

    # PDF を開く
    doc = fitz.open(pdf_path)

    # 一時ディレクトリ作成（tempfile を使う）
    import tempfile
    temp_dir = tempfile.mkdtemp()

    try:
        # EPUB 作成
        book = epub.EpubBook()
        book.set_identifier(f"pdf-convert-{pdf_path.stem}")
        book.set_title(pdf_path.stem)
        book.set_language("ja")
        book.add_author("PDF Converter")

        chapters = []
        image_files = []

        # 各ページを画像として保存
        for i in range(len(doc)):
            page = doc[i]
            # 高解像度でレンダリング（5 倍）
            mat = fitz.Matrix(5, 5)
            pix = page.get_pixmap(matrix=mat)

            img_path = Path(temp_dir) / f"page_{i:04d}.png"
            pix.save(str(img_path))

            # デバイスサイズにリサイズ
            if target_size:
                with Image.open(img_path) as img:
                    resized = high_quality_downscale(img, target_size, sharpen=sharpen)
                    resized.save(str(img_path))

            image_files.append(img_path)

        # 画像と XHTML を追加
        img_width, img_height = target_size if target_size else (1200, 1600)

        for i, img_path in enumerate(image_files):
            # 画像アイテム
            img_item = epub.EpubImage()
            img_item.file_name = f"images/{img_path.name}"
            img_item.media_type = "image/png"

            with open(img_path, "rb") as f:
                img_item.content = f.read()

            book.add_item(img_item)

            # ページ XHTML
            c = epub.EpubHtml(title=f"Page {i+1}", file_name=f"page_{i:04d}.xhtml", lang="ja")
            c.content = f"""<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>Page {i+1}</title>
</head>
<body>
<div>
<img src="images/{img_path.name}" alt="Page {i+1}" width="{img_width}" height="{img_height}"/>
</div>
</body>
</html>"""

            book.add_item(c)
            chapters.append(c)

        doc.close()

        book.toc = chapters
        book.add_item(epub.EpubNcx())
        book.spine = chapters

        # EPUB 保存
        epub.write_epub(str(output_path), book, {})

        console.print(f"[green]✓ EPUB 作成完了：{output_path}[/green]")
        return str(output_path)

    finally:
        import shutil
        shutil.rmtree(temp_dir)


# =============================================================================
# EPUB → XTC
# =============================================================================

def extract_epub(epub_path: str, extract_dir: str) -> str:
    """EPUB を解凍"""
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(epub_path, 'r') as zf:
        zf.extractall(extract_dir)

    return str(extract_dir)


def parse_opf(opf_path: str) -> dict:
    """OPF ファイルからメタデータを抽出"""
    import xml.etree.ElementTree as ET

    tree = ET.parse(opf_path)
    root = tree.getroot()

    ns = {'opf': 'http://www.idpf.org/2007/opf',
          'dc': 'http://purl.org/dc/elements/1.1/'}

    metadata = {}
    meta_elem = root.find('.//opf:metadata', ns)

    if meta_elem is not None:
        for tag in ['title', 'creator', 'language']:
            elem = meta_elem.find(f'dc:{tag}', ns)
            if elem is not None:
                metadata[tag] = elem.text
            else:
                metadata[tag] = 'Unknown'

    return metadata


def apply_extreme_contrast(img, pixels, width, height, t1, t2, t3, brightness=100):
    """極限コントラスト処理"""
    # 明るさ調整を閾値変換前に適用
    if brightness != 100:
        factor = brightness / 100
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)

    pixels = img.load()

    # LUT 作成
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

    # 誤差拡散 (Floyd-Steinberg)
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

            if x + 1 < width:
                pixels[x + 1, y] = max(0, min(255, pixels[x + 1, y] + error * 7 // 16))
            if y + 1 < height:
                if x > 0:
                    pixels[x - 1, y + 1] = max(0, min(255, pixels[x - 1, y + 1] + error * 3 // 16))
                pixels[x, y + 1] = max(0, min(255, pixels[x, y + 1] + error * 5 // 16))
                if x + 1 < width:
                    pixels[x + 1, y + 1] = max(0, min(255, pixels[x + 1, y + 1] + error * 1 // 16))

    return img


def apply_dithering_2bit(img: Image.Image, method: str = 'normal',
                          enhance_contrast: bool = True,
                          text_mode: bool = False,
                          morph_dilation: int = 0,
                          brightness: int = 90) -> Image.Image:
    """2 ビット（4 階調）用ディザリング"""
    img = img.convert('L').copy()
    pixels = img.load()
    width, height = img.size

    # プリセットから閾値を取得
    if method in QUANTIZE_PRESETS:
        t1, t2, t3 = QUANTIZE_PRESETS[method]
        return apply_extreme_contrast(img, pixels, width, height, t1, t2, t3, brightness)

    # 後方互換: extreme_T1_T2_T3 形式
    if method.startswith('extreme_'):
        parts = method.split('_')
        if len(parts) == 3:
            t1, t2, t3 = int(parts[1]), int(parts[2]), int(parts[3])
            return apply_extreme_contrast(img, pixels, width, height, t1, t2, t3, brightness)

    # デフォルト
    t1, t2, t3 = QUANTIZE_PRESETS['normal']
    return apply_extreme_contrast(img, pixels, width, height, t1, t2, t3, brightness)


def render_page_to_image(img_path: str, width: int = DISPLAY_WIDTH,
                         height: int = DISPLAY_HEIGHT, bits: int = 2,
                         dither_2bit: str = None,
                         brightness: int = 90) -> Image.Image:
    """画像ファイルを XTEINK 向けにレンダリング"""
    with Image.open(img_path) as img:
        img.thumbnail((width, height), Image.Resampling.LANCZOS)
        img = ImageOps.grayscale(img)

        if bits == 2:
            if dither_2bit is None:
                dither_2bit = 'normal'
            img = apply_dithering_2bit(img, dither_2bit, brightness=brightness)

        result = Image.new('L', (width, height), 255)
        offset = ((width - img.width) // 2, (height - img.height) // 2)
        result.paste(img, offset)

        return result


def encode_xtg_page(img: Image.Image, bits: int = 2) -> bytes:
    """画像を XTG/XTH フォーマットのページデータにエンコード
    （epub2xtc.py 準拠：ページヘッダー 22 バイト + データ）"""
    width, height = img.size
    image_data = bytearray()

    if bits == 1:
        # 1 ビット：行優先、8 ピクセル/バイト
        for y in range(height):
            for x in range(0, width, 8):
                byte = 0
                for bit in range(8):
                    if x + bit < width:
                        pixel = img.getpixel((x + bit, y))
                        if pixel >= 128:
                            byte |= (0x80 >> bit)
                image_data.append(byte)

    else:
        # 2 ビット：列優先、2 ビットプレーン構造
        bytes_per_col = (height + 7) // 8
        p1 = bytearray(width * bytes_per_col)
        p2 = bytearray(width * bytes_per_col)

        for x_idx, x in enumerate(range(width - 1, -1, -1)):  # 右から左
            for y in range(height):
                pixel = img.getpixel((x, y))
                raw_value = pixel // 64  # 0-3
                bit_value = 3 - raw_value  # 反転（XTEINK は 0=白，3=黒）

                bit1 = (bit_value >> 1) & 0x01
                bit2 = bit_value & 0x01

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

    page_header = bytearray()
    page_header.extend(struct.pack('<I', page_magic))
    page_header.extend(struct.pack('<H', width))
    page_header.extend(struct.pack('<H', height))
    page_header.extend(b'\x00\x00\x00')
    page_header.append(width // 4 - 1)
    page_header.extend(struct.pack('<I', 1))
    page_header.extend(b'\x00' * 6)

    return bytes(page_header) + bytes(image_data)


def create_xtc_file(output_path: str, pages: list, title: str, author: str, bits: int = 2):
    """XTC/XTCH ファイルを作成（epub2xtc.py 準拠の正式フォーマット）"""
    output_path = Path(output_path)
    page_count = len(pages)

    # オフセット計算
    metadata_offset = HEADER_SIZE
    metadata_size = METADATA_SIZE
    chapter_offset = metadata_offset + metadata_size
    chapter_size = page_count * CHAPTER_ENTRY_SIZE
    index_table_offset = chapter_offset + chapter_size
    index_table_size = page_count * PAGE_INDEX_ENTRY_SIZE
    data_offset = index_table_offset + index_table_size

    # 各ページのオフセットとサイズ
    page_offsets = []
    page_sizes = []
    current_offset = data_offset
    for page_data in pages:
        page_offsets.append(current_offset)
        page_sizes.append(len(page_data))
        current_offset += len(page_data)

    # ヘッダー
    magic = XTCH_MAGIC if bits == 2 else XTC_MAGIC
    header = struct.pack('<I', magic)
    header += struct.pack('<H', 0x0100)   # version
    header += struct.pack('<H', page_count)
    header += struct.pack('<B', 0)        # readDirection
    header += struct.pack('<B', 1)        # hasMetadata
    header += struct.pack('<B', 0)        # hasThumbnails
    header += struct.pack('<B', 1)        # hasChapters
    header += struct.pack('<I', 0)        # currentPage
    header += struct.pack('<Q', metadata_offset)
    header += struct.pack('<Q', index_table_offset)
    header += struct.pack('<Q', data_offset)
    header += b'\x00' * 8                 # padding

    # メタデータ
    title_bytes = title.encode('utf-8')[:128].ljust(128, b'\x00')
    author_bytes = author.encode('utf-8')[:112].ljust(112, b'\x00')

    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(title_bytes)
        f.write(author_bytes)

        # チャプター情報
        for i in range(page_count):
            chapter_name = f"Page {i+1}"
            name_hash = _fnv1a_hash(chapter_name.encode('utf-8'))
            f.write(struct.pack('<I', name_hash))
            f.write(struct.pack('<H', 0))          # reserved
            f.write(struct.pack('<H', page_count))
            f.write(b'\x00' * 8)                    # reserved
            chapter_name_bytes = chapter_name.encode('utf-8')[:96]
            f.write(chapter_name_bytes)
            f.write(b'\x00' * (96 - len(chapter_name_bytes)))

        # インデックステーブル
        for i in range(page_count):
            f.write(struct.pack('<Q', page_offsets[i]))
            f.write(struct.pack('<I', page_sizes[i]))
            f.write(struct.pack('<H', DISPLAY_WIDTH))
            f.write(struct.pack('<H', DISPLAY_HEIGHT))

        # データエリア
        for page_data in pages:
            f.write(page_data)

    console.print(f"[green]✓ ファイル作成完了：{output_path}[/green]")


def process_page(args):
    """個別ページ処理（トップレベル関数）"""
    i, img_path, width, height, bits, dither_2bit, brightness = args
    try:
        img = render_page_to_image(str(img_path), width, height, bits, dither_2bit, brightness)
        page_data = encode_xtg_page(img, bits)
        return (i, page_data, True, None)
    except Exception as e:
        return (i, None, False, str(e))


def epub_to_xtc(epub_path: str, output_dir: str = None, bits: int = 2,
                dither_2bit: str = None, brightness: int = 100,
                max_workers: int = None) -> str:
    """EPUB を XTC 形式に変換"""
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
        extract_dir = extract_epub(str(epub_path), str(temp_dir / "epub_contents"))

        opf_files = list(Path(extract_dir).rglob("*.opf"))
        if opf_files:
            metadata = parse_opf(str(opf_files[0]))
        else:
            metadata = {'title': epub_path.stem, 'author': 'Unknown'}

        # 画像ディレクトリを探す
        images_dir = Path(extract_dir) / "images"
        if not images_dir.exists():
            for d in Path(extract_dir).rglob("images"):
                images_dir = d
                break
        if not images_dir.exists():
            epub_images = Path(extract_dir) / "EPUB" / "images"
            if epub_images.exists():
                images_dir = epub_images

        if not images_dir.exists():
            raise ValueError("画像ディレクトリが見つかりません")

        # 画像ファイル一覧
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
        image_files = sorted([
            f for f in images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ])

        if not image_files:
            raise ValueError("画像ファイルが見つかりません")

        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)

        page_count = len(image_files)
        console.print(f"  [dim]{page_count} ページを{max_workers}スレッドで処理[/dim]")

        # 並列処理
        tasks = [
            (i, str(img_path), DISPLAY_WIDTH, DISPLAY_HEIGHT, bits, dither_2bit, brightness)
            for i, img_path in enumerate(image_files)
        ]

        pages = [None] * page_count

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_page, task): task[0] for task in tasks}

            for future in as_completed(futures):
                i, page_data, success, error = future.result()

                if success:
                    pages[i] = page_data
                else:
                    console.print(f"  [red]ページ {i+1} 失敗：{error}[/red]")

        pages = [p for p in pages if p is not None]

        create_xtc_file(
            str(output_path),
            pages,
            metadata.get('title', 'Unknown'),
            metadata.get('author', 'Unknown'),
            bits
        )

        return str(output_path)

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


# =============================================================================
# メイン - 統合フロー
# =============================================================================

def select_device() -> str:
    """デバイス選択"""
    console.print(Panel.fit(
        "[bold] デバイスを選んでね[/bold]\n",
        border_style="cyan"
    ))

    devices = [
        ("1", "XTEINK X4（480x800）←おすすめ"),
        ("2", "XTEINK X3（528x792）"),
        ("3", "Raw（高解像度・タブレット）"),
    ]

    for key, desc in devices:
        console.print(f"  [yellow]{key}[/yellow]: {desc}")

    device_map = {"1": "x4", "2": "x3", "3": "raw"}
    choice = Prompt.ask("\n選択", choices=["1", "2", "3"], default="1")
    return device_map[choice]


def select_dither_2bit() -> str:
    """2 ビット用ディザリング選択"""
    console.print(Panel.fit(
        "[bold]2 ビット用ディザリングを選んでね[/bold]\n",
        border_style="yellow"
    ))

    dithers = [
        ("1", f"Normal（{QUANTIZE_PRESETS['normal'][0]}-{QUANTIZE_PRESETS['normal'][1]}-{QUANTIZE_PRESETS['normal'][2]}・デフォルト）"),
        ("2", f"Soft（{QUANTIZE_PRESETS['soft'][0]}-{QUANTIZE_PRESETS['soft'][1]}-{QUANTIZE_PRESETS['soft'][2]}）"),
        ("3", f"Hard（{QUANTIZE_PRESETS['hard'][0]}-{QUANTIZE_PRESETS['hard'][1]}-{QUANTIZE_PRESETS['hard'][2]}）"),
        ("4", f"Sharp（{QUANTIZE_PRESETS['sharp'][0]}-{QUANTIZE_PRESETS['sharp'][1]}-{QUANTIZE_PRESETS['sharp'][2]}）"),
        ("5", f"Crisp（{QUANTIZE_PRESETS['crisp'][0]}-{QUANTIZE_PRESETS['crisp'][1]}-{QUANTIZE_PRESETS['crisp'][2]}）"),
        ("6", f"Lloyd-Max（{QUANTIZE_PRESETS['lloyd'][0]}-{QUANTIZE_PRESETS['lloyd'][1]}-{QUANTIZE_PRESETS['lloyd'][2]}）"),
    ]

    for key, desc in dithers:
        console.print(f"  [yellow]{key}[/yellow]: {desc}")

    dither_map = {
        "1": "normal", "2": "soft",
        "3": "hard", "4": "sharp",
        "5": "crisp", "6": "lloyd"
    }
    choice = Prompt.ask("\n選択", choices=["1", "2", "3", "4", "5", "6"], default="1")
    return dither_map[choice]


# =============================================================================
# バッチ並列用ワーカー関数（トップレベル）
# =============================================================================

def _worker_zip_to_pdf(zip_path_str: str, pdf_dir_str: str) -> tuple:
    """ZIP → PDF ワーカー（戻り値: (成功, PDFパスorエラーメッセージ)）"""
    try:
        result = zip_to_pdf(zip_path_str, pdf_dir_str)
        return (True, result)
    except Exception as e:
        return (False, str(e))


def _worker_images_to_pdf(folder_str: str, pdf_dir_str: str) -> tuple:
    """画像フォルダ → PDF ワーカー"""
    try:
        folder = Path(folder_str)
        output_path = Path(pdf_dir_str) / f"{folder.name}.pdf"
        result = images_to_pdf(folder_str, str(output_path))
        return (True, result)
    except Exception as e:
        return (False, str(e))


def _worker_pdf_to_epub(pdf_path_str: str, epub_dir_str: str, device: str) -> tuple:
    """PDF → EPUB ワーカー"""
    try:
        result = pdf_to_epub_images(pdf_path_str, epub_dir_str, device=device)
        return (True, result)
    except Exception as e:
        return (False, str(e))


def _worker_epub_to_xtc(epub_path_str: str, xtc_dir_str: str, bits: int,
                        dither_2bit: str, brightness: int) -> tuple:
    """EPUB → XTC ワーカー"""
    try:
        result = epub_to_xtc(epub_path_str, xtc_dir_str, bits=bits,
                            dither_2bit=dither_2bit, brightness=brightness,
                            max_workers=1)
        return (True, result)
    except Exception as e:
        return (False, str(e))


def main():
    console.print(Panel.fit(
        "[bold magenta]📚 PDF/ZIP/EPUB → XTC 変換ツール[/bold magenta]\n"
        "XTEINK X4/X3 向けに変換するよ！",
        border_style="magenta"
    ))
    console.print()

    work_dir = Path.cwd()
    input_dir = work_dir / "input"
    pdf_dir = work_dir / "pdf"
    epub_dir = work_dir / "epub"
    xtc_dir = work_dir / "xtc"

    input_dir.mkdir(exist_ok=True)
    pdf_dir.mkdir(exist_ok=True)
    epub_dir.mkdir(exist_ok=True)
    xtc_dir.mkdir(exist_ok=True)

    # ===== 最初に設定を聞く =====
    console.print("[yellow]ステップ 1/3: サイズ縮小[/yellow]")
    console.print("  PDF→EPUB 時の画像サイズ")
    device = select_device()

    console.print("\n[yellow]ステップ 2/3: 2bit 方式[/yellow]")
    dither_2bit = select_dither_2bit()

    console.print("\n[yellow]ステップ 3/3: 明るさ調整（60-100%、100 が標準）[/yellow]")
    brightness_input = Prompt.ask("  明るさ %", default="100")
    try:
        brightness = int(brightness_input)
        if brightness < 60 or brightness > 100:
            console.print("[red]60-100 の範囲で入力してね。デフォルト 100% を使います。[/red]")
            brightness = 100
    except ValueError:
        console.print("[red]数値で入力してね。デフォルト 100% を使います。[/red]")
        brightness = 100

    console.print("\n[yellow] 変換モードを選んでね[/yellow]")
    console.print("  1: 1 個ずつ変換")
    console.print("  2: 全部まとめて変換\n")

    batch_mode = Prompt.ask("選択", choices=["1", "2"], default="1") == "2"

    # ===== input/ 内のソースファイルを収集 =====
    def collect_sources():
        """input/ 内の全ソースファイルを収集"""
        zip_files = [(f, 'zip') for f in input_dir.glob("*.zip")]
        pdf_files = [(f, 'pdf') for f in input_dir.glob("*.pdf")]
        epub_files = [(f, 'epub') for f in input_dir.glob("*.epub")]

        image_folders = []
        image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        for d in sorted(input_dir.iterdir()):
            if d.is_dir() and not d.name.startswith('.'):
                try:
                    has_images = any(f.suffix.lower() in image_exts for f in d.iterdir())
                    if has_images:
                        image_folders.append((d, 'image_folder'))
                except PermissionError:
                    pass

        return zip_files + pdf_files + image_folders + epub_files

    all_files = collect_sources()

    if not all_files:
        console.print("[red]input/ にファイルがありません[/red]")
        return

    # ===== 全部まとめて変換（3ステージ並列） =====
    if batch_mode:
        console.print(f"\n[cyan]{len(all_files)} 個のファイルを変換します[/cyan]\n")

        # 分類
        zip_files = [f for f, t in all_files if t == 'zip']
        pdf_files = [f for f, t in all_files if t == 'pdf']
        image_folders = [f for f, t in all_files if t == 'image_folder']
        epub_files = [f for f, t in all_files if t == 'epub']

        # --- Stage 1: ZIP/画像フォルダ → PDF（並列 4 workers） ---
        stage1_tasks = []
        for zf in zip_files:
            stage1_tasks.append(('_worker_zip_to_pdf', str(zf), str(pdf_dir), f"ZIP→PDF: {zf.name}"))
        for fd in image_folders:
            stage1_tasks.append(('_worker_images_to_pdf', str(fd), str(pdf_dir), f"画像→PDF: {fd.name}"))

        generated_pdfs = []
        if stage1_tasks:
            console.print(f"\n[cyan]━━━ Stage 1: {len(stage1_tasks)} ファイルを並列処理 ━━━[/cyan]")
            max_w = min(4, len(stage1_tasks))
            with ProcessPoolExecutor(max_workers=max_w) as executor:
                futures = {}
                for worker_name, arg1, arg2, label in stage1_tasks:
                    console.print(f"  [dim]  {label}[/dim]")
                    if worker_name == '_worker_zip_to_pdf':
                        f = executor.submit(_worker_zip_to_pdf, arg1, arg2)
                    else:
                        f = executor.submit(_worker_images_to_pdf, arg1, arg2)
                    futures[f] = label

                for future in as_completed(futures):
                    label = futures[future]
                    ok, result = future.result()
                    if ok:
                        console.print(f"  [green]  ✓ {label}[/green]")
                        generated_pdfs.append(Path(result))
                    else:
                        console.print(f"  [red]  ✗ {label}: {result}[/red]")

        # Stage 2: 全PDF → EPUB（並列 4 workers）
        # 処理対象: input/内の元PDF + Stage 1で生成されたPDF
        all_pdfs = pdf_files + generated_pdfs
        all_pdfs = list(dict.fromkeys(all_pdfs))  # 重複除去（パスベースで）

        generated_epubs = []
        if all_pdfs:
            console.print(f"\n[cyan]━━━ Stage 2: {len(all_pdfs)} ファイルを並列処理 ━━━[/cyan]")
            max_w = min(4, len(all_pdfs))
            with ProcessPoolExecutor(max_workers=max_w) as executor:
                futures = {}
                for pdf in all_pdfs:
                    console.print(f"  [dim]  PDF→EPUB: {pdf.name}[/dim]")
                    f = executor.submit(_worker_pdf_to_epub, str(pdf), str(epub_dir), device)
                    futures[f] = pdf.name

                for future in as_completed(futures):
                    name = futures[future]
                    ok, result = future.result()
                    if ok:
                        console.print(f"  [green]  ✓ PDF→EPUB: {name}[/green]")
                        generated_epubs.append(Path(result))
                    else:
                        console.print(f"  [red]  ✗ PDF→EPUB: {name}: {result}[/red]")

        # Stage 3: 全EPUB → XTC（並列 4 workers、各ファイルは内部 max_workers=1）
        # 処理対象: input/内の元EPUB + Stage 2で生成されたEPUB
        all_epubs = epub_files + generated_epubs
        all_epubs = list(dict.fromkeys(all_epubs))

        generated_xtc = []
        if all_epubs:
            console.print(f"\n[cyan]━━━ Stage 3: {len(all_epubs)} ファイルを並列処理 ━━━[/cyan]")
            max_w = min(4, len(all_epubs))
            with ProcessPoolExecutor(max_workers=max_w) as executor:
                futures = {}
                for epub in all_epubs:
                    console.print(f"  [dim]  EPUB→XTCH: {epub.name}[/dim]")
                    f = executor.submit(_worker_epub_to_xtc, str(epub), str(xtc_dir),
                                       2, dither_2bit, brightness)
                    futures[f] = epub.name

                for future in as_completed(futures):
                    name = futures[future]
                    ok, result = future.result()
                    if ok:
                        console.print(f"  [green]  ✓ EPUB→XTCH: {name}[/green]")
                        generated_xtc.append(Path(result))
                    else:
                        console.print(f"  [red]  ✗ EPUB→XTCH: {name}: {result}[/red]")

        # 結果まとめ
        total = len(generated_pdfs) + len(generated_epubs) + len(generated_xtc)
        console.print(f"\n[green bold] 完了！ XTCH {len(generated_xtc)} 個作成[/green bold]")

        # フォルダを開く
        if generated_xtc:
            open_folder = Prompt.ask("\nXTCH フォルダを開きますか？", choices=["y", "n"], default="y")
            if open_folder.lower() == "y":
                import subprocess
                subprocess.run(["open", str(xtc_dir)])

        return

    # ===== 1 個ずつ変換モード =====
    console.print(f"\n[yellow]入力ファイルを選んでね[/yellow]")
    console.print(f"  参照：input/（ZIP, PDF, EPUB, 画像フォルダ）")

    # 表示
    for i, (f, ftype) in enumerate(all_files[:20], 1):
        if ftype == 'zip':
            icon = "[ZIP]"
        elif ftype == 'pdf':
            icon = "[PDF]"
        elif ftype == 'image_folder':
            icon = "[IMG]"
        else:
            icon = "[EPUB]"
        console.print(f"    [{i}] {icon} {f.name}")

    choice = Prompt.ask("\nファイル番号", default="1")

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(all_files):
            input_path, input_type = all_files[idx]
        else:
            console.print("[red]無効な選択[/red]")
            return
    except ValueError:
        console.print("[red]数値で入力してね[/red]")
        return

    try:
        if input_type == 'zip':
            console.print(f"\n[cyan]Step 1/3: ZIP → PDF[/cyan]")
            pdf_path = zip_to_pdf(str(input_path), str(pdf_dir))
            console.print(f"\n[cyan]Step 2/3: PDF → EPUB（{device}）[/cyan]")
            epub_path = pdf_to_epub_images(pdf_path, str(epub_dir), device=device)
            console.print(f"\n[cyan]Step 3/3: EPUB → XTCH[/cyan]")
            xtc_path = epub_to_xtc(epub_path, str(xtc_dir), bits=2, dither_2bit=dither_2bit, brightness=brightness)

        elif input_type == 'pdf':
            console.print(f"\n[cyan]Step 1/2: PDF → EPUB（{device}）[/cyan]")
            epub_path = pdf_to_epub_images(str(input_path), str(epub_dir), device=device)
            console.print(f"\n[cyan]Step 2/2: EPUB → XTCH[/cyan]")
            xtc_path = epub_to_xtc(epub_path, str(xtc_dir), bits=2, dither_2bit=dither_2bit, brightness=brightness)

        elif input_type == 'image_folder':
            console.print(f"\n[cyan]Step 1/3: 画像 → PDF[/cyan]")
            pdf_path = images_to_pdf(str(input_path), str(pdf_dir / f"{input_path.name}.pdf"))
            console.print(f"\n[cyan]Step 2/3: PDF → EPUB（{device}）[/cyan]")
            epub_path = pdf_to_epub_images(pdf_path, str(epub_dir), device=device)
            console.print(f"\n[cyan]Step 3/3: EPUB → XTCH[/cyan]")
            xtc_path = epub_to_xtc(epub_path, str(xtc_dir), bits=2, dither_2bit=dither_2bit, brightness=brightness)

        else:  # epub
            console.print(f"\n[cyan]EPUB → XTCH[/cyan]")
            xtc_path = epub_to_xtc(str(input_path), str(xtc_dir), bits=2, dither_2bit=dither_2bit, brightness=brightness)

        console.print(f"\n[green bold]✓ 完了！[/green bold]")
        console.print(f"  XTCH ファイル：{xtc_path}")

        open_folder = Prompt.ask("\nXTCH フォルダを開きますか？", choices=["y", "n"], default="y")
        if open_folder.lower() == "y":
            import subprocess
            subprocess.run(["open", str(xtc_dir)])

    except Exception as e:
        console.print(f"[red] エラー：{e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
