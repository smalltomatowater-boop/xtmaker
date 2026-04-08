#!/usr/bin/env python3
"""
PDF/画像 → EPUB 変換ツール
対話式で操作できるシンプルアプリ
"""

import os
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path

from PIL import Image, ImageOps

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.progress import Progress

import fitz  # PyMuPDF
import pymupdf4llm
from ebooklib import epub

console = Console()


# デバイス別ターゲットサイズ
DEVICE_TARGETS = {
    'raw': None,        # サイズ変更なし
    'x4': (480, 800),   # XTEINK X4
    'x3': (528, 792),   # XTEINK X3
}

# 4階調量子化プリセット (t1, t2, t3) → 出力レベル 0, 85, 170, 255
QUANTIZE_PRESETS = {
    'deep':   (40, 128, 168),   # 最も階調豊か
    'light':  (48, 128, 176),   # 階調豊か
    'soft':   (56, 128, 184),   # 中間域広め、読みやすい
    'normal': (64, 128, 192),   # デフォルト（均一量子化標準）
    'hard':   (72, 128, 200),   # コントラスト強め
}


def high_quality_downscale(img: Image.Image, target_size: tuple,
                           sharpen: bool = True, vectorize: bool = False,
                           quantize_preset: str = 'normal') -> Image.Image:
    """
    高品質な画像縮小処理

    多段階リサイズ＋シャープで、文字・線画を綺麗に維持

    Args:
        img: 元画像（RGB またはグレースケール）
        target_size: (幅，高さ)
        sharpen: True で輪郭強調（デフォルトで有効）
        vectorize: True でベクトル風変換（線画・文字用・試験的）

    Returns:
        縮小後の画像（グレースケール維持、vectorize の場合はバイナリ風）
    """
    from PIL import ImageFilter, ImageOps

    orig_width, orig_height = img.size
    target_width, target_height = target_size

    # 縮小率を計算
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height
    scale = min(scale_x, scale_y)

    # 前処理：シャープはリサイズ後にやる（前だと潰れる）

    # 多段階リサイズ（各ステップ最大 0.5 倍）
    current = img.copy()

    if scale < 0.5:
        remaining_scale = scale
        while True:
            if remaining_scale >= 0.5:
                step_scale = remaining_scale
                new_width = int(current.width * step_scale)
                new_height = int(current.height * step_scale)
                current = current.resize((new_width, new_height), Image.Resampling.LANCZOS)
                break
            else:
                new_width = current.width // 2
                new_height = current.height // 2
                current = current.resize((new_width, new_height), Image.Resampling.LANCZOS)
                remaining_scale /= 0.5

    if current.size != target_size:
        current = current.resize(target_size, Image.Resampling.LANCZOS)

    # ベクトル風変換（試験的）
    if vectorize:
        # グレースケール化
        current = ImageOps.grayscale(current)
        # コントラスト調整（cutoff=1 で両端を少し詰める）
        current = ImageOps.autocontrast(current, cutoff=1)
        # 4 階調量子化（プリセット選択）
        t1, t2, t3 = QUANTIZE_PRESETS.get(quantize_preset, QUANTIZE_PRESETS['normal'])
        current = current.point(lambda x, t1=t1, t2=t2, t3=t3:
            0 if x < t1 else (85 if x < t2 else (170 if x < t3 else 255)))

    # 輪郭強調（オプション）
    if sharpen and not vectorize:
        # グレースケール化
        current = ImageOps.grayscale(current)
        # 弱いアンシャープで細い線を維持（threshold=5 で細部を保護）
        current = current.filter(ImageFilter.UnsharpMask(radius=0.5, percent=100, threshold=5))
        # コントラストを少し上げる（細部を潰さないよう控えめ）
        current = ImageOps.autocontrast(current, cutoff=1)

    return current


def pdf_to_epub(pdf_path: str, output_dir: str = None, use_images: bool = False,
                device: str = 'x4', sharpen: bool = True, vectorize: bool = False,
                quantize_preset: str = 'normal') -> str:
    """PDF ファイルを EPUB に変換

    Args:
        pdf_path: PDF ファイルのパス
        output_dir: 出力ディレクトリ
        use_images: True の場合、各ページを画像として変換（漫画など画像ベース PDF 用）
        device: デバイス最適化 ('raw', 'x4', 'x3')
        sharpen: True で輪郭強調（漫画・文字向け）
        vectorize: True でベクトル風変換（線画・文字用・試験的）
    """
    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF ファイルが見つかりません：{pdf_path}")

    # デバッグ情報
    console.print(f"  [dim] パス：{pdf_path}[/dim]")
    console.print(f"  [dim] サイズ：{pdf_path.stat().st_size} bytes[/dim]")

    if output_dir is None:
        output_dir = Path.cwd() / "epub"
    else:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{pdf_path.stem}.epub"

    console.print(f"[cyan] 変換中：{pdf_path.name}[/cyan]")

    # EPUB 作成
    book = epub.EpubBook()
    book.set_identifier(f"pdf-convert-{pdf_path.stem}")
    book.set_title(pdf_path.stem)
    book.set_language("ja")
    book.add_author("PDF Converter")

    # 固定レイアウト指定（漫画・画像 EPUB 用）
    if use_images:
        book.add_metadata('DC', 'epub: rendition:layout', 'pre-paginated')
        book.add_metadata('DC', 'epub: rendition:spread', 'none')  # 見開き禁止
        book.add_metadata('DC', 'epub: rendition:orientation', 'portrait')  # 縦向き固定

    # PDF ドキュメントを開く
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        raise RuntimeError(f"PDF を開けません：{e}")

    # 空のドキュメントチェック
    if doc.page_count == 0:
        doc.close()
        raise ValueError(f"PDF が空です：{pdf_path.name}")

    chapters = []
    temp_dir = None

    if use_images:
        # 画像ベース変換：各ページを画像として埋め込み
        temp_dir = tempfile.mkdtemp()

        # デバイス別ターゲットサイズ取得
        target_size = DEVICE_TARGETS.get(device, (480, 800))

        for i, page in enumerate(doc):
            # ページを画像としてレンダリング（高解像度）
            mat = fitz.Matrix(5.0, 5.0)
            pix = page.get_pixmap(matrix=mat)

            # 一旦 PNG として保存
            img_path = Path(temp_dir) / f"page_{i:04d}.png"
            pix.save(str(img_path))

            # デバイス向け最適化処理（縮小のみ、色空間は維持）
            if target_size:
                with Image.open(img_path) as img:
                    # 高品質な多段階ダウンサンプリング
                    resized = high_quality_downscale(img, target_size,
                                                     sharpen=sharpen,
                                                     vectorize=vectorize,
                                                     quantize_preset=quantize_preset)
                    img_width, img_height = target_size
                    resized.save(str(img_path))
            else:
                img_width, img_height = pix.width, pix.height

            # 画像アイテムとして EPUB に追加
            img_item = epub.EpubImage()
            img_item.file_name = f"images/page_{i:04d}.png"
            img_item.media_type = "image/png"

            with open(img_path, "rb") as f:
                img_item.content = f.read()

            book.add_item(img_item)

            # 画像表示用のチャプター
            c = epub.EpubHtml(title=f"Page {i+1}", file_name=f"page_{i:04d}.xhtml", lang="ja")
            c.content = f"""<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>Page {i+1}</title>
</head>
<body>
<div>
<img src="images/page_{i:04d}.png" alt="Page {i+1}" width="{img_width}" height="{img_height}"/>
</div>
</body>
</html>"""

            book.add_item(c)
            chapters.append(c)

        # テンポラリーディレクトリ cleanup
        shutil.rmtree(temp_dir)

    else:
        # テキストベース変換
        try:
            md_text = pymupdf4llm.to_markdown(str(pdf_path))
        except Exception:
            console.print("[yellow] テキスト抽出失敗、画像モードで再試行します[/yellow]")
            doc.close()
            return pdf_to_epub(pdf_path, output_dir, use_images=True)

        c1 = epub.EpubHtml(title="Content", file_name="content.xhtml", lang="ja")
        c1.content = f"""<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>{pdf_path.stem}</title>
    <meta charset="utf-8"/>
    <style>
        body {{ font-family: "YuMincho", "Yu Mincho", serif; line-height: 1.8; }}
        h1 {{ font-size: 1.5em; margin: 1em 0; }}
        h2 {{ font-size: 1.3em; margin: 0.8em 0; }}
        p {{ margin: 0.5em 0; text-indent: 1em; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
{md_text}
</body>
</html>"""

        book.add_item(c1)
        chapters.append(c1)

    doc.close()
    book.toc = chapters
    book.add_item(epub.EpubNcx())

    # spine 設定
    book.spine = ['nav', chapters] if not use_images else chapters

    # EPUB 保存
    epub.write_epub(output_path, book, {})

    return str(output_path)


def images_to_epub(images_dir: str, output_dir: str = None, title: str = None) -> str:
    """画像フォルダを EPUB に変換"""
    images_dir = Path(images_dir).resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f" 画像フォルダが見つかりません：{images_dir}")

    if output_dir is None:
        output_dir = Path.cwd() / "epub"
    else:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    if title is None:
        title = images_dir.name

    output_path = output_dir / f"{title}.epub"

    # 画像ファイル一覧取得（再帰的）
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    image_files = sorted([
        f for f in images_dir.rglob("*")
        if f.suffix.lower() in image_extensions
    ])

    if not image_files:
        raise ValueError(f" 画像ファイルが見つかりません：{images_dir}")

    console.print(f"[cyan] 画像 {len(image_files)} 枚を EPUB 化：{title}[/cyan]")

    # EPUB 作成
    book = epub.EpubBook()
    book.set_identifier(f"images-convert-{title}")
    book.set_title(title)
    book.set_language("ja")
    book.add_author("Image Converter")

    chapters = []

    for i, img_path in enumerate(image_files):
        img_item = epub.EpubImage()
        img_item.file_name = f"images/{img_path.name}"
        img_item.media_type = f"image/{img_path.suffix.lower().lstrip('.')}"

        with open(img_path, "rb") as f:
            img_item.content = f.read()

        book.add_item(img_item)

        c = epub.EpubHtml(title=img_path.stem, file_name=f"page_{i:03d}.xhtml", lang="ja")
        c.content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>{img_path.stem}</title>
    <meta charset="utf-8"/>
    <style>
        body {{
            font-family: "YuMincho", "Yu Mincho", serif;
            line-height: 1.8;
            text-align: center;
            padding: 20px;
        }}
        img {{
            max-width: 100%;
            max-height: 90vh;
            object-fit: contain;
        }}
        .caption {{
            margin-top: 10px;
            font-size: 0.9em;
            color: #666;
        }}
    </style>
</head>
<body>
    <img src="images/{img_path.name}" alt="{img_path.stem}"/>
    <div class="caption">{img_path.stem}</div>
</body>
</html>"""

        book.add_item(c)
        chapters.append(c)

    book.toc = chapters
    book.add_item(epub.EpubNcx())
    book.spine = ['nav'] + [c.id for c in chapters]

    epub.write_epub(output_path, book, {})

    return str(output_path)


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


def select_device() -> str:
    """デバイス最適化選択"""
    console.print(Panel.fit(
        "[bold] デバイスを選んでね[/bold]\n",
        border_style="cyan"
    ))

    devices = [
        ("1", "XTEINK X4（480x800）←おすすめ"),
        ("2", "XTEINK X3（528x792）"),
        ("3", "Raw（高解像度維持・タブレット向け）"),
    ]

    for key, desc in devices:
        console.print(f"  [yellow]{key}[/yellow]: {desc}")

    device_map = {"1": "x4", "2": "x3", "3": "raw"}
    choice = Prompt.ask("\n選択", choices=["1", "2", "3"], default="1")
    return device_map[choice]


def select_sharpen() -> bool:
    """輪郭強調の要否を選択"""
    console.print(Panel.fit(
        "[bold] 輪郭強調をしますか？[/bold]\n"
        "漫画・文字 PDF は ON がおすすめ",
        border_style="yellow"
    ))

    choice = Prompt.ask("\n選択", choices=["y", "n"], default="y")
    return choice.lower() == "y"


def select_vectorize() -> bool:
    """ベクトル風変換の要否を選択"""
    console.print(Panel.fit(
        "[bold] ベクトル風変換（線画モード）[/bold]\n"
        "文字・線画をくっきり表示（試験的機能）\n"
        "※グレースケールが失われ、輪郭ベースの表現になります",
        border_style="magenta"
    ))

    choice = Prompt.ask("\n選択", choices=["y", "n"], default="n")
    return choice.lower() == "y"


def select_mode() -> str:
    """変換モード選択"""
    console.print(Panel.fit(
        "[bold] 変換モードを選んでね[/bold]\n",
        border_style="cyan"
    ))

    modes = [
        ("1", "PDF → EPUB 変換（単体・テキスト）"),
        ("2", "PDF → EPUB 変換（単体・画像/漫画）"),
        ("3", "PDF → EPUB 変換（まとめて）"),
        ("4", "ZIP 内の画像 → EPUB 変換（単体）"),
        ("5", "ZIP 内の画像 → EPUB 変換（まとめて）"),
        ("6", "画像フォルダ → EPUB 変換"),
        ("7", "終了"),
    ]

    for key, desc in modes:
        console.print(f"  [yellow]{key}[/yellow]: {desc}")

    return Prompt.ask("\n選択", choices=["1", "2", "3", "4", "5", "6", "7"], default="1")


def main():
    """メインループ"""
    console.print(Panel.fit(
        "[bold magenta]📚 PDF/画像 → EPUB 変換ツール[/bold magenta]",
        border_style="magenta"
    ))
    console.print()

    work_dir = Path.cwd()
    pdf_dir = work_dir / "pdf"
    output_dir = work_dir / "epub"
    output_dir.mkdir(exist_ok=True)
    pdf_dir.mkdir(exist_ok=True)

    while True:
        mode = select_mode()

        if mode == "1":
            console.print("\n[yellow]PDF ファイルを選んでね[/yellow]")
            console.print(f"  参照フォルダ：{pdf_dir}")

            pdf_files = list(pdf_dir.glob("*.pdf"))
            if pdf_files:
                console.print("\n  見つかった PDF:")
                for i, pdf in enumerate(pdf_files[:10], 1):
                    console.print(f"    [{i}] {pdf.name}")
                if len(pdf_files) > 10:
                    console.print(f"    ... ほか {len(pdf_files) - 10} 個")

            pdf_input = Prompt.ask("\nPDF ファイル名（または番号）")

            if pdf_input.isdigit():
                idx = int(pdf_input) - 1
                if 0 <= idx < len(pdf_files):
                    pdf_path = pdf_files[idx]
                else:
                    console.print("[red] 無効な選択[/red]")
                    continue
            else:
                pdf_path = pdf_dir / pdf_input

            try:
                output_path = pdf_to_epub(pdf_path, output_dir, use_images=False)
                console.print(f"\n[green]✓ 変換完了：{output_path}[/green]")
                console.print(f"  [dim]epub/フォルダに保存されました[/dim]")
            except Exception as e:
                console.print(f"[red] エラー：{e}[/red]")

        elif mode == "2":
            device = select_device()

            console.print("\n[yellow]PDF ファイルを選んでね（漫画モード）[/yellow]")
            console.print(f"  参照フォルダ：{pdf_dir}")

            # 処理モード選択
            console.print(Panel.fit(
                "[bold] 処理モードを選んでね[/bold]",
                border_style="cyan"
            ))
            modes = [
                ("1", "通常（輪郭強調のみ）"),
                ("2", "ベクトル風変換（試験的・文字向け）"),
            ]
            for key, desc in modes:
                console.print(f"  [yellow]{key}[/yellow]: {desc}")
            mode_choice = Prompt.ask("\n選択", choices=["1", "2"], default="1")

            sharpen = mode_choice == "1"
            vectorize = mode_choice == "2"

            pdf_files = list(pdf_dir.glob("*.pdf"))
            if pdf_files:
                console.print("\n  見つかった PDF:")
                for i, pdf in enumerate(pdf_files[:10], 1):
                    console.print(f"    [{i}] {pdf.name}")
                if len(pdf_files) > 10:
                    console.print(f"    ... ほか {len(pdf_files) - 10} 個")

            pdf_input = Prompt.ask("\nPDF ファイル名（または番号）")

            if pdf_input.isdigit():
                idx = int(pdf_input) - 1
                if 0 <= idx < len(pdf_files):
                    pdf_path = pdf_files[idx]
                else:
                    console.print("[red] 無効な選択[/red]")
                    continue
            else:
                pdf_path = pdf_dir / pdf_input

            try:
                output_path = pdf_to_epub(pdf_path, output_dir, use_images=True, device=device, sharpen=sharpen)
                console.print(f"\n[green]✓ 変換完了：{output_path}[/green]")
                console.print(f"  [dim]epub/フォルダに保存されました[/dim]")
            except Exception as e:
                import traceback
                console.print(f"[red] エラー：{e}[/red]")
                console.print(f"[dim]{traceback.format_exc()}[/dim]")

        elif mode == "3":
            console.print("\n[yellow]PDF をまとめて変換するよ[/yellow]")
            console.print(f"  参照フォルダ：{pdf_dir}")

            pdf_files = list(pdf_dir.glob("*.pdf"))
            if not pdf_files:
                console.print("[red]PDF ファイルが見つかりません[/red]")
                continue

            console.print("\n  見つかった PDF:")
            for i, pdf in enumerate(pdf_files, 1):
                console.print(f"    [{i}] {pdf.name}")

            use_images = Confirm.ask("\n漫画 PDF を変換しますか？（画像モード）", default=False)

            device = 'raw'
            sharpen = False
            if use_images:
                device = select_device()
                sharpen = select_sharpen()

            selection = Prompt.ask(
                "\n変換するファイルを選択（例：1,3,5 または 1-5、すべては all）",
                default="all"
            )

            selected_files = []
            if selection.lower() == "all":
                selected_files = pdf_files
            else:
                for part in selection.split(","):
                    part = part.strip()
                    if "-" in part:
                        start, end = map(int, part.split("-"))
                        selected_files.extend(pdf_files[start-1:end])
                    elif part.isdigit():
                        idx = int(part) - 1
                        if 0 <= idx < len(pdf_files):
                            selected_files.append(pdf_files[idx])

            if not selected_files:
                console.print("[red] 有効な選択がありません[/red]")
                continue

            console.print(f"\n[cyan]{len(selected_files)} 個の PDF を変換します[/cyan]")

            success_count = 0
            error_count = 0

            with Progress() as progress:
                task = progress.add_task("[cyan] 変換中...", total=len(selected_files))
                for pdf_path in selected_files:
                    progress.update(task, description=f"[cyan]{pdf_path.name}[/cyan]")
                    try:
                        output_path = pdf_to_epub(pdf_path, output_dir, use_images=use_images, device=device, sharpen=sharpen)
                        console.print(f"  [green]✓ {pdf_path.name} → {output_path}[/green]")
                        success_count += 1
                    except Exception as e:
                        console.print(f"  [red]✗ {pdf_path.name}: {e}[/red]")
                        error_count += 1
                    progress.advance(task)

            console.print(f"\n[green] 完了：{success_count} 個成功，{error_count} 個失敗[/green]")

        elif mode == "4":
            console.print("\n[yellow]ZIP ファイルを選んでね[/yellow]")
            console.print(f"  参照フォルダ：{pdf_dir}")

            zip_files = list(pdf_dir.glob("*.zip"))
            if zip_files:
                console.print("\n  見つかった ZIP:")
                for i, zip_file in enumerate(zip_files[:10], 1):
                    console.print(f"    [{i}] {zip_file.name}")
                if len(zip_files) > 10:
                    console.print(f"    ... ほか {len(zip_files) - 10} 個")

            zip_input = Prompt.ask("\nZIP ファイル名（または番号）")

            if zip_input.isdigit():
                idx = int(zip_input) - 1
                if 0 <= idx < len(zip_files):
                    zip_path = zip_files[idx]
                else:
                    console.print("[red] 無効な選択[/red]")
                    continue
            else:
                zip_path = pdf_dir / zip_input

            try:
                console.print("[cyan] 画像を抽出中...[/cyan]")
                extract_dir = extract_zip_images(zip_path)

                device = select_device()

                output_path = images_to_epub(extract_dir, output_dir, title=zip_path.stem)
                console.print(f"\n[green]✓ 変換完了：{output_path}[/green]")

                shutil.rmtree(extract_dir)
            except Exception as e:
                console.print(f"[red] エラー：{e}[/red]")

        elif mode == "5":
            console.print("\n[yellow]ZIP ファイルをまとめて変換するよ[/yellow]")
            console.print(f"  参照フォルダ：{pdf_dir}")

            zip_files = list(pdf_dir.glob("*.zip"))
            if not zip_files:
                console.print("[red]ZIP ファイルが見つかりません[/red]")
                continue

            console.print("\n  見つかった ZIP:")
            for i, zip_file in enumerate(zip_files, 1):
                console.print(f"    [{i}] {zip_file.name}")

            selection = Prompt.ask(
                "\n変換するファイルを選択（例：1,3,5 または 1-5、すべては all）",
                default="all"
            )

            selected_files = []
            if selection.lower() == "all":
                selected_files = zip_files
            else:
                for part in selection.split(","):
                    part = part.strip()
                    if "-" in part:
                        start, end = map(int, part.split("-"))
                        selected_files.extend(zip_files[start-1:end])
                    elif part.isdigit():
                        idx = int(part) - 1
                        if 0 <= idx < len(zip_files):
                            selected_files.append(zip_files[idx])

            if not selected_files:
                console.print("[red] 有効な選択がありません[/red]")
                continue

            device = select_device()

            console.print(f"\n[cyan]{len(selected_files)} 個の ZIP を変換します[/cyan]")

            success_count = 0
            error_count = 0

            for zip_path in selected_files:
                try:
                    extract_dir = extract_zip_images(zip_path)
                    output_path = images_to_epub(extract_dir, output_dir, title=zip_path.stem)
                    console.print(f"  [green]✓ {zip_path.name} → {output_path}[/green]")
                    shutil.rmtree(extract_dir)
                    success_count += 1
                except Exception as e:
                    console.print(f"  [red]✗ {zip_path.name}: {e}[/red]")
                    error_count += 1

            console.print(f"\n[green] 完了：{success_count} 個成功，{error_count} 個失敗[/green]")

        elif mode == "6":
            # デフォルト画像フォルダ
            default_images = Path.cwd() / "images"
            if not default_images.exists():
                default_images = Path.home() / "Pictures"

            console.print("\n[yellow]画像フォルダを選んでね[/yellow]")
            console.print(f"  デフォルト：{default_images}")

            folder_input = Prompt.ask("フォルダ名（絶対パスでも可）", default=str(default_images))
            images_path = Path(folder_input)

            if not images_path.exists():
                console.print("[red] フォルダが見つかりません[/red]")
                continue

            try:
                output_path = images_to_epub(images_path, output_dir)
                console.print(f"\n[green]✓ 変換完了：{output_path}[/green]")
            except Exception as e:
                console.print(f"[red] エラー：{e}[/red]")

        elif mode == "7":
            console.print("\n[magenta]bye[/magenta]")
            break

        else:
            console.print("[red] 無効な選択です[/red]")


if __name__ == "__main__":
    main()
