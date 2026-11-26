"""Build the Traditional Chinese PDF from Markdown and verified local results."""

import argparse
from html import escape
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from PIL import Image as PillowImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from image_sharpening.algorithms import LABELS
from image_sharpening.artifacts import PDF_NAME, pdf_subject, read_student, verify_results

ROOT = Path(__file__).resolve().parents[1]
INK = colors.HexColor("#1e2e40")
NAVY = colors.HexColor("#163b58")
MUTED = colors.HexColor("#50667a")
PALE = colors.HexColor("#eef3f8")
CONTENT = A4[0] - 92
METHOD_LABELS_ZH = {
    "original": "原圖",
    "laplacian": "拉普拉斯算子",
    "unsharp": "反遮罩",
    "high_boost": "高提升濾波",
    "homomorphic": "同態濾波",
    "combined": "頻域組合增強",
}


def register_fonts(regular: Path | None, bold: Path | None):
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    regular = regular or fonts / "msjh.ttc"
    bold = bold or (fonts / "msjhbd.ttc" if regular == fonts / "msjh.ttc" else regular)
    for name, file in (("TC", regular), ("TC-Bold", bold)):
        if not file.is_file():
            raise ValueError("Provide a Traditional Chinese TrueType font with --font and optional --bold-font.")
        pdfmetrics.registerFont(TTFont(name, str(file), subfontIndex=0))
    pdfmetrics.registerFontFamily("TC", normal="TC", bold="TC-Bold", italic="TC", boldItalic="TC-Bold")


def layout_styles():
    body = ParagraphStyle("Body", fontName="TC", fontSize=10.3, leading=17, wordWrap="CJK",
                          textColor=INK, spaceAfter=7)
    return {
        "body": body,
        "h2": ParagraphStyle("H2", parent=body, fontName="TC-Bold", fontSize=19, leading=26,
                             spaceAfter=13, keepWithNext=True, textColor=NAVY),
        "h3": ParagraphStyle("H3", parent=body, fontName="TC-Bold", fontSize=12, leading=18,
                             spaceBefore=8, keepWithNext=True, textColor=NAVY),
        "title": ParagraphStyle("Title", parent=body, fontName="TC-Bold", fontSize=27,
                                leading=39, spaceAfter=19, textColor=NAVY),
        "caption": ParagraphStyle("Caption", parent=body, fontSize=8.1, leading=12, textColor=MUTED),
        "cell": ParagraphStyle("Cell", parent=body, fontSize=8.3, leading=10.5, spaceAfter=0),
        "head": ParagraphStyle("Head", parent=body, fontName="TC-Bold", fontSize=8.3,
                               leading=10.5, textColor=colors.white, spaceAfter=0),
        "code": ParagraphStyle("Code", fontName="Courier", fontSize=8.5, leading=12,
                               textColor=NAVY, leftIndent=14, spaceAfter=9, spaceBefore=3),
    }


def rich(text):
    return re.sub(r"`([^`]+)`", r"<font color='#163b58'>\1</font>", escape(text))


def make_table(rows, styles, widths=None):
    widths = widths or [1 / len(rows[0])] * len(rows[0])
    cells = [[Paragraph(rich(str(cell)), styles["head" if i == 0 else "cell"])
              for cell in row] for i, row in enumerate(rows)]
    table = Table(cells, colWidths=[w * CONTENT for w in widths], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PALE, colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, -1), (-1, -1), .4, PALE),
    ]))
    return table


def figure(path: Path, max_height: float):
    with PillowImage.open(path) as image:
        ratio = min(CONTENT / image.width, max_height / image.height)
        return Image(str(path), width=image.width * ratio, height=image.height * ratio)


def image_story(name, outputs, record, styles):
    stem = Path(name).stem
    rows = [["方法", "均值", "標準差", "平均梯度", "L 變異數", "黑 / 白 (%)", "飽和 (%)", "耗時 ms"]]
    for key, m in record["metrics"].items():
        timing = f"{record['runtime_ms'][key]:.2f}" if key in record["runtime_ms"] else "—"
        rows.append([f"{LABELS[key]}（{METHOD_LABELS_ZH[key]}）", f"{m['mean']:.2f}",
                     f"{m['std']:.2f}", f"{m['mean_gradient']:.2f}",
                     f"{m['laplacian_variance']:.2f}", f"{m['black_pct']:.3f} / {m['white_pct']:.3f}",
                     f"{m['saturated_pct']:.3f}", timing])
    note = (f"{stem} 演算法共用階段｜高斯細節 {record['runtime_ms']['shared_gaussian']:.2f} ms，"
            f"正向 FFT 與 Homomorphic（同態濾波）增益 {record['runtime_ms']['shared_frequency']:.2f} ms。")
    return [
        figure(outputs / "comparisons" / f"{stem}_comparison.png", 263),
        Paragraph("原圖與五種增強圖｜相同 0 至 255 灰階顯示。", styles["caption"]),
        figure(outputs / "spectra" / f"{stem}_spectrum.png", 160),
        Paragraph("中心化 log(1 + |Z|) 與沿 v = 0 的實際濾波增益。", styles["caption"]),
        make_table(rows, styles, [.26, .085, .08, .10, .10, .155, .105, .115]),
        Spacer(1, 5),
        Paragraph(note, styles["caption"]),
    ]


def dynamic_blocks(results, outputs, styles):
    records = results["images"]
    rows = [["影像", "α / 鄰域", "m / σ", "k", "γL / γH / D0 / c", "Db / kb"]]
    for name, record in records.items():
        p = record["parameters"]
        rows.append([name, f"{p['alpha']:.2f} / {p['neighbors']}", f"{p['size']} / {p['sigma']:.2f}",
                     f"{p['high_boost']:.2f}",
                     f"{p['gamma_l']:.2f} / {p['gamma_h']:.2f} / {p['cutoff']:.3f} / {p['c']:.1f}",
                     f"{p['frequency_cutoff']:.3f} / {p['frequency_gain']:.2f}"])
    total = sum(sum(record["runtime_ms"].values()) for record in records.values()) / 1000
    changes = []
    gain = []
    saturation = []
    for record in records.values():
        metrics = record["metrics"]
        changes.append(metrics["combined"]["mean"] - metrics["original"]["mean"])
        gain.append(metrics["combined"]["mean_gradient"] - metrics["homomorphic"]["mean_gradient"])
        saturation.append(metrics["high_boost"]["saturated_pct"])
    timing = (f"本次批次耗時 {results['processing_seconds']:.2f} 秒，包含讀檔、量測、PNG 編碼及繪圖。"
              f"各影像記錄的演算法階段合計 {total:.2f} 秒。單法耗時與共用步驟分開列出，"
              "時間為本機此次量測。")
    discussion = (f"六張圖的 Combined（頻域組合增強）平均灰階較原圖改變 {min(changes):+.2f} 至 {max(changes):+.2f} 級，"
                  f"平均梯度較 Homomorphic（同態濾波）增加 {min(gain):.2f} 至 {max(gain):.2f}。"
                  f"空間域的 High-boost（高提升濾波）兩端總飽和率介於 {min(saturation):.3f}% 至 {max(saturation):.3f}%。")
    blocks = {"parameters": [make_table(rows, styles, [.12, .13, .13, .09, .32, .21]), Spacer(1, 6)],
              "timing": [Paragraph(timing, styles["body"])],
              "discussion": [Paragraph(discussion, styles["body"])]}
    blocks.update({f"image:{name}": image_story(name, outputs, record, styles)
                   for name, record in records.items()})
    return blocks


def markdown_story(lines, styles, blocks):
    story = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line:
            continue
        if line == "<!-- page -->":
            story.append(PageBreak())
        elif line.startswith("### "):
            story.append(Paragraph(rich(line[4:]), styles["h3"]))
        elif line.startswith("## "):
            story.append(Paragraph(rich(line[3:]), styles["h2"]))
        elif re.fullmatch(r"https://\S+", line):
            url = escape(line)
            story.append(Paragraph(f'<link href="{url}" color="{NAVY.hexval()}">{url}</link>', styles["body"]))
        elif line == "```":
            code = []
            while index < len(lines) and lines[index].strip() != "```":
                code.append(lines[index])
                index += 1
            if index == len(lines):
                raise ValueError("Unclosed Markdown code block.")
            index += 1
            story.append(Preformatted("\n".join(code), styles["code"]))
        elif line.startswith("|"):
            rows = []
            while True:
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    rows.append(cells)
                if index == len(lines) or not lines[index].strip().startswith("|"):
                    break
                line = lines[index].strip()
                index += 1
            story.append(KeepTogether([make_table(rows, styles), Spacer(1, 6)]))
        elif line.startswith("{{") and line.endswith("}}"):
            story.extend(blocks[line[2:-2]])
        else:
            story.append(Paragraph(rich(line), styles["body"]))
    return story


def footer(canvas, doc):
    if doc.page == 1:
        return
    canvas.setStrokeColor(PALE)
    canvas.line(46, 39, A4[0] - 46, 39)
    canvas.setFont("TC", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(46, 26, "影像處理作業二  /  IMAGE SHARPENING")
    canvas.drawRightString(A4[0] - 46, 26, str(doc.page))


def build(project, inputs, outputs, config, student_file, target, font=None, bold_font=None):
    results = verify_results(project, inputs, outputs, config)
    student = read_student(student_file)
    register_fonts(font, bold_font)
    styles = layout_styles()
    source = (project / "report/report.md").read_text(encoding="utf-8").splitlines()
    if not source or not source[0].startswith("# "):
        raise ValueError("Start report.md with a Markdown level-one title.")
    title = source[0][2:]
    cover = [Spacer(1, 120), Paragraph("IMAGE PROCESSING  /  HOMEWORK 02", styles["caption"]),
             Paragraph(rich(title), styles["title"]), Paragraph("Image Sharpening", styles["h2"]), Spacer(1, 32)]
    for label, value in [("姓名", student["name"]), ("學號", student["student_id"])]:
        cover.append(Paragraph(rich(f"{label}　{value}"), styles["body"]))
    cover.append(PageBreak())
    story = cover + markdown_story(source[1:], styles, dynamic_blocks(results, outputs, styles))
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as temporary:
        staged = Path(temporary) / target.name
        doc = SimpleDocTemplate(str(staged), pagesize=A4, leftMargin=46, rightMargin=46,
                                topMargin=42, bottomMargin=48, title=title, author=student["name"],
                                subject=pdf_subject(project, outputs, student_file))
        doc.build(story, onFirstPage=footer, onLaterPages=footer)
        staged.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "HW2_test_image")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/selected.json")
    parser.add_argument("--student-file", type=Path, default=ROOT / "report/student.json")
    parser.add_argument("--output", type=Path, default=ROOT / "report" / PDF_NAME)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--bold-font", type=Path)
    args = parser.parse_args()
    try:
        build(ROOT, args.input, args.outputs, args.config, args.student_file, args.output, args.font, args.bold_font)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Report unavailable: {error}\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
