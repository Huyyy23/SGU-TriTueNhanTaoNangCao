#!/usr/bin/env python3
"""Build report/bao_cao.docx and report/bao_cao.pdf from report/report.md.

Input contract: UTF-8 Markdown; supports # through ### headings, pipe tables
(with a separator row), ![caption](PNG-path) images, unordered/numbered lists,
and plain paragraphs. Image paths are relative to report.md. Equations remain
ordinary text; raw HTML and nested lists are not interpreted. LibreOffice
renders the report and PyMuPDF resolves the static page-numbered indexes.
"""

from __future__ import annotations

import json
import re
import sys
import fitz
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "report.md"
DOCX = ROOT / "bao_cao.docx"
PDF = ROOT / "bao_cao.pdf"
IMAGE_RE = re.compile(r"!\[([^]]*)\]\(([^)]+)\)")


def field(paragraph, instruction: str, result: str = "") -> None:
    """Append a Word field that can be refreshed by Word/LibreOffice."""
    for kind, value in (("begin", None), ("instrText", instruction), ("separate", None), ("end", None)):
        node = OxmlElement("w:fldChar" if kind != "instrText" else "w:instrText")
        if kind == "instrText":
            node.set(qn("xml:space"), "preserve")
            node.text = value
        else:
            node.set(qn("w:fldCharType"), kind)
        run = OxmlElement("w:r")
        run.append(node)
        paragraph._p.append(run)
        if kind == "separate" and result:
            result_run = OxmlElement("w:r")
            text = OxmlElement("w:t")
            text.text = result
            result_run.append(text)
            paragraph._p.append(result_run)


def set_cell_shading(cell, fill: str) -> None:
    props = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    props.append(shading)


def configure(document: Document) -> None:
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin, section.bottom_margin = Cm(2.4), Cm(2.2)
    section.left_margin, section.right_margin = Cm(2.7), Cm(2.2)

    for name in ("Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Heading 3"):
        style = document.styles[name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    document.styles["Normal"].font.size = Pt(12)
    for name, size in (("Heading 1", 16), ("Heading 2", 14), ("Heading 3", 12)):
        document.styles[name].font.size = Pt(size)
        document.styles[name].font.bold = True

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Trang ")
    field(footer, " PAGE ")


def add_contents(document: Document, title: str, indexes) -> None:
    cover = document.add_paragraph()
    cover.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cover.paragraph_format.space_before = Pt(145)
    run = cover.add_run(title)
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(24)
    sub = document.add_paragraph("BÁO CÁO")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].font.size = Pt(16)
    sub.paragraph_format.space_before = Pt(18)
    document.add_page_break()

    for heading, entries in zip(("Mục lục", "Danh mục hình", "Danh mục bảng"), indexes):
        document.add_heading(heading, level=1)
        for text, page in entries:
            p = document.add_paragraph()
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.tab_stops.add_tab_stop(Cm(15.5), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
            run = p.add_run(f"{text}\t{page}")
            run.font.size = Pt(10)
        document.add_page_break()


def add_caption(document: Document, label: str, caption: str, number: int) -> None:
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(f"{label} ")
    field(p, f" SEQ {label} \\* ARABIC ", str(number))
    p.add_run(f": {caption}")
    p.paragraph_format.keep_with_next = True


def parse_table_row(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]



def build(indexes) -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Không tìm thấy Markdown đầu vào: {SOURCE}")
    source = SOURCE.read_text(encoding="utf-8-sig")
    source = re.sub(r"(?<!!)\[([^]]+)\]\(([^)]+)\)", r"\1 (\2)", source)
    source = source.replace("**", "").replace("`", "")
    lines = source.splitlines()
    title = "GOMOKU 11×11 VÀ ALPHAZERO"

    doc = Document()
    configure(doc)
    add_contents(doc, title, indexes)
    figures = tables = 0
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        heading = re.match(r"^(#{1,3})\s+(.+?)\s*#*\s*$", line)
        if heading:
            text = heading.group(2).strip()
            # The title is already on the cover; keep it as the first document heading too.
            doc.add_heading(text, level=len(heading.group(1)))
            i += 1
            continue
        image = IMAGE_RE.fullmatch(line)
        if image:
            caption, path_text = image.groups()
            caption = re.sub(r"^Hình\s+\d+\s*:\s*", "", caption, flags=re.IGNORECASE)
            path = (SOURCE.parent / path_text.strip()).resolve()
            if path.suffix.lower() != ".png" or not path.is_file():
                raise FileNotFoundError(f"Ảnh PNG không tồn tại hoặc không hỗ trợ: {path}")
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(str(path), width=Cm(15.5))
            figures += 1
            add_caption(doc, "Hình", caption or path.stem, figures)
            i += 1
            continue
        if "|" in line and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            headers = parse_table_row(line)
            i += 2
            rows = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(parse_table_row(lines[i]))
                i += 1
            table = doc.add_table(rows=1, cols=len(headers))
            table.style = "Table Grid"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            header = OxmlElement("w:tblHeader")
            header.set(qn("w:val"), "true")
            table.rows[0]._tr.get_or_add_trPr().append(header)
            for idx, text in enumerate(headers):
                cell = table.rows[0].cells[idx]
                cell.text = text
                set_cell_shading(cell, "D9EAF7")
                for run in cell.paragraphs[0].runs:
                    run.bold = True
            for row in rows:
                cells = table.add_row().cells
                for idx, cell in enumerate(cells):
                    cell.text = row[idx] if idx < len(row) else ""
                    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tables += 1
            add_caption(doc, "Bảng", " — ".join(headers[:3]), tables)
            continue
        bullet = re.match(r"^\s*[-*+]\s+(.+)$", line)
        numbered = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        if bullet or numbered:
            doc.add_paragraph((bullet or numbered).group(1), style="List Bullet" if bullet else "List Number")
            i += 1
            continue
        # Consecutive plain lines form one paragraph; inline syntax is preserved literally.
        paragraph_lines = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,3})\s+", lines[i].strip()) and not IMAGE_RE.fullmatch(lines[i].strip()) and not ("|" in lines[i] and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1])) and not re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", lines[i]):
            paragraph_lines.append(lines[i].strip())
            i += 1
        doc.add_paragraph(" ".join(paragraph_lines))

    settings = doc.settings._element
    update = OxmlElement("w:updateFields")
    update.set(qn("w:val"), "true")
    settings.append(update)
    doc.save(DOCX)


def collect_indexes():
    source = SOURCE.read_text(encoding="utf-8-sig")
    source = re.sub(r"(?<!!)\[([^]]+)\]\(([^)]+)\)", r"\1 (\2)", source)
    lines = source.replace("**", "").replace("`", "").splitlines()
    headings = [re.sub(r"^#{1,2}\s+", "", line).strip()
                for line in lines if re.match(r"^#{1,2}\s+", line)]
    figures = []
    tables = []
    for i, line in enumerate(lines):
        image = IMAGE_RE.fullmatch(line.strip())
        if image:
            caption = re.sub(r"^Hình\s+\d+\s*:\s*", "", image.group(1), flags=re.IGNORECASE)
            figures.append(f"Hình {len(figures) + 1}: {caption}")
        if "|" in line and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            tables.append(f"Bảng {len(tables) + 1}: " + " — ".join(parse_table_row(line)[:3]))

    with fitz.open(PDF) as pdf:
        pages = [" ".join(page.get_text().split()) for page in pdf]
    start = next(i for i, text in enumerate(pages) if "Gomoku là trò chơi đối kháng" in text)

    def page_for(query):
        query = " ".join(query.split())
        for i in range(start, len(pages)):
            if query in pages[i]:
                return i + 1
        raise ValueError(f"Không tìm thấy trong PDF: {query}")

    return (
        [(heading, page_for(heading)) for heading in headings],
        [(caption, page_for(f"Hình {i}:")) for i, caption in enumerate(figures, 1)],
        [(caption, page_for(f"Bảng {i}:")) for i, caption in enumerate(tables, 1)],
    )


if __name__ == "__main__":
    index_file = ROOT / "_indexes.json"
    previous = json.loads(index_file.read_text(encoding="utf-8")) if index_file.exists() else [[], [], []]
    if sys.argv[1:] == ["--build"]:
        build(previous)
    elif sys.argv[1:] == ["--index"]:
        current = json.loads(json.dumps(collect_indexes(), ensure_ascii=False))
        index_file.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
        sys.exit(0 if current == previous else 3)
    else:
        raise SystemExit("Use --build or --index; run build_report.sh for the complete export.")
