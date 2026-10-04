"""Everything that talks to PyMuPDF or reasons about page geometry lives here.

The rest of the ingestion code only sees plain Python objects (RawPage,
Block, Table) and strings, so it can be read without knowing PyMuPDF.

Units: PDF points (1/72 inch), origin at the top-left, y grows downwards.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

pymupdf.no_recommend_layout()  # silence the "consider pymupdf_layout" notice printed by find_tables()

# First cell of the boxed header that every SOP-template page starts with.
SOP_HEADER_MARKER = "Policy & Standard Operating Procedure"

# Vertical gap (points) below which two consecutive text blocks are treated as
# one wrapped paragraph rather than separate paragraphs. Working Hours Policy
# stores every visual line as its own block with ~6pt gaps; real paragraph
# gaps in all four PDFs are >= 8pt.
SOFT_WRAP_GAP = 7.0

# Horizontal gap (points) that separates two real columns, e.g. "Role" and
# "Key Responsibilities" in the BYOM Responsibility Matrix (measured 49-150pt).
# Smaller same-row gaps are justified text that PyMuPDF split into pieces
# (measured 27.7pt), so those pieces are joined with a plain space.
MIN_COLUMN_GAP = 40.0

# Text that is only a list marker: "●", "o", "4.1.", "a." ... When such a marker
# sits on the same baseline as the text after it, they belong on one line.
_MARKER_ONLY = re.compile(r"^([●○•▪o]|\d+(\.\d+)*\.?|[a-z]\.)$")


@dataclass
class Line:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass
class Block:
    """A PyMuPDF text block: usually one paragraph, list item or table cell."""

    lines: list[Line]

    @property
    def x0(self) -> float:
        return min(l.x0 for l in self.lines)

    @property
    def y0(self) -> float:
        return min(l.y0 for l in self.lines)

    @property
    def x1(self) -> float:
        return max(l.x1 for l in self.lines)

    @property
    def y1(self) -> float:
        return max(l.y1 for l in self.lines)

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def text(self) -> str:
        return _join_lines(self.lines)


@dataclass
class Table:
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    rows: list[list[str]]  # merged/missing cells come back as ""

    def contains(self, block: Block) -> bool:
        x0, y0, x1, y1 = self.bbox
        cx = (block.x0 + block.x1) / 2
        return x0 <= cx <= x1 and y0 <= block.center_y <= y1


@dataclass
class RawPage:
    number: int  # 1-based, as a human would cite it
    raw_text: str  # page.get_text(): what PyMuPDF gives with no help from us
    blocks: list[Block]  # all non-empty text blocks, in PyMuPDF reading order
    tables: list[Table]  # tables found by find_tables(), excluding the SOP header
    header: Table | None  # the SOP header box, if this page has one


def read_pdf(path: Path) -> list[RawPage]:
    with pymupdf.open(path) as doc:
        return [_read_page(page) for page in doc]


def _read_page(page: pymupdf.Page) -> RawPage:
    header = None
    tables = []
    for found in page.find_tables().tables:
        table = Table(tuple(found.bbox), [[cell or "" for cell in row] for row in found.extract()])
        if table.rows and table.rows[0][0].strip().startswith(SOP_HEADER_MARKER):
            header = table
        else:
            tables.append(table)

    return RawPage(
        number=page.number + 1,
        raw_text=page.get_text(),
        blocks=_text_blocks(page),
        tables=tables,
        header=header,
    )


def _text_blocks(page: pymupdf.Page) -> list[Block]:
    blocks = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:  # 1 = image (the logo)
            continue
        lines = [
            Line(*l["bbox"], text="".join(span["text"] for span in l["spans"]))
            for l in b["lines"]
        ]
        lines = [l for l in lines if l.text.strip(" \u200b")]
        if lines:
            blocks.append(Block(lines))
    return blocks


def _same_row(left_y0: float, left_x1: float, right_y0: float, right_x0: float) -> bool:
    """True when `right` starts on the same baseline as `left`, to its right (a second column)."""
    return abs(left_y0 - right_y0) < 2.5 and right_x0 > left_x1 + 2


def _column_separator(left_text: str, gap: float) -> str:
    # "● Night Shift..." / "4.1. Each employee..." -> plain space;
    # pieces of one justified line (gap < MIN_COLUMN_GAP) -> plain space;
    # "Employee | Submit BYOM requests..." (two-column layout) -> " | ".
    if _MARKER_ONLY.match(left_text.strip(" \u200b")) or gap < MIN_COLUMN_GAP:
        return " "
    return " | "


def _join_lines(lines: list[Line]) -> str:
    out = lines[0].text
    for prev, line in zip(lines, lines[1:]):
        if _same_row(prev.y0, prev.x1, line.y0, line.x0):
            out += _column_separator(prev.text, line.x0 - prev.x1) + line.text
        else:
            out += "\n" + line.text
    return out


def body_text(page: RawPage) -> str:
    """Page text with the SOP header removed and tables rendered as Markdown.

    Paragraph breaks are "\\n\\n"; single "\\n" means a visual line wrap that
    text_cleaning.unwrap_lines() may rejoin.
    """
    header_bottom = page.header.bbox[3] if page.header else 0.0
    emitted: set[int] = set()
    parts: list[str] = []
    prev: Block | None = None

    for block in page.blocks:
        # 1) Header removal by position: anything centred above the header box's
        #    bottom edge is header (title, "Issued By", empty logo cells).
        if block.center_y < header_bottom:
            continue

        # 2) Blocks inside a detected table are replaced by the table itself,
        #    rendered once, where its first block would have been.
        table_idx = next((i for i, t in enumerate(page.tables) if t.contains(block)), None)
        if table_idx is not None:
            if table_idx not in emitted:
                emitted.add(table_idx)
                parts += ["\n\n", table_to_markdown(page.tables[table_idx])]
                prev = None
            continue

        # 3) Choose how this block attaches to the previous one.
        if prev is None:
            sep = "\n\n" if parts else ""
        elif _same_row(prev.y0, prev.x1, block.y0, block.x0):
            # Separate blocks side by side are always separate columns; the gap
            # threshold only matters for justified-text pieces inside one block.
            sep = _column_separator(prev.lines[-1].text, gap=float("inf"))
        elif block.y0 - prev.y1 < SOFT_WRAP_GAP:
            sep = "\n"
        else:
            sep = "\n\n"
        parts += [sep, block.text]
        prev = block

    for i, table in enumerate(page.tables):  # tables with no text block inside (unlikely)
        if i not in emitted:
            parts += ["\n\n", table_to_markdown(table)]
    return "".join(parts)


def table_to_markdown(table: Table) -> str:
    rows = [[" ".join(cell.split()) for cell in row] for row in table.rows]
    rows = [r for r in rows if any(r)]  # drop empty template rows
    if not rows:
        return ""
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * len(rows[0])]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)
