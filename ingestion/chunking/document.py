"""Shared building blocks for every chunking strategy.

The key idea: each document's pages are joined into ONE string (DocStream.text),
and we remember where each page starts and ends inside it. Chunkers never copy
text around blindly; they pick character ranges ("spans") of that string. A
span can always be mapped back to the PDF pages it came from, which is how a
chunk that crosses a page break ends up with page_start=3, page_end=4.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

PAGE_SEPARATOR = "\n\n"  # pages are joined like paragraphs


@dataclass
class PageSpan:
    page: int  # 1-based PDF page number from Phase 2
    start: int
    end: int
    page_type: str  # "cover" or "content"


@dataclass
class DocStream:
    doc: str
    text: str
    pages: list[PageSpan]
    metadata: dict  # document-level fields copied from the Phase 2 page records

    @property
    def cover(self) -> PageSpan | None:
        return next((p for p in self.pages if p.page_type == "cover"), None)

    @property
    def content_start(self) -> int:
        return next((p.start for p in self.pages if p.page_type != "cover"), len(self.text))

    def pages_for(self, start: int, end: int) -> list[int]:
        """PDF pages that the character range [start, end) touches."""
        hits = [p.page for p in self.pages if p.start < end and start < p.end]
        if hits:
            return hits
        # Range lies entirely inside a page separator: attribute it to the next page.
        return [next((p.page for p in self.pages if p.start >= start), self.pages[-1].page)]


def load_documents(pages_file: Path) -> list[DocStream]:
    """Group Phase 2 page records by document and build one DocStream per document."""
    by_doc: dict[str, list[dict]] = {}
    with pages_file.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                by_doc.setdefault(record["doc"], []).append(record)

    streams = []
    for doc, records in by_doc.items():
        records.sort(key=lambda r: r["page"])
        text, spans = "", []
        for r in records:
            if text:
                text += PAGE_SEPARATOR
            spans.append(PageSpan(r["page"], len(text), len(text) + len(r["text"]), r["metadata"]["page_type"]))
            text += r["text"]
        meta_keys = ["source_file", "title", "document_id", "version", "effective_date"]
        metadata = {k: records[0]["metadata"][k] for k in meta_keys if k in records[0]["metadata"]}
        streams.append(DocStream(doc, text, spans, metadata))
    return streams


# --------------------------------------------------------------------------
# Units: the pieces chunkers produce and combine
# --------------------------------------------------------------------------


@dataclass
class Unit:
    start: int  # span in DocStream.text (used for page numbers)
    end: int
    text: str  # usually text[start:end]; rebuilt text for row-split tables
    is_table: bool = False
    verbatim: bool = field(default=True)  # False when `text` is not a plain slice

    @classmethod
    def slice(cls, stream: DocStream, start: int, end: int, is_table: bool = False) -> "Unit":
        return cls(start, end, stream.text[start:end], is_table)


def trim(stream: DocStream, start: int, end: int) -> tuple[int, int]:
    """Shrink a span so it does not start or end with whitespace."""
    while start < end and stream.text[start].isspace():
        start += 1
    while end > start and stream.text[end - 1].isspace():
        end -= 1
    return start, end


def _lines_with_offsets(text: str, start: int, end: int):
    for m in re.finditer(r"[^\n]*\n?", text[start:end]):
        if m.group():
            yield start + m.start(), start + m.end(), m.group().rstrip("\n")


def regions(stream: DocStream, start: int, end: int) -> list[Unit]:
    """Split [start, end) into alternating prose and table units.

    A table is a run of consecutive lines starting with "|" (the Markdown tables
    written in Phase 2). Tables are returned whole so no strategy cuts through one.
    """
    out: list[Unit] = []
    cur_start, cur_is_table = start, None
    for line_start, _, line in _lines_with_offsets(stream.text, start, end):
        is_table = line.startswith("|")
        if cur_is_table is None:
            cur_is_table = is_table
        elif is_table != cur_is_table:
            out.append(Unit.slice(stream, *trim(stream, cur_start, line_start), is_table=cur_is_table))
            cur_start, cur_is_table = line_start, is_table
    if cur_is_table is not None:
        out.append(Unit.slice(stream, *trim(stream, cur_start, end), is_table=cur_is_table))
    return [u for u in out if u.end > u.start]


def table_units(stream: DocStream, table: Unit, max_chars: int) -> list[Unit]:
    """Keep a table whole if it fits in max_chars; otherwise split it by rows,
    repeating the header (column names + "---" line) at the top of every piece."""
    if len(table.text) <= max_chars:
        return [table]

    lines = list(_lines_with_offsets(stream.text, table.start, table.end))
    header = "\n".join(l for _, _, l in lines[:2])
    pieces: list[Unit] = []
    group: list[tuple[int, int, str]] = []

    def flush():
        if group:
            text = header + "\n" + "\n".join(l for _, _, l in group)
            pieces.append(Unit(group[0][0], group[-1][1], text, is_table=True, verbatim=False))

    for row in lines[2:]:
        candidate = len(header) + sum(len(l) + 1 for _, _, l in group) + len(row[2]) + 1
        if group and candidate > max_chars:
            flush()
            group = []
        group.append(row)
    flush()
    return pieces


def pack(stream: DocStream, units: list[Unit], max_chars: int) -> list[Unit]:
    """Greedily merge neighbouring units while the result stays <= max_chars.

    A unit that is already bigger than max_chars (e.g. a whole table) is kept
    as-is: tables are atomic, prose has already been split small enough.
    """
    merged: list[Unit] = []
    group: list[Unit] = []

    def size(us: list[Unit]) -> int:
        if all(u.verbatim for u in us):
            return us[-1].end - us[0].start
        return sum(len(u.text) for u in us) + 2 * (len(us) - 1)

    def flush():
        if not group:
            return
        if all(u.verbatim for u in group):
            merged.append(Unit.slice(stream, group[0].start, group[-1].end, any(u.is_table for u in group)))
        else:
            text = "\n\n".join(u.text.strip() for u in group)
            merged.append(Unit(group[0].start, group[-1].end, text, any(u.is_table for u in group), False))

    for unit in units:
        if group and size(group + [unit]) > max_chars:
            flush()
            group = []
        group.append(unit)
    flush()
    return merged
