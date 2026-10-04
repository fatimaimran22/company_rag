"""Knowledge about the company SOP template: the header box and the cover page.

The cover page is a stack of tables (key/value fields, approval boxes with
"Signature" placeholders, and a mostly empty Revision History grid). We turn
it into structured metadata, then render a short, readable text version of
that metadata so the cover page can still be inspected (and later indexed)
without any "Signature" noise.
"""

import re

from .pdf_reader import RawPage, Table

# Cover labels that become top-level document metadata; every other
# "Label: value" pair is an approval/ownership role.
_TOP_LEVEL_FIELDS = {
    "document id": "document_id",
    "effective date": "effective_date",
    "version": "version",
}
_REVISION_KEYS = ["rev_no", "description", "rev_date", "effective_date"]
_PLACEHOLDERS = {"signature"}
_LOOSE_FIELD = re.compile(r"^([A-Z][A-Za-z ]{1,30}):\s*(\S.*)$")  # e.g. "Prepared by: ..."


def _squash(text: str) -> str:
    return " ".join(text.replace("\u200b", " ").split())


def _useful_lines(cell: str) -> list[str]:
    lines = (_squash(line) for line in cell.split("\n"))
    return [line for line in lines if line and line.lower() not in _PLACEHOLDERS]


def _snake(label: str) -> str:
    return re.sub(r"\W+", "_", label.strip().lower()).strip("_")


def parse_header(header: Table) -> dict[str, str]:
    """[['Policy & ...'], ['Title: Test Project', 'Issued to: Human Asset (HA)']] -> fields."""
    fields = {}
    for row in header.rows[1:]:
        for cell in row:
            label, sep, value = _squash(cell).partition(":")
            if sep and value.strip():
                fields[_snake(label)] = value.strip()
    return fields


def _label_value_pairs(table: Table) -> list[tuple[str, str]]:
    """Handles both cover layouts:
    - label and value in neighbouring cells: ['Version:', '# 003']
    - label and value stacked in one cell:   ['Process Owner:\\nVP Human Asset\\nSignature']
    """
    pairs = []
    for row in table.rows:
        cells = [c for c in row if c.strip()]
        i = 0
        while i < len(cells):
            lines = _useful_lines(cells[i])
            if lines and lines[0].endswith(":"):
                value_lines = lines[1:]
                if not value_lines and i + 1 < len(cells):
                    i += 1
                    value_lines = _useful_lines(cells[i])
                if value_lines:
                    pairs.append((lines[0][:-1].strip(), " ".join(value_lines)))
            i += 1
    return pairs


def _revision_history(table: Table) -> list[dict[str, str]]:
    revisions = []
    for row in table.rows[2:]:  # skip the "Revision History" title row and column headings
        values = [_squash(c) for c in row]
        if any(values):
            revisions.append(dict(zip(_REVISION_KEYS, values)))
    return revisions


def parse_cover(page: RawPage) -> dict:
    meta: dict = {"approvals": {}, "revision_history": []}
    for table in page.tables:
        if table.rows and table.rows[0][0].strip().startswith("Revision History"):
            meta["revision_history"] = _revision_history(table)
            continue
        for label, value in _label_value_pairs(table):
            key = _TOP_LEVEL_FIELDS.get(label.lower())
            if key:
                meta[key] = value
            else:
                meta["approvals"][label.title()] = value

    # Fields printed outside any table, e.g. "Prepared by: <name>" on the Night Shift cover.
    for block in page.blocks:
        if any(t.contains(block) for t in page.tables) or (
            page.header and block.center_y < page.header.bbox[3]
        ):
            continue
        for line in block.text.split("\n"):
            match = _LOOSE_FIELD.match(_squash(line))
            if match:
                meta["approvals"][match.group(1).title()] = match.group(2)
    return meta


def render_cover_text(doc_meta: dict) -> str:
    """Readable summary of the cover page, built from metadata (no 'Signature' noise)."""
    lines = [f"Document: {doc_meta.get('title', '')}"]
    for key, label in [
        ("document_id", "Document ID"),
        ("version", "Version"),
        ("effective_date", "Effective Date"),
        ("issued_by", "Issued By"),
        ("issued_to", "Issued To"),
    ]:
        if doc_meta.get(key):
            lines.append(f"{label}: {doc_meta[key]}")

    if doc_meta.get("approvals"):
        lines += ["", "Approvals:"]
        lines += [f"- {role}: {who}" for role, who in doc_meta["approvals"].items()]

    if doc_meta.get("revision_history"):
        lines += ["", "Revision History:"]
        for rev in doc_meta["revision_history"]:
            lines.append(
                f"- {rev.get('rev_no', '')} (revised {rev.get('rev_date', '')}, "
                f"effective {rev.get('effective_date', '')}): {rev.get('description', '')}"
            )
    return "\n".join(lines)
