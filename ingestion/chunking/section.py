"""Strategy 3: section-based chunking, following the document's own headings.

Walk the section tree (see outline.py) from the top:

* If a whole section, including its sub-sections, fits in max_chars, it
  becomes ONE chunk. "4. Allowance Entitlement Structure" with its table
  stays together this way.
* Otherwise go one level deeper and treat each sub-section the same way.
  A parent's own text that is only a heading or a short intro
  ("3.0 PROCEDURE:") is attached to its first sub-section instead of
  becoming a tiny chunk of its own.
* A section with no sub-sections that is still too long is split with the
  recursive splitter (tables kept whole).

Every chunk is labelled with its full breadcrumb, e.g.
"Test Project > 3.0 PROCEDURE > 3.2 Test Projects".
"""

from .document import DocStream, Unit
from .outline import Outline, Section
from .recursive import split_prose_and_tables

MIN_OWN_CHARS = 200  # a parent's own text shorter than this rides along with its first child


def chunk(stream: DocStream, outline: Outline, max_chars: int, table_max_chars: int):
    """Yield (unit, section breadcrumb) pairs for one document."""
    if stream.cover:
        yield Unit.slice(stream, stream.cover.start, stream.cover.end), outline.section_at(stream.cover.start)
    yield from _emit(stream, outline, outline.root, outline.root.start, max_chars, table_max_chars)


def _emit(stream: DocStream, outline: Outline, node: Section, start: int, max_chars: int, table_max_chars: int):
    label = outline.breadcrumb(node)
    whole = stream.text[start : node.end_full]

    if len(whole.strip()) <= max_chars:
        units = split_prose_and_tables(stream, start, node.end_full, len(whole) + 1, table_max_chars)
        for unit in units:  # normally exactly one unit
            yield unit, label
        return

    if not node.children:
        for unit in split_prose_and_tables(stream, start, node.end_full, max_chars, table_max_chars):
            yield unit, label
        return

    carry = None  # where the first child should start (to absorb the parent's short intro)
    if len(stream.text[start : node.end].strip()) < MIN_OWN_CHARS:
        carry = start
    else:
        for unit in split_prose_and_tables(stream, start, node.end, max_chars, table_max_chars):
            yield unit, label

    for i, child in enumerate(node.children):
        child_start = carry if (i == 0 and carry is not None) else child.start
        yield from _emit(stream, outline, child, child_start, max_chars, table_max_chars)
