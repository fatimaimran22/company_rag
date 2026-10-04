"""Strategy 2: recursive (separator-based) chunking.

How it works, for a piece of text that is too long:

1. Try the coarsest separator first (a blank line = paragraph break) and split on it.
2. Any piece that is still too long is split again with the NEXT separator
   (line break, then sentence end, then space).
3. If no separator is left, cut at chunk_size characters (last resort).
4. Finally, glue neighbouring small pieces back together while they fit,
   so we do not end up with lots of tiny chunks.

The result: chunks break at the most natural boundary that keeps them under
chunk_size. Tables are kept whole, exactly as in the fixed strategy.
"""

from .document import DocStream, Unit, pack, regions, table_units, trim
from .outline import Outline

# (name, separator) from coarsest to finest.
SEPARATORS = [
    ("paragraph", "\n\n"),
    ("line", "\n"),
    ("sentence", ". "),
    ("word", " "),
]


def _split_on(stream: DocStream, start: int, end: int, sep: str) -> list[tuple[int, int]]:
    """Split [start, end) on `sep`, keeping the separator at the end of the left piece."""
    pieces, piece_start = [], start
    i = stream.text.find(sep, start, end)
    while i != -1:
        pieces.append((piece_start, i + len(sep)))
        piece_start = i + len(sep)
        i = stream.text.find(sep, piece_start, end)
    if piece_start < end:
        pieces.append((piece_start, end))
    return pieces


def recursive_split(stream: DocStream, start: int, end: int, chunk_size: int, level: int = 0) -> list[Unit]:
    if end - start <= chunk_size:
        return [Unit.slice(stream, start, end)]

    if level == len(SEPARATORS):  # nothing natural left to split on: hard cut
        return [Unit.slice(stream, i, min(i + chunk_size, end)) for i in range(start, end, chunk_size)]

    _, sep = SEPARATORS[level]
    pieces = _split_on(stream, start, end, sep)
    if len(pieces) == 1:  # separator not present: try the next, finer one
        return recursive_split(stream, start, end, chunk_size, level + 1)

    units: list[Unit] = []
    for s, e in pieces:
        if e - s > chunk_size:
            units += recursive_split(stream, s, e, chunk_size, level + 1)
        else:
            units.append(Unit.slice(stream, s, e))
    return pack(stream, units, chunk_size)


def split_prose_and_tables(stream: DocStream, start: int, end: int, chunk_size: int, table_max_chars: int) -> list[Unit]:
    """Recursive split for prose, whole (or row-split) tables, then pack neighbours together."""
    units: list[Unit] = []
    for region in regions(stream, start, end):
        if region.is_table:
            units += table_units(stream, region, table_max_chars)
        else:
            units += recursive_split(stream, region.start, region.end, chunk_size)
    packed = pack(stream, units, chunk_size)
    out = []
    for u in packed:
        if u.verbatim:
            u = Unit.slice(stream, *trim(stream, u.start, u.end), is_table=u.is_table)
        if u.text.strip():
            out.append(u)
    return out


def chunk(stream: DocStream, outline: Outline, chunk_size: int, table_max_chars: int):
    """Yield (unit, section breadcrumb) pairs for one document."""
    for unit in split_prose_and_tables(stream, 0, len(stream.text), chunk_size, table_max_chars):
        yield unit, outline.section_at(unit.start)
