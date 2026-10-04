"""Strategy 1: fixed-size chunking.

Slide a window of `chunk_size` characters over the text, moving forward by
`chunk_size - overlap` each time. It ignores sentences, paragraphs and
headings completely; that is both its simplicity and its weakness.

The one exception: tables are cut out first and kept whole (see
document.regions / table_units), so the window never slices a table row.
"""

from .document import DocStream, Unit, regions, table_units, trim
from .outline import Outline


def fixed_windows(stream: DocStream, start: int, end: int, chunk_size: int, overlap: int) -> list[Unit]:
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    step = chunk_size - overlap
    units = []
    for window_start in range(start, end, step):
        window_end = min(window_start + chunk_size, end)
        s, e = trim(stream, window_start, window_end)
        if e > s:
            units.append(Unit.slice(stream, s, e))
        if window_end == end:
            break
    return units


def chunk(stream: DocStream, outline: Outline, chunk_size: int, overlap: int, table_max_chars: int):
    """Yield (unit, section breadcrumb) pairs for one document."""
    for region in regions(stream, 0, len(stream.text)):
        if region.is_table:
            units = table_units(stream, region, table_max_chars)
        else:
            units = fixed_windows(stream, region.start, region.end, chunk_size, overlap)
        for unit in units:
            yield unit, outline.section_at(unit.start)
