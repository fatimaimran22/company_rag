"""Compare the chunking strategies side by side.

Run from the project root (after chunking.chunk):
    .venv/bin/python -m chunking.compare
    .venv/bin/python -m chunking.compare --doc byom --section "9. Process Flow"
    .venv/bin/python -m chunking.compare --stats-only

Prints:
  1. statistics per strategy
  2. a table check: where does the BYOM "MacBook Pro (M3)" row end up?
  3. what fixed-size chunking would do to that table WITHOUT table protection
  4. what row-splitting a table looks like (header repeated)
  5. the SAME section as chunked by each strategy
"""

import argparse
import statistics

import config
from .document import load_documents, regions, table_units
from .fixed import fixed_windows
from .outline import build_outline
from .inspect import load_chunks, page_range

RULE = "=" * 100


def _cut_mid_word(stream_text: str, chunk: dict) -> bool:
    """Does the chunk end in the middle of a word (letters on both sides of the cut)?"""
    end = chunk["metadata"]["char_end"]
    return 0 < end < len(stream_text) and stream_text[end - 1].isalnum() and stream_text[end].isalnum()


def _mixes_top_sections(top_heading_starts: list[int], chunk: dict) -> bool:
    """Does a top-level heading (e.g. "5. Terms & Conditions") start somewhere INSIDE this chunk,
    i.e. does the chunk mix the end of one top-level section with the start of another?"""
    start, end = chunk["metadata"]["char_start"], chunk["metadata"]["char_end"]
    return any(start < h < end for h in top_heading_starts)


def print_stats(all_chunks: dict[str, list[dict]], streams: dict) -> None:
    print(RULE + "\n1. STATISTICS (characters)\n" + RULE)
    top_starts = {doc: [s.start for s in build_outline(st).root.children] for doc, st in streams.items()}
    cols = ["strategy", "chunks", "avg", "median", "min", "max", "cross-page", "with table", "<150 chars",
            "mid-word end", "mixes top sec"]
    print("".join(f"{c:>14}" for c in cols))
    for strategy, chunks in all_chunks.items():
        sizes = [c["metadata"]["char_count"] for c in chunks]
        row = [
            strategy,
            len(chunks),
            round(statistics.mean(sizes)),
            round(statistics.median(sizes)),
            min(sizes),
            max(sizes),
            sum(c["page_start"] != c["page_end"] for c in chunks),
            sum(c["metadata"]["has_table"] for c in chunks),
            sum(s < 150 for s in sizes),
            sum(_cut_mid_word(streams[c["doc"]].text, c) for c in chunks),
            sum(_mixes_top_sections(top_starts[c["doc"]], c) for c in chunks),
        ]
        print("".join(f"{v:>14}" for v in row))
    for strategy, chunks in all_chunks.items():
        print(f"  {strategy:<10} params: {chunks[0]['metadata']['params']}")


def print_table_check(all_chunks: dict[str, list[dict]]) -> None:
    print("\n" + RULE + '\n2. TABLE CHECK: "What is the monthly allowance for a used MacBook Pro M3?"\n' + RULE)
    print("Does the chunk holding the M3 row also hold the column headers (so '7,000' is understandable)?\n")
    for strategy, chunks in all_chunks.items():
        for c in chunks:
            if "MacBook Pro (M3)" in c["text"]:
                has_header = "Monthly Allowance" in c["text"]
                rows = sum(line.lower().startswith("| macbook") for line in c["text"].split("\n"))
                print(
                    f"  {strategy:<10} {c['chunk_id']:<30} {page_range(c):<6} {c['metadata']['char_count']:>5} chars  "
                    f"header in chunk: {'yes' if has_header else 'NO'}   model rows in chunk: {rows}  "
                    f"| has intro text: {'yes' if 'monthly allowance rates apply' in c['text'] else 'no'}"
                )


def print_unprotected_demo(stream) -> None:
    print("\n" + RULE + "\n3. DEMO: fixed-size WITHOUT table protection (not saved, for illustration only)\n" + RULE)
    table = next(u for u in regions(stream, 0, len(stream.text)) if u.is_table)
    windows = fixed_windows(stream, 0, len(stream.text), config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    for w in windows:
        if w.start < table.end and table.start < w.end:
            print(f"--- window chars {w.start}-{w.end} ---")
            print(w.text)
    print("\n-> rows are cut mid-line and the header ends up in a different window than some rows.")


def print_row_split_demo(stream, max_chars: int = 260) -> None:
    print("\n" + RULE + f"\n4. DEMO: row-splitting a table that is too big (table_max_chars={max_chars})\n" + RULE)
    table = next(u for u in regions(stream, 0, len(stream.text)) if u.is_table)
    for i, piece in enumerate(table_units(stream, table, max_chars), start=1):
        print(f"--- piece {i} ({len(piece.text)} chars) ---")
        print(piece.text)


def print_same_content(all_chunks: dict[str, list[dict]], doc: str, section: str) -> None:
    anchor = [c for c in all_chunks["section"] if doc.lower() in c["doc"].lower() and section.lower() in c["section"].lower()]
    if not anchor:
        print(f"\nNo section chunk matches doc={doc!r} section={section!r}")
        return
    start = min(c["metadata"]["char_start"] for c in anchor)
    end = max(c["metadata"]["char_end"] for c in anchor)
    print("\n" + RULE + f"\n5. SAME CONTENT, THREE STRATEGIES: {anchor[0]['section']}  (chars {start}-{end})\n" + RULE)
    for strategy, chunks in all_chunks.items():
        overlapping = [
            c for c in chunks
            if c["doc"] == anchor[0]["doc"] and c["metadata"]["char_start"] < end and start < c["metadata"]["char_end"]
        ]
        print(f"\n{'#' * 30} {strategy.upper()}: {len(overlapping)} chunk(s) touch this section {'#' * 30}")
        for c in overlapping:
            spill = []
            if c["metadata"]["char_start"] < start:
                spill.append("starts BEFORE the section")
            if c["metadata"]["char_end"] > end:
                spill.append("runs PAST the section")
            note = f"  <- {', '.join(spill)}" if spill else ""
            print(f"\n┌─ {c['chunk_id']} | {page_range(c)} | {c['metadata']['char_count']} chars | {c['section']}{note}")
            for line in c["text"].split("\n"):
                print("│ " + line)
            print("└" + "─" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chunking strategies.")
    parser.add_argument("--doc", default="byom", help="document for the side-by-side example")
    parser.add_argument("--section", default="4. Allowance", help="section for the side-by-side example")
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    all_chunks = {s: load_chunks(s) for s in config.CHUNK_STRATEGIES}
    streams = {s.doc: s for s in load_documents(config.PAGES_FILE)}
    print_stats(all_chunks, streams)
    if args.stats_only:
        return
    print_table_check(all_chunks)
    byom = streams["BYOM Allowance"]
    print_unprotected_demo(byom)
    print_row_split_demo(byom)
    print_same_content(all_chunks, args.doc, args.section)


if __name__ == "__main__":
    main()
