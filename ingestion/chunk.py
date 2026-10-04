"""Phase 3 entry point: pages.jsonl -> one chunks_<strategy>.jsonl per strategy.

Run from the project root:
    .venv/bin/python -m ingestion.chunk
    .venv/bin/python -m ingestion.chunk --chunk-size 800 --overlap 100 --strategies fixed recursive

Reads data/processed/pages.jsonl (never modifies it) and writes
data/processed/chunks_fixed.jsonl, chunks_recursive.jsonl, chunks_section.jsonl.
"""

import argparse
import json
import re

from . import config
from .chunking import fixed, recursive, section
from .chunking.context import embedding_text, table_info
from .chunking.document import DocStream, Unit, load_documents
from .chunking.outline import build_outline


# Document-level metadata carried on every chunk.
DOC_FIELDS = ["source_file", "title", "document_id", "version", "effective_date"]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def chunk_document(stream: DocStream, strategy: str, args) -> list[tuple[Unit, str]]:
    outline = build_outline(stream)
    if strategy == "fixed":
        return list(fixed.chunk(stream, outline, args.chunk_size, args.overlap, args.table_max_chars))
    if strategy == "recursive":
        return list(recursive.chunk(stream, outline, args.chunk_size, args.table_max_chars))
    if strategy == "section":
        return list(section.chunk(stream, outline, args.section_max_chars, args.table_max_chars))
    raise ValueError(f"unknown strategy {strategy!r}")


def to_records(stream: DocStream, strategy: str, pieces: list[tuple[Unit, str]], params: dict) -> list[dict]:
    records = []
    for i, (unit, section_label) in enumerate(pieces, start=1):
        text = unit.text.strip()
        pages = stream.pages_for(unit.start, unit.end)
        tables = table_info(text, row_split=unit.is_table and not unit.verbatim)
        title = stream.metadata.get("title")
        records.append(
            {
                "chunk_id": f"{strategy}-{_slug(stream.doc)}-{i:03d}",
                "doc": stream.doc,
                "chunking_strategy": strategy,
                "section": section_label,
                "page_start": pages[0],
                "page_end": pages[-1],
                "text": text,  # clean content, for display and citations
                "embedding_text": embedding_text(stream.doc, title, section_label, text),  # what gets embedded
                "metadata": {
                    # Same keys for every chunk; None when a document does not have the field.
                    **{key: stream.metadata.get(key) for key in DOC_FIELDS},
                    "pages": pages,
                    "char_count": len(text),
                    "has_table": bool(tables),
                    "tables": tables,
                    "char_start": unit.start,  # offsets into the document's joined text
                    "char_end": unit.end,
                    "params": params,
                },
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk pages.jsonl with several strategies.")
    parser.add_argument("--strategies", nargs="+", choices=config.CHUNK_STRATEGIES, default=config.CHUNK_STRATEGIES)
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE, help="fixed + recursive")
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP, help="fixed")
    parser.add_argument("--section-max-chars", type=int, default=config.SECTION_MAX_CHARS, help="section")
    parser.add_argument("--table-max-chars", type=int, default=config.TABLE_MAX_CHARS, help="all strategies")
    args = parser.parse_args()

    if not config.PAGES_FILE.exists():
        raise SystemExit("pages.jsonl not found. Run Phase 2 first: .venv/bin/python -m ingestion.extract")
    streams = load_documents(config.PAGES_FILE)

    params_by_strategy = {
        "fixed": {"chunk_size": args.chunk_size, "overlap": args.overlap, "table_max_chars": args.table_max_chars},
        "recursive": {"chunk_size": args.chunk_size, "table_max_chars": args.table_max_chars},
        "section": {"max_chars": args.section_max_chars, "table_max_chars": args.table_max_chars},
    }
    for strategy in args.strategies:
        params = params_by_strategy[strategy]
        records = []
        for stream in streams:
            records += to_records(stream, strategy, chunk_document(stream, strategy, args), params)
        out = config.chunks_file(strategy)
        with out.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{strategy:<10} {len(records):>3} chunks -> {out.relative_to(config.PROJECT_ROOT)}   {params}")

    print("\nInspect:  .venv/bin/python -m ingestion.inspect_chunks --strategy section --doc byom")
    print("Compare:  .venv/bin/python -m ingestion.compare_chunks")


if __name__ == "__main__":
    main()
