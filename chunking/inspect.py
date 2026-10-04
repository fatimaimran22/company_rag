"""Look at generated chunks from the terminal.

Examples (run from the project root):
    .venv/bin/python -m chunking.inspect --strategy fixed --doc byom
    .venv/bin/python -m chunking.inspect --strategy section --doc test --section 3.2
    .venv/bin/python -m chunking.inspect --strategy recursive --doc working --page 3
    .venv/bin/python -m chunking.inspect --strategy section --doc byom --list   # one line per chunk
    .venv/bin/python -m chunking.inspect --id section-byom-allowance-005 --embedding --meta

--doc and --section match case-insensitively on any part of the name.
"""

import argparse
import json

import config


def load_chunks(strategy: str) -> list[dict]:
    path = config.chunks_file(strategy)
    if not path.exists():
        raise SystemExit(f"{path.name} not found. Run: .venv/bin/python -m chunking.chunk")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def page_range(chunk: dict) -> str:
    if chunk["page_start"] == chunk["page_end"]:
        return f"p{chunk['page_start']}"
    return f"p{chunk['page_start']}-{chunk['page_end']}"


def header_line(chunk: dict) -> str:
    table = "  [table]" if chunk["metadata"]["has_table"] else ""
    return (
        f"{chunk['chunk_id']}  |  {chunk['chunking_strategy']}  |  {page_range(chunk)}  |  "
        f"{chunk['metadata']['char_count']} chars{table}\n  section: {chunk['section']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect chunks produced by chunking.chunk.")
    parser.add_argument("--strategy", choices=config.CHUNK_STRATEGIES + ["all"], default="all")
    parser.add_argument("--doc", help="document name (substring, case-insensitive)")
    parser.add_argument("--section", help="section breadcrumb (substring, case-insensitive)")
    parser.add_argument("--page", type=int, help="only chunks that include this PDF page")
    parser.add_argument("--id", help="a single chunk_id")
    parser.add_argument("--list", action="store_true", help="one line per chunk, no text")
    parser.add_argument("--embedding", action="store_true", help="show embedding_text (what the model sees)")
    parser.add_argument("--meta", action="store_true", help="also print the chunk's metadata")
    args = parser.parse_args()

    strategies = config.CHUNK_STRATEGIES if args.strategy == "all" else [args.strategy]
    if args.id:
        strategies = [s for s in config.CHUNK_STRATEGIES if args.id.startswith(s + "-")] or strategies

    shown = 0
    for strategy in strategies:
        for c in load_chunks(strategy):
            if args.id and c["chunk_id"] != args.id:
                continue
            if args.doc and args.doc.lower() not in c["doc"].lower():
                continue
            if args.section and args.section.lower() not in c["section"].lower():
                continue
            if args.page and not c["page_start"] <= args.page <= c["page_end"]:
                continue
            shown += 1
            if args.list:
                print(
                    f"{c['chunk_id']:<36} {page_range(c):<6} {c['metadata']['char_count']:>5}  "
                    f"{'T' if c['metadata']['has_table'] else ' '}  {c['section']}"
                )
            else:
                print("\n" + "=" * 100)
                print(header_line(c))
                print("-" * 100)
                print(c["embedding_text"] if args.embedding else c["text"])
                if args.meta:
                    print("-" * 100)
                    print(json.dumps(c["metadata"], ensure_ascii=False))
    print(f"\n({shown} chunks)")


if __name__ == "__main__":
    main()
