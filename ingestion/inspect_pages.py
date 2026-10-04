"""Look at what the extraction produced, page by page, from the terminal.

Examples (run from the project root):
    .venv/bin/python -m ingestion.inspect_pages                          # list documents
    .venv/bin/python -m ingestion.inspect_pages --doc "BYOM Allowance"   # all cleaned pages
    .venv/bin/python -m ingestion.inspect_pages --doc byom --page 2      # one page
    .venv/bin/python -m ingestion.inspect_pages --doc byom --page 2 --compare  # raw vs cleaned
    .venv/bin/python -m ingestion.inspect_pages --doc night --raw        # raw PyMuPDF text only
    .venv/bin/python -m ingestion.inspect_pages --doc test --meta        # include metadata

--doc matches case-insensitively against the short name or the PDF file name.
"""

import argparse
import json

from .config import DOCUMENTS_FILE, PAGES_FILE, RAW_PAGES_FILE


def _load_jsonl(path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _matches(query: str, doc: dict) -> bool:
    q = query.lower()
    return q in doc["doc"].lower() or q in doc["source_file"].lower()


def _show_invisible(text: str) -> str:
    """Make characters you cannot see in a terminal visible, for --compare."""
    return text.replace("\u200b", "⟨ZWSP⟩").replace("\u00a0", "⟨NBSP⟩")


def _rule(title: str) -> str:
    return f"\n{'=' * 8} {title} {'=' * max(4, 70 - len(title))}"


def list_documents(documents: list[dict], pages: list[dict]) -> None:
    print(f"{'doc':<24}{'file':<40}{'pages':>6}{'kept':>6}{'chars':>8}")
    for d in documents:
        chars = sum(len(p["text"]) for p in pages if p["doc"] == d["doc"])
        print(
            f"{d['doc']:<24}{d['source_file']:<40}{d['total_pages']:>6}"
            f"{len(d['extracted_pages']):>6}{chars:>8,}"
        )
    print('\nUse --doc "<name>" to view pages, --help for more options.')


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect extracted PDF pages.")
    parser.add_argument("--doc", help="document short name or file name (substring, case-insensitive)")
    parser.add_argument("--page", type=int, help="1-based PDF page number")
    view = parser.add_mutually_exclusive_group()
    view.add_argument("--raw", action="store_true", help="show raw PyMuPDF text instead of cleaned")
    view.add_argument("--compare", action="store_true", help="show raw (invisible chars marked) and cleaned text")
    parser.add_argument("--meta", action="store_true", help="also print page metadata")
    args = parser.parse_args()

    if not PAGES_FILE.exists():
        raise SystemExit("No processed output yet. Run: .venv/bin/python -m ingestion.extract")

    documents = json.loads(DOCUMENTS_FILE.read_text(encoding="utf-8"))
    pages = _load_jsonl(PAGES_FILE)
    if not args.doc:
        list_documents(documents, pages)
        return

    selected = [d for d in documents if _matches(args.doc, d)]
    if not selected:
        raise SystemExit(f"No document matches {args.doc!r}. Known: {[d['doc'] for d in documents]}")

    raw = {(r["doc"], r["page"]): r["text"] for r in _load_jsonl(RAW_PAGES_FILE)}
    cleaned = {(p["doc"], p["page"]): p for p in pages}

    for d in selected:
        print(_rule(f"{d['doc']}  ({d['source_file']}, {d['total_pages']} pages)"))
        if args.meta:
            print(json.dumps(d["metadata"], indent=2, ensure_ascii=False))

        page_numbers = [args.page] if args.page else range(1, d["total_pages"] + 1)
        for n in page_numbers:
            key = (d["doc"], n)
            if key not in raw:
                print(f"\n(page {n} does not exist)")
                continue
            record = cleaned.get(key)

            if args.raw or args.compare:
                print(_rule(f"page {n} | RAW (PyMuPDF page.get_text())"))
                print(_show_invisible(raw[key]) if args.compare else raw[key])
            if not args.raw:
                if record is None:
                    print(_rule(f"page {n} | SKIPPED (see skip_pages in ingestion/config.py)"))
                    continue
                print(_rule(f"page {n} | CLEANED ({record['metadata']['page_type']})"))
                print(record["text"])
                if args.meta:
                    print("\nmetadata:", json.dumps(record["metadata"], ensure_ascii=False))


if __name__ == "__main__":
    main()
