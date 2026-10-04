"""Look at the cleaned pages, and what cleaning changed, from the terminal.

Examples (run from the project root):
    .venv/bin/python -m cleaning.inspect                                 # list documents
    .venv/bin/python -m cleaning.inspect --doc "BYOM Allowance"          # all cleaned pages
    .venv/bin/python -m cleaning.inspect --doc byom --page 3 --compare   # raw vs cleaned
    .venv/bin/python -m cleaning.inspect --doc test --meta               # include metadata

--doc matches case-insensitively against the short name or the PDF file name.
"""

import argparse

from extraction.inspect import show_pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect cleaned pages.")
    parser.add_argument("--doc", help="document short name or file name (substring, case-insensitive)")
    parser.add_argument("--page", type=int, help="1-based PDF page number")
    parser.add_argument("--compare", action="store_true", help="show raw (invisible chars marked) and cleaned text")
    parser.add_argument("--meta", action="store_true", help="also print document and page metadata")
    args = parser.parse_args()
    show_pages(args.doc, args.page, view="compare" if args.compare else "cleaned", meta=args.meta)


if __name__ == "__main__":
    main()
