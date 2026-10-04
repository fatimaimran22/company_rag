"""Phase 2 entry point: PDFs in data/raw/ -> cleaned page records in data/processed/.

Run from the project root:
    .venv/bin/python -m ingestion.extract

Outputs (one record per PDF page, 1-based page numbers):
    data/processed/pages.jsonl      cleaned text + metadata  {"doc", "page", "text", "metadata"}
    data/processed/raw_pages.jsonl  untouched PyMuPDF text, for before/after comparison
    data/processed/documents.json   document-level metadata (cover page, skipped pages, ...)
"""

import json
from pathlib import Path

from .config import (
    DOCUMENTS_FILE,
    PAGES_FILE,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    RAW_PAGES_FILE,
    config_for,
)
from .cover_page import parse_cover, parse_header, render_cover_text
from .pdf_reader import body_text, read_pdf
from .text_cleaning import clean_text

# Document-level fields copied onto every page record (small, useful for citing/filtering later).
_PAGE_LEVEL_FIELDS = ["title", "document_id", "version", "effective_date", "issued_by", "issued_to"]


def process_pdf(path: Path) -> tuple[dict, list[dict], list[dict]]:
    cfg = config_for(path)
    pages = read_pdf(path)

    doc_meta: dict = {"title": cfg.name}
    header = next((p.header for p in pages if p.header), None)
    if header:
        doc_meta.update(parse_header(header))
    if cfg.cover_page:
        doc_meta.update(parse_cover(pages[cfg.cover_page - 1]))

    page_meta_base = {
        "source_file": path.name,
        "total_pages": len(pages),
        **{k: doc_meta[k] for k in _PAGE_LEVEL_FIELDS if doc_meta.get(k)},
    }

    records, raw_records = [], []
    for page in pages:
        raw_records.append({"doc": cfg.name, "page": page.number, "text": page.raw_text})
        if page.number in cfg.skip_pages:
            continue

        if page.number == cfg.cover_page:
            page_type, text = "cover", render_cover_text(doc_meta)
        else:
            page_type, text = "content", clean_text(body_text(page))

        records.append(
            {
                "doc": cfg.name,
                "page": page.number,
                "text": text,
                "metadata": {
                    **page_meta_base,
                    "page_type": page_type,
                    "has_table": page_type == "content" and bool(page.tables),
                    "char_count": len(text),
                },
            }
        )

    doc_record = {
        "doc": cfg.name,
        "source_file": path.name,
        "template": cfg.template,
        "total_pages": len(pages),
        "extracted_pages": [r["page"] for r in records],
        "skipped_pages": sorted(cfg.skip_pages),
        "metadata": doc_meta,
    }
    return doc_record, records, raw_records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {RAW_DIR}")

    documents, all_pages, all_raw = [], [], []
    for path in pdfs:
        doc_record, records, raw_records = process_pdf(path)
        documents.append(doc_record)
        all_pages += records
        all_raw += raw_records
        skipped = f", skipped {doc_record['skipped_pages']}" if doc_record["skipped_pages"] else ""
        print(
            f"{doc_record['doc']:<24} {doc_record['total_pages']} pages -> "
            f"{len(records)} records ({sum(len(r['text']) for r in records):,} chars){skipped}"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(PAGES_FILE, all_pages)
    _write_jsonl(RAW_PAGES_FILE, all_raw)
    DOCUMENTS_FILE.write_text(json.dumps(documents, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"\nWrote {len(all_pages)} page records to {PROCESSED_DIR.relative_to(PROJECT_ROOT)}/ "
        f"({PAGES_FILE.name}, {RAW_PAGES_FILE.name}, {DOCUMENTS_FILE.name})"
    )


if __name__ == "__main__":
    main()
