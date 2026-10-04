"""Stage 1 - Extraction: PDF files -> page-based records.

pdf_reader.py - the only code that uses PyMuPDF: text blocks, tables, the SOP header box
cover_page.py - reads the SOP cover page into metadata (title, version, approvals, ...)
extract.py    - command: runs extraction (+ the cleaning stage) and writes data/processed/
inspect.py    - command: view the raw extracted text and document metadata

Input:  data/raw/*.pdf
Output: data/processed/raw_pages.jsonl, documents.json, pages.jsonl
"""
