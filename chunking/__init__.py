"""Stage 3 - Chunking: cleaned pages -> chunks, with three strategies to compare.

document.py  - joins pages into one string per document, maps offsets back to pages,
               finds tables, packs small pieces together
outline.py   - finds headings and builds the section tree / breadcrumbs
fixed.py     - strategy 1: fixed-size sliding window with overlap
recursive.py - strategy 2: split on paragraphs -> lines -> sentences -> words
section.py   - strategy 3: follow the heading hierarchy (the primary strategy)
context.py   - embedding_text (breadcrumb + text) and table metadata for each chunk
chunk.py     - command: run the strategies, write one chunks_<strategy>.jsonl each
check.py     - command: validate a chunk file before embedding
compare.py   - command: statistics and side-by-side comparison of the strategies
inspect.py   - command: view chunks

Input:  data/processed/pages.jsonl
Output: data/processed/chunks_fixed.jsonl, chunks_recursive.jsonl, chunks_section.jsonl
"""
