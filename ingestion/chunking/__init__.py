"""Phase 3: three chunking strategies over the cleaned pages, for comparison.

document.py  - joins pages into one string per document, maps offsets back to pages,
               finds tables, packs small pieces together
outline.py   - finds headings and builds the section tree / breadcrumbs
fixed.py     - strategy 1: fixed-size sliding window with overlap
recursive.py - strategy 2: split on paragraphs -> lines -> sentences -> words
section.py   - strategy 3: follow the heading hierarchy
"""
