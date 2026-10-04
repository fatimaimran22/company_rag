"""Stage 2 - Cleaning: extracted page text -> clean page text.

text_cleaning.py - pure string functions: bullets, zero-width spaces, whitespace, line unwrapping
inspect.py       - command: view cleaned pages, or raw vs cleaned (--compare)

Cleaning has no command of its own: extraction.extract calls clean_text() on each
page, because its input (header removed, tables as Markdown) is built from page
geometry that is not saved anywhere. Output: data/processed/pages.jsonl
"""
