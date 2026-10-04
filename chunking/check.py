"""Check that the primary chunk set is ready to embed.

Run from the project root (after chunking.chunk):
    .venv/bin/python -m chunking.check
    .venv/bin/python -m chunking.check --strategy recursive

Errors are things that must be fixed before embedding; notes are things
worth knowing about but not necessarily wrong.
"""

import argparse
import re

import config
from .context import context_line
from .inspect import load_chunks

REQUIRED = ["chunk_id", "doc", "chunking_strategy", "section", "page_start", "page_end", "text", "embedding_text"]
REQUIRED_META = ["source_file", "title", "document_id", "version", "effective_date", "pages", "has_table", "tables"]
SHORT_CONTENT = 150  # chars of content below which a chunk is flagged as "short"


def check(chunks: list[dict]) -> tuple[list[str], list[str]]:
    errors, notes = [], []
    seen, reported_docs = set(), set()
    for c in chunks:
        cid = c.get("chunk_id", "?")
        missing = [k for k in REQUIRED if not c.get(k) and c.get(k) != 0]
        missing += [f"metadata.{k}" for k in REQUIRED_META if k not in c.get("metadata", {})]
        if missing:
            errors.append(f"{cid}: missing {missing}")
            continue
        if cid in seen:
            errors.append(f"{cid}: duplicate chunk_id")
        seen.add(cid)
        if not c["page_start"] <= c["page_end"]:
            errors.append(f"{cid}: page_start > page_end")

        meta = c["metadata"]
        expected_head = context_line(c["doc"], meta["title"], c["section"])
        if not c["embedding_text"].startswith(expected_head + "\n\n" + c["text"][:20]):
            errors.append(f"{cid}: embedding_text does not start with its context line + text")
        if meta["has_table"] != bool(meta["tables"]):
            errors.append(f"{cid}: has_table disagrees with tables")
        for t in meta["tables"]:
            if not all(t["columns"]) or t["rows"] == 0:
                notes.append(f"{cid}: table with empty column names or no rows: {t}")

        if not _has_own_heading(c):
            first_line = c["text"].split("\n")[0]
            notes.append(f"{cid}: no heading in text ({first_line[:40]!r}...); context line supplies it")
        if len(c["text"]) < SHORT_CONTENT:
            notes.append(f"{cid}: short content ({len(c['text'])} chars)")
        missing_doc_fields = [k for k in ["version", "effective_date"] if meta[k] is None]
        if missing_doc_fields and c["doc"] not in reported_docs:
            reported_docs.add(c["doc"])
            notes.append(f"{c['doc']}: document has no {missing_doc_fields} (stored as null on all its chunks)")
    return errors, notes


def _has_own_heading(chunk: dict) -> bool:
    """Does the chunk text itself contain its (deepest) section heading?"""
    leaf = chunk["section"].split(" > ")[-1]
    if leaf == "Document Information":
        return True
    first_token = leaf.split()[0]
    if first_token[0].isdigit() or len(first_token) == 2:  # "3.2.1.2", "4.", "b."
        pattern = re.compile(rf"^{re.escape(first_token.rstrip('.'))}\.?\s")
    else:  # title headings: "Part 2: Weekly...", "Extra Working Days"
        pattern = re.compile(rf"^{re.escape(leaf[:20])}")
    return any(pattern.match(line) for line in chunk["text"].split("\n"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a chunk file before embedding.")
    parser.add_argument("--strategy", choices=config.CHUNK_STRATEGIES, default=config.PRIMARY_STRATEGY)
    args = parser.parse_args()

    chunks = load_chunks(args.strategy)
    errors, notes = check(chunks)
    print(f"{config.chunks_file(args.strategy).name}: {len(chunks)} chunks, {len(errors)} errors, {len(notes)} notes\n")
    for e in errors:
        print("ERROR ", e)
    for n in notes:
        print("note  ", n)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
