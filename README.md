# Company Policy RAG

A retrieval-augmented generation (RAG) pipeline over internal company policy PDFs,
built one stage at a time so each step can be inspected on its own.

The repository is organised by pipeline stage: each folder is one stage, each
stage reads the previous stage's output from `data/processed/` and writes its own.

```text
PDF files (data/raw/)
        ↓
  extraction/   PDF -> page records (PyMuPDF), SOP header removed, cover page -> metadata
        ↓
  cleaning/     bullets, zero-width spaces, whitespace, wrapped lines
        ↓
  chunking/     pages -> chunks (fixed / recursive / section-based; section is primary)
        ↓
  embedding/    chunks -> 384-d vectors (sentence-transformers/all-MiniLM-L6-v2)
        ↓
  storage/      vector store (Chroma)                         - not built yet
        ↓
  retrieval/    question -> most similar chunks               - not built yet
        ↓
  evaluation/   measure retrieval / answer quality            - not built yet
        ↓
  rag/          retrieved chunks -> cited answer              - not built yet
```

## Project structure

```text
company_rag/
├── config.py            all shared settings: paths, per-PDF settings, chunk sizes, model
├── requirements.txt
├── data/
│   ├── raw/             source PDFs (git-ignored)
│   └── processed/       everything the pipeline generates (git-ignored)
├── extraction/          pdf_reader.py, cover_page.py, extract.py, inspect.py
├── cleaning/            text_cleaning.py, inspect.py
├── chunking/            document.py, outline.py, fixed.py, recursive.py, section.py,
│                        context.py, chunk.py, check.py, compare.py, inspect.py
├── embedding/           embeddings.py, embed.py, inspect.py
├── storage/             (next)
├── retrieval/           (next)
├── evaluation/          (later)
└── rag/                 (later)
```

Each stage folder's `__init__.py` describes its files, inputs and outputs.
Library modules hold the logic; `extract.py`, `chunk.py`, `embed.py`, ... are the
commands; `inspect.py` in each stage lets you look at that stage's output.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu  # CPU-only build
.venv/bin/pip install -r requirements.txt
```

Put the policy PDFs in `data/raw/` and register each one in `config.DOCUMENTS`.

## Running the pipeline

Run every command **from the project root** with `python -m`, so the stage
packages and `config.py` are importable.

| Stage | Command | Writes |
|---|---|---|
| Extraction + cleaning | `.venv/bin/python -m extraction.extract` | `raw_pages.jsonl`, `documents.json`, `pages.jsonl` |
| Chunking | `.venv/bin/python -m chunking.chunk` | `chunks_fixed.jsonl`, `chunks_recursive.jsonl`, `chunks_section.jsonl` |
| Embedding | `.venv/bin/python -m embedding.embed` | `embeddings.jsonl` |

Cleaning has no separate command: `extraction.extract` applies
`cleaning.text_cleaning.clean_text()` to each page, because the cleaner's input
(header removed, tables as Markdown) is built from page geometry that only
exists during extraction.

### Inspecting each stage

```bash
.venv/bin/python -m extraction.inspect --doc byom --page 2           # raw PyMuPDF text
.venv/bin/python -m extraction.inspect --doc test --meta             # + cover-page metadata
.venv/bin/python -m cleaning.inspect --doc byom --page 3 --compare   # raw vs cleaned

.venv/bin/python -m chunking.inspect --strategy section --doc byom --list
.venv/bin/python -m chunking.inspect --id section-byom-allowance-005 --embedding --meta
.venv/bin/python -m chunking.check                                   # is the primary set ready to embed?
.venv/bin/python -m chunking.compare                                 # fixed vs recursive vs section

.venv/bin/python -m embedding.inspect --id section-byom-allowance-005
.venv/bin/python -m embedding.inspect --demo                         # cosine similarity: related vs unrelated
```

`--doc` and `--section` match any part of the name, case-insensitively.

## Generated data (`data/processed/`)

| File | Stage | One record per | Key fields |
|---|---|---|---|
| `raw_pages.jsonl` | extraction | PDF page | `doc`, `page`, `text` (untouched PyMuPDF output) |
| `documents.json` | extraction | document | cover metadata: version, effective date, approvals, revision history |
| `pages.jsonl` | extraction + cleaning | kept PDF page | `doc`, `page` (1-based), `text`, `metadata` |
| `chunks_<strategy>.jsonl` | chunking | chunk | `chunk_id`, `section` breadcrumb, `page_start`/`page_end`, `text`, `embedding_text`, `metadata` |
| `embeddings.jsonl` | embedding | primary chunk | chunk fields + `embedding` (384 floats), `embedding_model`, `token_info` |

## Design notes

- **Page numbers** come from the PDF's page order (the PDFs print none) and are
  carried through every stage, so a chunk can cite `p3-4` when it crosses a page break.
- **Tables** are kept whole (or split by rows with the header repeated) by every
  chunking strategy.
- **Section-based chunking** is the primary strategy: in the comparison
  (`chunking.compare`) it was the only one that never mixed two top-level sections.
  The fixed and recursive outputs are kept for demonstrations.
- **`embedding_text`** = `<document (title)> > <section breadcrumb>` + the chunk text,
  so every chunk still says which policy it belongs to when embedded on its own.
- **Model input limit:** all-MiniLM-L6-v2 reads 256 tokens; 3 of the 39 section
  chunks are longer and their tail is not represented in the vector
  (`embedding.embed` lists exactly what is cut).
