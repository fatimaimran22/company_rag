"""Shared configuration for every pipeline stage, in one place.

Sections follow the pipeline order: paths -> extraction -> chunking -> embedding.
Each value is defined once here; stage code imports it (`import config`).
Run every command from the project root (`python -m <stage>.<command>`) so
that this module is importable.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths: source data in data/raw/, everything generated in data/processed/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_PAGES_FILE = PROCESSED_DIR / "raw_pages.jsonl"  # extraction: untouched PyMuPDF text, for comparison
DOCUMENTS_FILE = PROCESSED_DIR / "documents.json"  # extraction: document-level metadata
PAGES_FILE = PROCESSED_DIR / "pages.jsonl"  # extraction + cleaning: cleaned, one record per page
EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings.jsonl"  # embedding: vectors + chunk metadata
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"  # storage: persistent Chroma vector database


def chunks_file(strategy: str) -> Path:
    """chunking: data/processed/chunks_<strategy>.jsonl"""
    return PROCESSED_DIR / f"chunks_{strategy}.jsonl"


# ---------------------------------------------------------------------------
# Extraction: how each PDF is treated
# Each PDF is registered explicitly so you can see (and change) how it is
# treated, instead of the code guessing from the file contents.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DocConfig:
    name: str  # short name used in output and in --doc
    template: str  # "sop" = company SOP template (header + cover page), "plain" = no template
    cover_page: int | None = None  # 1-based page holding the SOP cover table
    skip_pages: frozenset[int] = field(default_factory=frozenset)  # 1-based pages to drop entirely


DOCUMENTS: dict[str, DocConfig] = {
    "Bring Your Own Machine Allowance.pdf": DocConfig("BYOM Allowance", "sop", cover_page=1),
    "Night Shift & Night Allowance.pdf": DocConfig("Night Shift Allowance", "sop", cover_page=1),
    "Test Project.pdf": DocConfig("Test Project", "sop", cover_page=1),
    "Working Hours Policy.pdf": DocConfig(
        "Working Hours Policy", "plain", skip_pages=frozenset({1})  # page 1 is only a title page
    ),
}


def config_for(pdf_path: Path) -> DocConfig:
    """Registered config, or a safe default for a PDF nobody has registered yet."""
    return DOCUMENTS.get(pdf_path.name, DocConfig(pdf_path.stem, "plain"))


# ---------------------------------------------------------------------------
# Chunking (all overridable from the command line of chunking.chunk)
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500  # fixed + recursive: target characters per chunk
CHUNK_OVERLAP = 50  # fixed only: characters shared by consecutive windows
SECTION_MAX_CHARS = 1200  # section: a whole section up to this size stays one chunk
TABLE_MAX_CHARS = 1000  # all: a table up to this size is never split; larger ones split by rows

CHUNK_STRATEGIES = ["fixed", "recursive", "section"]
PRIMARY_STRATEGY = "section"  # chosen after the chunking comparison; the others are kept for demos
PRIMARY_CHUNKS_FILE = chunks_file(PRIMARY_STRATEGY)

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # what EMBEDDING_MODEL produces; embedding.embed checks it against the model
EMBEDDING_BATCH_SIZE = 16

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 5  # default number of chunks returned per question

# ---------------------------------------------------------------------------
# Storage (Chroma vector database)
# ---------------------------------------------------------------------------
CHROMA_COLLECTION = "company_chunks"
CHROMA_SPACE = "cosine"  # Chroma's distance for this collection: 1 - cosine similarity
