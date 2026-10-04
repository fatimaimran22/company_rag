"""Paths and per-document settings for the ingestion step.

Each PDF is registered explicitly so you can see (and change) how it is
treated, instead of the code guessing from the file contents.
"""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PAGES_FILE = PROCESSED_DIR / "pages.jsonl"  # cleaned, one record per page
RAW_PAGES_FILE = PROCESSED_DIR / "raw_pages.jsonl"  # untouched PyMuPDF text, for comparison
DOCUMENTS_FILE = PROCESSED_DIR / "documents.json"  # document-level metadata


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
