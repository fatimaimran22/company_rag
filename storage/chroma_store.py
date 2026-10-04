"""Put the existing chunk embeddings into a persistent Chroma collection, and read them back.

Chroma stores four things per chunk:
    ids        -> chunk_id                          ("section-byom-allowance-005")
    embeddings -> the 384-number vector from embeddings.jsonl (NOT recomputed)
    documents  -> the chunk text
    metadatas  -> everything else about the chunk (section, pages, version, ...)

Two Chroma details this module handles explicitly:

* No hidden embedding model. If a collection is opened without
  `embedding_function=None`, Chroma attaches its own default model. We always
  pass None, so every vector Chroma holds or searches with comes from our model.
* Flat metadata only. Chroma accepts str/int/float/bool (and lists of those) as
  metadata values, but not None or dicts. So `tables`, `params` and `token_info`
  are stored as JSON text, and None values are left out; record_from_chroma()
  turns them back into the original structure.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

import config

# Fields that sit at the top level of an embeddings.jsonl record (besides chunk_id, text, embedding).
TOP_LEVEL_FIELDS = [
    "doc", "chunking_strategy", "section", "page_start", "page_end",
    "embedding_text", "embedding_model", "embedding_dim", "token_info",
]
# Fields inside record["metadata"].
CHUNK_METADATA_FIELDS = [
    "source_file", "title", "document_id", "version", "effective_date", "pages",
    "char_count", "has_table", "tables", "char_start", "char_end", "params",
]
JSON_FIELDS = {"token_info", "tables", "params"}  # dicts / lists of dicts: stored as JSON text


class StorageError(Exception):
    """A problem with the Chroma database (message is shown to the user)."""


def open_client(path: Path = config.CHROMA_DIR) -> chromadb.ClientAPI:
    """A Chroma client that keeps its data on disk in `path` (survives after the program exits)."""
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def open_collection(client: chromadb.ClientAPI, name: str = config.CHROMA_COLLECTION):
    """Open the collection, creating it (cosine distance, no embedding function) if needed."""
    collection = client.get_or_create_collection(
        name,
        embedding_function=None,  # we always supply vectors ourselves
        configuration={"hnsw": {"space": config.CHROMA_SPACE}},
    )
    space = (collection.configuration.get("hnsw") or {}).get("space")
    if space != config.CHROMA_SPACE:
        raise StorageError(
            f"Collection {name!r} uses distance {space!r}, expected {config.CHROMA_SPACE!r}. "
            "Rebuild it: .venv/bin/python -m storage.ingest_chroma --reset"
        )
    return collection


def reset_collection(client: chromadb.ClientAPI, name: str = config.CHROMA_COLLECTION):
    """Delete the collection (if it exists) and create it again, empty."""
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    return open_collection(client, name)


def to_chroma_metadata(record: dict) -> dict:
    """embeddings.jsonl record -> flat Chroma metadata (None dropped, dicts as JSON text)."""
    flat = {"chunk_id": record["chunk_id"]}
    for key in TOP_LEVEL_FIELDS:
        flat[key] = record.get(key)
    for key in CHUNK_METADATA_FIELDS:
        flat[key] = record["metadata"].get(key)
    for key in JSON_FIELDS:
        if flat[key] is not None:
            flat[key] = json.dumps(flat[key], ensure_ascii=False)
    return {k: v for k, v in flat.items() if v is not None}


def record_from_chroma(chunk_id: str, document: str, metadata: dict) -> dict:
    """Chroma ids/documents/metadatas -> the same shape as an embeddings.jsonl record (without `embedding`)."""
    meta = dict(metadata)
    for key in JSON_FIELDS:
        if key in meta:
            meta[key] = json.loads(meta[key])
    record = {"chunk_id": chunk_id, "text": document}
    record.update({key: meta.get(key) for key in TOP_LEVEL_FIELDS})
    record["metadata"] = {key: meta.get(key) for key in CHUNK_METADATA_FIELDS}
    return record


@dataclass
class IngestReport:
    loaded: int
    inserted: int
    skipped: int  # chunk_ids that were already in the collection
    count_after: int  # what Chroma reports after ingestion


def ingest(collection, records: list[dict]) -> IngestReport:
    """Add records whose chunk_id is not in the collection yet; skip the others.

    Chroma silently ignores an add() for an id that already exists, so we check
    first; that way the report says exactly what was inserted.
    """
    ids = [r["chunk_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise StorageError("embeddings.jsonl contains duplicate chunk_ids")

    existing = set(collection.get(ids=ids, include=[])["ids"])
    new = [r for r in records if r["chunk_id"] not in existing]
    if new:
        collection.add(
            ids=[r["chunk_id"] for r in new],
            embeddings=[r["embedding"] for r in new],  # the stored vectors, unchanged
            documents=[r["text"] for r in new],
            metadatas=[to_chroma_metadata(r) for r in new],
        )
    return IngestReport(len(records), len(new), len(existing), collection.count())
