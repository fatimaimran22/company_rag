"""Load the existing chunk embeddings into the persistent Chroma collection.

Run from the project root (after embedding.embed):
    .venv/bin/python -m storage.ingest_chroma            # add chunks not yet in Chroma
    .venv/bin/python -m storage.ingest_chroma --reset    # delete the collection and rebuild it

Safe to run repeatedly: chunk_ids already in the collection are skipped, not duplicated.
No embeddings are computed here; the vectors come from data/processed/embeddings.jsonl.
"""

import argparse

import config
from retrieval.retrieve import RetrievalError, load_stored_embeddings

from .chroma_store import StorageError, ingest, open_client, open_collection, reset_collection

RULE = "=" * 60


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the stored embeddings into Chroma.")
    parser.add_argument("--reset", action="store_true", help="delete and rebuild the collection first")
    args = parser.parse_args()

    try:
        records = load_stored_embeddings()  # validates: 384 numbers each, same embedding model
        client = open_client()
        collection = reset_collection(client) if args.reset else open_collection(client)
        report = ingest(collection, records)
    except (RetrievalError, StorageError) as e:
        raise SystemExit(f"Error: {e}")

    dims = {len(v) for v in collection.get(include=["embeddings"])["embeddings"]}
    print(RULE + "\nCHROMA INGESTION\n" + RULE)
    print(f"\nSource:\n{config.EMBEDDINGS_FILE.relative_to(config.PROJECT_ROOT)}")
    print(f"\nCollection:\n{collection.name}  (distance: {config.CHROMA_SPACE}, embedding function: none)")
    if args.reset:
        print("(collection was reset before ingesting)")
    print(f"\nEmbeddings loaded: {report.loaded}")
    print(f"Embedding dimensions: {len(records[0]['embedding'])}")
    print(f"\nInserted: {report.inserted}")
    print(f"Skipped/already existing: {report.skipped}")
    print(f"Records in collection now: {report.count_after}  (stored vector dimensions: {sorted(dims)})")
    print(f"\nChroma database:\n{config.CHROMA_DIR.relative_to(config.PROJECT_ROOT)}/")
    print("\n" + RULE)


if __name__ == "__main__":
    main()
