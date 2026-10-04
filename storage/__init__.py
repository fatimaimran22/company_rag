"""Stage 5 - Storage: put the chunk embeddings in a persistent Chroma vector database.

chroma_store.py  - open the database/collection (cosine distance, no embedding function),
                   convert records to/from Chroma's flat metadata, idempotent ingest
ingest_chroma.py - command: load embeddings.jsonl into Chroma (--reset to rebuild)

Input:  data/processed/embeddings.jsonl (the existing vectors; nothing is re-embedded)
Output: data/chroma/  (collection "company_chunks")
"""
