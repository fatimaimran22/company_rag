"""Stage 6 - Retrieval: question -> the most similar stored chunks.

retrieve.py - brute force: embed the question once, compare it with EVERY stored
              chunk vector using cosine similarity (higher = better, sorted descending)
              or Euclidean distance (lower = better, sorted ascending), return the top_k. No index, no vector
              database (that comes with storage/ and is compared against this later).
              Also the command: python -m retrieval.retrieve "your question"
chroma_retrieve.py - the same question through Chroma: embed the question, then
              collection.query() with that vector. Returns cosine DISTANCE (1 - similarity).
              Command: python -m retrieval.chroma_retrieve "your question"

Input: data/processed/embeddings.jsonl (retrieve.py) or data/chroma/ (chroma_retrieve.py);
       read-only, only the question is embedded
"""
