"""Stage 4 - Embedding: chunks -> vectors.

embeddings.py - load the Sentence Transformers model, embed texts, token limits, cosine similarity
embed.py      - command: embed the primary chunks' embedding_text, write embeddings.jsonl
inspect.py    - command: view vectors, compare two chunks with cosine similarity

Input:  data/processed/chunks_section.jsonl (config.PRIMARY_CHUNKS_FILE)
Output: data/processed/embeddings.jsonl
"""
