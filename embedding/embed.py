"""Embedding command: primary chunks -> embedding vectors -> embeddings.jsonl.

Run from the project root:
    .venv/bin/python -m embedding.embed

Reads data/processed/chunks_section.jsonl (read-only), embeds each chunk's
`embedding_text`, and writes data/processed/embeddings.jsonl.
"""

import hashlib
import json

import numpy as np

import config
from .embeddings import embed_texts, load_model, token_info

RULE = "=" * 90


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_vector(vec: np.ndarray, per_line: int = 8) -> str:
    rows = [", ".join(f"{v:+.4f}" for v in vec[i : i + per_line]) for i in range(0, len(vec), per_line)]
    return "[" + ",\n ".join(rows) + "]"


def main() -> None:
    chunks_path = config.PRIMARY_CHUNKS_FILE
    checksum_before = _sha256(chunks_path)
    with chunks_path.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    # --- 1. chunk -> text sent to the model -------------------------------------------
    texts = [c["embedding_text"] for c in chunks]  # NOT c["text"]: we embed the version with context

    # --- 2. text -> vectors -----------------------------------------------------------
    model = load_model(config.EMBEDDING_MODEL)
    matrix = embed_texts(model, texts, config.EMBEDDING_BATCH_SIZE)
    dim = model.get_embedding_dimension()
    if dim != config.EMBEDDING_DIM:
        raise SystemExit(f"{config.EMBEDDING_MODEL} produces {dim}-d vectors but "
                         f"config.EMBEDDING_DIM is {config.EMBEDDING_DIM}; update config.py")
    tokens = [token_info(model, t) for t in texts]

    print(RULE + "\nEMBEDDING RUN\n" + RULE)
    print(f"Input file:          {chunks_path.relative_to(config.PROJECT_ROOT)}  ({config.PRIMARY_STRATEGY} chunks)")
    print(f"Chunks:              {len(chunks)}")
    print(f"Model:               {config.EMBEDDING_MODEL}")
    print(f"Embedding dimension: {dim}")
    print(f"Matrix shape:        {matrix.shape}   dtype={matrix.dtype}")
    norms = np.linalg.norm(matrix, axis=1)
    print(f"Vector lengths:      min={norms.min():.6f} max={norms.max():.6f}  (normalized -> cosine = dot product)")

    # --- 3. what the model could NOT read ---------------------------------------------
    print(f"\nModel reads at most {model.max_seq_length} tokens per text.")
    truncated = [(c, t) for c, t in zip(chunks, tokens) if t.truncated]
    print(f"Truncated chunks:    {len(truncated)} of {len(chunks)}")
    for c, t in truncated:
        unseen = c["embedding_text"][t.embedded_chars :].strip()
        print(f"  - {c['chunk_id']}: {t.token_count} tokens; last {len(unseen)} chars NOT in the vector:")
        for line in unseen.split("\n"):
            print(f"      | {line[:100]}")

    # --- 4. show vectors --------------------------------------------------------------
    example_idx = next(i for i, c in enumerate(chunks) if c["chunk_id"] == "section-byom-allowance-005")
    print("\n" + RULE + f"\nONE COMPLETE VECTOR: {chunks[example_idx]['chunk_id']}  ({dim} numbers)\n" + RULE)
    print(format_vector(matrix[example_idx]))

    print("\n" + RULE + "\nFIRST 6 VALUES OF SEVERAL OTHER VECTORS\n" + RULE)
    for i in [0, 8, 18, 24, 30, 36]:
        values = ", ".join(f"{v:+.4f}" for v in matrix[i][:6])
        print(f"{chunks[i]['chunk_id']:<36} [{values}, ...]")

    # --- 5. store ---------------------------------------------------------------------
    with config.EMBEDDINGS_FILE.open("w", encoding="utf-8") as f:
        for c, vec, t in zip(chunks, matrix, tokens):
            record = {
                "chunk_id": c["chunk_id"],
                "doc": c["doc"],
                "chunking_strategy": c["chunking_strategy"],
                "section": c["section"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "text": c["text"],
                "embedding_text": c["embedding_text"],
                "embedding_model": config.EMBEDDING_MODEL,
                "embedding_dim": dim,
                "embedding": [float(v) for v in vec],
                "token_info": {
                    "token_count": t.token_count,
                    "max_seq_length": t.max_seq_length,
                    "truncated": t.truncated,
                    "embedded_chars": t.embedded_chars,
                },
                "metadata": c["metadata"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --- 6. verify --------------------------------------------------------------------
    with config.EMBEDDINGS_FILE.open(encoding="utf-8") as f:
        reloaded = np.array([json.loads(line)["embedding"] for line in f], dtype=np.float32)
    print("\n" + RULE + "\nCHECKS\n" + RULE)
    print(f"Wrote {len(chunks)} records -> {config.EMBEDDINGS_FILE.relative_to(config.PROJECT_ROOT)}")
    print(f"Reloaded vectors identical to computed: {np.array_equal(reloaded, matrix)}")
    print(f"{chunks_path.name} unchanged: {_sha256(chunks_path) == checksum_before}")


if __name__ == "__main__":
    main()
