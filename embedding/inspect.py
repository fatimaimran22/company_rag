"""Look at stored embeddings and compare two of them, from the terminal.

Examples (run from the project root, after embedding.embed):
    .venv/bin/python -m embedding.inspect                    # list all vectors
    .venv/bin/python -m embedding.inspect --id section-byom-allowance-005
    .venv/bin/python -m embedding.inspect --id section-byom-allowance-005 --full
    .venv/bin/python -m embedding.inspect --compare section-byom-allowance-005 section-byom-allowance-012
    .venv/bin/python -m embedding.inspect --demo             # a related and an unrelated pair

Reads data/processed/embeddings.jsonl only; no model is loaded.
"""

import argparse
import json

import numpy as np

import config
from .embeddings import cosine_similarity

# Pairs used by --demo: same topic (allowance amount / how the allowance is set),
# and different topics (laptop allowance table / working on weekends).
RELATED_PAIR = ("section-byom-allowance-005", "section-byom-allowance-012")
UNRELATED_PAIR = ("section-byom-allowance-005", "section-working-hours-policy-007")


def load_embeddings() -> dict[str, dict]:
    if not config.EMBEDDINGS_FILE.exists():
        raise SystemExit("embeddings.jsonl not found. Run: .venv/bin/python -m embedding.embed")
    with config.EMBEDDINGS_FILE.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {r["chunk_id"]: r for r in records}


def _get(records: dict, chunk_id: str) -> dict:
    if chunk_id not in records:
        raise SystemExit(f"Unknown chunk_id {chunk_id!r}. Run without arguments to list them.")
    return records[chunk_id]


def _values(vec, n: int) -> str:
    shown = ", ".join(f"{v:+.4f}" for v in vec[:n])
    return f"[{shown}, ...]  ({len(vec)} values)" if n < len(vec) else f"[{shown}]"


def show_list(records: dict) -> None:
    print(f"{'chunk_id':<36}{'dim':>5}{'tokens':>8}  trunc  section")
    for r in records.values():
        t = r["token_info"]
        print(f"{r['chunk_id']:<36}{r['embedding_dim']:>5}{t['token_count']:>8}  {'YES' if t['truncated'] else '   '}    {r['section']}")
    first = next(iter(records.values()))
    print(f"\n{len(records)} vectors, model {first['embedding_model']}")


def show_one(r: dict, full: bool) -> None:
    t = r["token_info"]
    print(f"chunk_id:   {r['chunk_id']}")
    print(f"section:    {r['section']}   (pages {r['page_start']}-{r['page_end']})")
    print(f"model:      {r['embedding_model']}")
    print(f"\n--- original text ({len(r['text'])} chars) ---\n{r['text']}")
    print(f"\n--- embedding_text: what the model received ({len(r['embedding_text'])} chars) ---\n{r['embedding_text']}")
    print(f"\n--- tokens: {t['token_count']} (model reads {t['max_seq_length']}) ---")
    if t["truncated"]:
        print(f"TRUNCATED: the vector ignores this tail:\n{r['embedding_text'][t['embedded_chars']:].strip()}")
    vec = np.array(r["embedding"], dtype=np.float32)
    print(f"\n--- vector: dimension {len(vec)}, length (norm) {np.linalg.norm(vec):.6f} ---")
    if full:
        for i in range(0, len(vec), 8):
            print(f"  [{i:>3}-{i + 7:>3}] " + ", ".join(f"{v:+.4f}" for v in vec[i : i + 8]))
    else:
        print("first 10 values:", _values(vec, 10))


def show_comparison(a: dict, b: dict, label: str = "") -> float:
    va, vb = np.array(a["embedding"], dtype=np.float32), np.array(b["embedding"], dtype=np.float32)
    print("=" * 90 + (f"\n{label}\n" if label else "\n") + "=" * 90)
    for name, r, v in [("A", a, va), ("B", b, vb)]:
        first_line = r["text"].split("\n")[0][:70]
        print(f"chunk {name}: {r['chunk_id']}\n   section: {r['section']}\n   text:    {first_line!r}...")
        print(f"vector {name}: {_values(v, 8)}\n")

    products = va * vb
    terms = " + ".join(f"({x:+.4f} x {y:+.4f})" for x, y in zip(va[:3], vb[:3]))
    dot = float(products.sum())
    norm_a, norm_b = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    score = cosine_similarity(va, vb)
    print("cosine_similarity(A, B) = (A . B) / (|A| x |B|)")
    print(f"  A . B  = {terms} + ... ({len(va)} products)")
    print(f"         = {dot:.4f}")
    print(f"  |A| = {norm_a:.4f}   |B| = {norm_b:.4f}   (both 1: the model normalizes its vectors)")
    print(f"  score  = {dot:.4f} / ({norm_a:.4f} x {norm_b:.4f}) = {score:.4f}\n")
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect stored embeddings.")
    parser.add_argument("--id", help="show one chunk's text, embedding_text and vector")
    parser.add_argument("--full", action="store_true", help="with --id: print all vector values")
    parser.add_argument("--compare", nargs=2, metavar=("CHUNK_A", "CHUNK_B"), help="cosine similarity of two chunks")
    parser.add_argument("--demo", action="store_true", help="compare a related and an unrelated pair")
    args = parser.parse_args()

    records = load_embeddings()
    if args.id:
        show_one(_get(records, args.id), args.full)
    elif args.compare:
        show_comparison(_get(records, args.compare[0]), _get(records, args.compare[1]))
    elif args.demo:
        related = show_comparison(*(_get(records, i) for i in RELATED_PAIR), label="RELATED: same topic")
        unrelated = show_comparison(*(_get(records, i) for i in UNRELATED_PAIR), label="UNRELATED: different topics")
        print(f"related {related:.4f}  vs  unrelated {unrelated:.4f}")
    else:
        show_list(records)


if __name__ == "__main__":
    main()
