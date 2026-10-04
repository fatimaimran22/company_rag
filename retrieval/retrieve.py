"""Brute-force retrieval: compare a question against EVERY stored chunk embedding.

    question
      -> all-MiniLM-L6-v2              (the same model that embedded the chunks)
      -> query vector (384 numbers)    (embedded once, even when both methods run)
      -> compare with chunk 1, chunk 2, ..., chunk N   (all of them, one by one), using
           cosine similarity   (higher = more similar) -> sort descending, or
           Euclidean distance  (lower  = closer)       -> sort ascending
      -> keep the top_k

No vector database and no index: the chunk vectors are read from
data/processed/embeddings.jsonl (written by embedding.embed) and only the
question is embedded here. The cost grows linearly with the number of chunks,
which is fine for 39 and is exactly what a vector index later avoids.

Run from the project root:
    .venv/bin/python -m retrieval.retrieve "What is the monthly allowance for a used MacBook Pro M3?"
    .venv/bin/python -m retrieval.retrieve "night support allowance" --top-k 3 --all
    .venv/bin/python -m retrieval.retrieve "night support allowance" --method euclidean
    .venv/bin/python -m retrieval.retrieve "night support allowance" --method both
    .venv/bin/python -m retrieval.retrieve            # interactive: choose a method, then type questions
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

import config
from embedding.embeddings import cosine_similarity, embed_texts, euclidean_distance, load_model


class RetrievalError(Exception):
    """A problem with the stored embeddings or with the request (message is shown to the user)."""


# ---------------------------------------------------------------------------
# Step 1: load the stored chunk embeddings (never re-embedded here)
# ---------------------------------------------------------------------------
def load_stored_embeddings(path: Path = config.EMBEDDINGS_FILE, expected_dim: int = config.EMBEDDING_DIM) -> list[dict]:
    """Read embeddings.jsonl and check every vector. Each record gets a numpy `vector` added."""
    if not path.exists():
        raise RetrievalError(f"{path} not found. Run: .venv/bin/python -m embedding.embed")

    records = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise RetrievalError(f"{path.name} line {line_no}: not valid JSON ({e})") from None
            record["vector"] = _checked_vector(record, expected_dim, f"{path.name} line {line_no}")
            records.append(record)

    if not records:
        raise RetrievalError(f"{path.name} contains no embeddings. Run: .venv/bin/python -m embedding.embed")
    return records


def _checked_vector(record: dict, expected_dim: int, where: str) -> np.ndarray:
    values = record.get("embedding")
    if not isinstance(values, list) or not all(isinstance(v, (int, float)) for v in values):
        raise RetrievalError(f"{where}: 'embedding' must be a list of numbers")
    if len(values) != expected_dim:
        raise RetrievalError(f"{where}: embedding has {len(values)} values, expected {expected_dim}")
    if not all(math.isfinite(v) for v in values) or not any(values):
        raise RetrievalError(f"{where}: embedding contains NaN/inf or is all zeros")
    if record.get("embedding_model") not in (None, config.EMBEDDING_MODEL):
        raise RetrievalError(
            f"{where}: embedded with {record['embedding_model']}, but questions are embedded with "
            f"{config.EMBEDDING_MODEL}; vectors from different models cannot be compared"
        )
    return np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# Steps 2-6: embed the question, compare with every chunk, sort, keep top_k
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Method:
    name: str  # shown in the output
    score_label: str  # "Similarity" or "Distance"
    compare: Callable[[np.ndarray, np.ndarray], float]
    higher_is_better: bool  # decides the sort direction


METHODS = {
    "cosine": Method("Cosine Similarity", "Similarity", cosine_similarity, higher_is_better=True),
    "euclidean": Method("Euclidean Distance", "Distance", euclidean_distance, higher_is_better=False),
}


@dataclass
class Result:
    rank: int
    score: float  # cosine similarity or Euclidean distance between the question and this chunk
    record: dict  # the stored chunk: chunk_id, doc, section, page_start/end, text, metadata, ...

    @property
    def chunk_id(self) -> str:
        return self.record["chunk_id"]


@dataclass
class Retrieval:
    query: str
    query_vector: np.ndarray
    method: str  # key of METHODS
    results: list[Result]  # top_k, best first
    all_scores: list[tuple[str, float]]  # every chunk_id with its score, best first
    timings: dict[str, float] = field(default_factory=dict)  # seconds

    @property
    def searched(self) -> int:
        return len(self.all_scores)


def check_top_k(top_k) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise RetrievalError(f"top_k must be a whole number >= 1, got {top_k!r}")


def check_method(method: str) -> Method:
    if method not in METHODS:
        raise RetrievalError(f"method must be one of {list(METHODS)}, got {method!r}")
    return METHODS[method]


def embed_query(query: str, model, records: list[dict]) -> tuple[np.ndarray, float]:
    """Steps 2-3: only the question is embedded (one text -> one 384-number vector)."""
    if not isinstance(query, str) or not query.strip():
        raise RetrievalError("Please provide a question (the query is empty).")
    t0 = time.perf_counter()
    query_vector = embed_texts(model, [query])[0]
    if query_vector.shape != records[0]["vector"].shape:
        raise RetrievalError(f"query vector has shape {query_vector.shape}, stored vectors {records[0]['vector'].shape}")
    return query_vector, time.perf_counter() - t0


def search(query: str, query_vector: np.ndarray, records: list[dict], top_k: int, method: str,
           embedding_seconds: float = 0.0) -> Retrieval:
    """Steps 4-6 for an already embedded question."""
    check_top_k(top_k)
    m = check_method(method)

    # Step 4: brute force. One comparison per stored chunk, no shortcuts.
    t1 = time.perf_counter()
    scored = []
    for record in records:
        score = m.compare(query_vector, record["vector"])
        scored.append((score, record))
    t2 = time.perf_counter()

    # Step 5: sort all of them, best first:
    #   cosine similarity  -> highest first (descending)
    #   Euclidean distance -> lowest first  (ascending)
    scored.sort(key=lambda pair: pair[0], reverse=m.higher_is_better)
    t3 = time.perf_counter()

    # Step 6: keep the first top_k (fewer if there are fewer chunks).
    results = [Result(rank, score, record) for rank, (score, record) in enumerate(scored[:top_k], start=1)]

    return Retrieval(
        query=query,
        query_vector=query_vector,
        method=method,
        results=results,
        all_scores=[(record["chunk_id"], score) for score, record in scored],
        timings={
            "query_embedding": embedding_seconds,
            "search": t2 - t1,
            "sort_and_top_k": t3 - t2,
            "total": embedding_seconds + (t3 - t1),
        },
    )


def retrieve(query: str, top_k: int = config.TOP_K, model=None, records: list[dict] | None = None,
             method: str = "cosine") -> Retrieval:
    """Return the top_k stored chunks closest to `query` with one method ("cosine" or "euclidean").

    Pass an already loaded `model` and `records` when asking several questions,
    so they are not loaded again for every question.
    """
    if not isinstance(query, str) or not query.strip():
        raise RetrievalError("Please provide a question (the query is empty).")
    check_top_k(top_k)
    check_method(method)
    model = model if model is not None else load_model(config.EMBEDDING_MODEL)
    records = records if records is not None else load_stored_embeddings()

    query_vector, seconds = embed_query(query, model, records)
    return search(query, query_vector, records, top_k, method, seconds)


def retrieve_all_methods(query: str, top_k: int = config.TOP_K, model=None,
                         records: list[dict] | None = None) -> list[Retrieval]:
    """Embed the question ONCE, then search with every method using that same vector."""
    if not isinstance(query, str) or not query.strip():
        raise RetrievalError("Please provide a question (the query is empty).")
    check_top_k(top_k)
    model = model if model is not None else load_model(config.EMBEDDING_MODEL)
    records = records if records is not None else load_stored_embeddings()

    query_vector, seconds = embed_query(query, model, records)
    return [search(query, query_vector, records, top_k, method, seconds) for method in METHODS]


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def print_retrieval(r: Retrieval, show_all: bool) -> None:
    m = METHODS[r.method]
    v = r.query_vector
    direction = "higher = more similar, sorted descending" if m.higher_is_better else "lower = closer, sorted ascending"
    print(f"\nQuery:\n{r.query}\n")
    print(f"Method: {m.name}  ({m.score_label.lower()}: {direction})\n")
    print(f"Query embedding dimensions: {len(v)}")
    print(f"Query embedding: [{', '.join(f'{x:+.4f}' for x in v[:6])}, ...]\n")
    print(f"Embeddings searched: {r.searched}  (1 query x {r.searched} stored chunk vectors)")
    search_label = f"{m.score_label} search time:"
    print(f"Query embedding time:   {r.timings['query_embedding']:.4f} s")
    print(f"{search_label:<24}{r.timings['search']:.4f} s   ({r.searched} {m.name.lower()} calculations)")
    print(f"Sort + top-k time:      {r.timings['sort_and_top_k']:.4f} s")
    print(f"Total retrieval time:   {r.timings['total']:.4f} s")

    if show_all:
        print(f"\nAll {r.searched} {m.score_label.lower()} values, best first:")
        for i, (chunk_id, score) in enumerate(r.all_scores, start=1):
            marker = "  <- top-k" if i <= len(r.results) else ""
            print(f"  {i:>3}. {score:.4f}  {chunk_id}{marker}")

    print(f"\nTop {len(r.results)} results:")
    for res in r.results:
        rec = res.record
        pages = f"p{rec['page_start']}" if rec["page_start"] == rec["page_end"] else f"p{rec['page_start']}-{rec['page_end']}"
        print("\n" + "-" * 90)
        print(f"Rank {res.rank}")
        print(f"Chunk ID:   {res.chunk_id}")
        print(f"{m.score_label + ':':<12}{res.score:.4f}")
        print(f"Section:    {rec['section']}  ({pages})")
        meta = rec["metadata"]
        print(f"Metadata:   version={meta.get('version')}  effective_date={meta.get('effective_date')}  "
              f"has_table={meta.get('has_table')}  source={meta.get('source_file')}")
        print(f"Text:\n{rec['text']}")


def choose_method_interactively() -> str:
    print("\nSelect retrieval method:\n\n1. Cosine similarity\n2. Euclidean distance\n3. Both (same query embedding)")
    choice = input("\nMethod [1]: ").strip() or "1"
    return {"1": "cosine", "2": "euclidean", "3": "both"}.get(choice) or choose_method_interactively()


def run(question: str, method: str, top_k: int, model, records: list[dict], show_all: bool) -> None:
    if method == "both":
        retrievals = retrieve_all_methods(question, top_k, model, records)
    else:
        retrievals = [retrieve(question, top_k, model, records, method)]
    for r in retrievals:
        print("\n" + "=" * 90 + f"\n{METHODS[r.method].name.upper()}\n" + "=" * 90)
        print_retrieval(r, show_all)


def main() -> None:
    parser = argparse.ArgumentParser(description="Brute-force retrieval over the stored chunk embeddings.")
    parser.add_argument("question", nargs="*", help="the question (omit for interactive mode)")
    parser.add_argument("--method", choices=[*METHODS, "both"],
                        help="cosine (default), euclidean, or both; asked interactively if omitted in interactive mode")
    parser.add_argument("--top-k", type=int, default=config.TOP_K, help=f"number of results (default {config.TOP_K})")
    parser.add_argument("--all", action="store_true", help="also list the score of every stored chunk")
    args = parser.parse_args()

    try:
        check_top_k(args.top_k)  # fail fast, before the slow model load
        t0 = time.perf_counter()
        records = load_stored_embeddings()
        t1 = time.perf_counter()
        model = load_model(config.EMBEDDING_MODEL)
        t2 = time.perf_counter()
        print(f"Loaded {len(records)} stored embeddings from {config.EMBEDDINGS_FILE.relative_to(config.PROJECT_ROOT)} "
              f"in {t1 - t0:.3f} s; model {config.EMBEDDING_MODEL} in {t2 - t1:.2f} s (one-off setup, not part of retrieval)")

        question = " ".join(args.question)
        if question:
            run(question, args.method or "cosine", args.top_k, model, records, args.all)
            return
        method = args.method or choose_method_interactively()
        while True:  # interactive mode
            question = input("\nQuestion (empty line to quit): ").strip()
            if not question:
                break
            run(question, method, args.top_k, model, records, args.all)
    except RetrievalError as e:
        raise SystemExit(f"Error: {e}")
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
