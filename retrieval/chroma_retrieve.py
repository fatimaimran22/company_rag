"""Retrieval through the Chroma vector database (compare with retrieval/retrieve.py, the brute force).

    question
      -> all-MiniLM-L6-v2              (the same model that embedded the chunks)
      -> query vector (384 numbers)    (we embed it; Chroma has no embedding model attached)
      -> collection.query(query_embeddings=[vector], n_results=top_k)
      -> Chroma searches its vector index (HNSW, cosine space) and returns the top_k

What Chroma returns is a DISTANCE, not a similarity. For this collection's
"cosine" space:   cosine distance = 1 - cosine similarity
so 0 = same direction, and LOWER is better. We also show the cosine similarity
derived from it (1 - distance) so it can be compared with the brute-force scores.

Note on speed: with 39 vectors an index has nothing to save; brute force may be
just as fast or faster. The index pays off with many more vectors.

Run from the project root (after storage.ingest_chroma):
    .venv/bin/python -m retrieval.chroma_retrieve "What is the monthly allowance for a used MacBook Pro M3?"
    .venv/bin/python -m retrieval.chroma_retrieve "night support allowance" --top-k 3
    .venv/bin/python -m retrieval.chroma_retrieve            # interactive
"""

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

import config
from embedding.embeddings import embed_texts, load_model
from storage.chroma_store import StorageError, open_client, open_collection, record_from_chroma

from .retrieve import RetrievalError, check_top_k


@dataclass
class ChromaResult:
    rank: int
    distance: float  # Chroma's cosine distance: 1 - cosine similarity, lower = closer
    record: dict  # chunk_id, doc, section, page_start/end, text, metadata, ... (rebuilt from Chroma)

    @property
    def chunk_id(self) -> str:
        return self.record["chunk_id"]

    @property
    def cosine_similarity(self) -> float:
        """Derived from the distance, for comparison with brute force: 1 - distance."""
        return 1.0 - self.distance


@dataclass
class ChromaRetrieval:
    query: str
    query_vector: np.ndarray
    top_k: int
    stored_records: int  # how many vectors the collection holds
    results: list[ChromaResult]
    timings: dict[str, float] = field(default_factory=dict)  # seconds


def retrieve(query: str, top_k: int = config.TOP_K, model=None, collection=None) -> ChromaRetrieval:
    """Return the top_k chunks Chroma finds closest to `query`.

    Pass an already loaded `model` and opened `collection` when asking several
    questions, so they are not set up again for every question.
    """
    if not isinstance(query, str) or not query.strip():
        raise RetrievalError("Please provide a question (the query is empty).")
    check_top_k(top_k)
    model = model if model is not None else load_model(config.EMBEDDING_MODEL)
    collection = collection if collection is not None else open_collection(open_client())

    stored = collection.count()
    if stored == 0:
        raise RetrievalError("The Chroma collection is empty. Run: .venv/bin/python -m storage.ingest_chroma")

    # 1. Embed the question ourselves, with the same model as the stored chunks.
    t0 = time.perf_counter()
    query_vector = embed_texts(model, [query])[0]
    if query_vector.shape != (config.EMBEDDING_DIM,):
        raise RetrievalError(f"query vector has shape {query_vector.shape}, expected ({config.EMBEDDING_DIM},)")
    t1 = time.perf_counter()

    # 2. Ask Chroma for the nearest stored vectors.
    found = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=min(top_k, stored),
        include=["documents", "metadatas", "distances"],
    )
    t2 = time.perf_counter()

    # 3. Chroma answers per query vector; we sent one, so take element [0] of each list.
    results = [
        ChromaResult(rank, float(distance), record_from_chroma(chunk_id, document, metadata))
        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(found["ids"][0], found["documents"][0], found["metadatas"][0], found["distances"][0]), start=1
        )
    ]
    return ChromaRetrieval(
        query=query,
        query_vector=query_vector,
        top_k=top_k,
        stored_records=stored,
        results=results,
        timings={"query_embedding": t1 - t0, "chroma_search": t2 - t1, "total": t2 - t0},
    )


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def print_retrieval(r: ChromaRetrieval) -> None:
    v = r.query_vector
    print(f"\nQuery:\n{r.query}\n")
    print(f"Method: Chroma  (collection {config.CHROMA_COLLECTION!r}, {config.CHROMA_SPACE} distance: lower = closer)\n")
    print(f"Query embedding dimensions: {len(v)}")
    print(f"Query embedding: [{', '.join(f'{x:+.4f}' for x in v[:6])}, ...]\n")
    print(f"Stored records:         {r.stored_records}")
    print(f"Results requested:      {r.top_k}")
    print(f"Query embedding time:   {r.timings['query_embedding']:.4f} s")
    print(f"Chroma search time:     {r.timings['chroma_search']:.4f} s")
    print(f"Total retrieval time:   {r.timings['total']:.4f} s")

    print(f"\nTop {len(r.results)} results:")
    for res in r.results:
        rec = res.record
        pages = f"p{rec['page_start']}" if rec["page_start"] == rec["page_end"] else f"p{rec['page_start']}-{rec['page_end']}"
        meta = rec["metadata"]
        print("\n" + "-" * 90)
        print(f"Rank {res.rank}")
        print(f"Chunk ID:   {res.chunk_id}")
        print(f"Distance:   {res.distance:.4f}   (cosine distance; cosine similarity = 1 - distance = {res.cosine_similarity:.4f})")
        print(f"Section:    {rec['section']}  ({pages})")
        print(f"Metadata:   version={meta.get('version')}  effective_date={meta.get('effective_date')}  "
              f"has_table={meta.get('has_table')}  source={meta.get('source_file')}")
        print(f"Text:\n{rec['text']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval through the Chroma vector database.")
    parser.add_argument("question", nargs="*", help="the question (omit for interactive mode)")
    parser.add_argument("--top-k", type=int, default=config.TOP_K, help=f"number of results (default {config.TOP_K})")
    args = parser.parse_args()

    try:
        check_top_k(args.top_k)  # fail fast, before the slow model load
        t0 = time.perf_counter()
        collection = open_collection(open_client())
        t1 = time.perf_counter()
        model = load_model(config.EMBEDDING_MODEL)
        t2 = time.perf_counter()
        print(f"Setup (one-off, not part of retrieval): opened Chroma at "
              f"{config.CHROMA_DIR.relative_to(config.PROJECT_ROOT)}/ in {t1 - t0:.3f} s; "
              f"model {config.EMBEDDING_MODEL} in {t2 - t1:.2f} s")

        question = " ".join(args.question)
        if question:
            print_retrieval(retrieve(question, args.top_k, model, collection))
            return
        while True:  # interactive mode
            question = input("\nQuestion (empty line to quit): ").strip()
            if not question:
                break
            print_retrieval(retrieve(question, args.top_k, model, collection))
    except (RetrievalError, StorageError) as e:
        raise SystemExit(f"Error: {e}")
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
