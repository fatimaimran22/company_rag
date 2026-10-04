"""HNSW / Embedding Space Visualizer for the company_chunks ChromaDB collection.

What this script shows
----------------------
All 39 chunk vectors live in 384-dimensional space.  A computer can store and
search that space efficiently, but humans cannot look at 384 axes at once.  PCA
(Principal Component Analysis) finds the two directions that capture the most
variance and projects every vector onto them, giving us a 2-D scatter plot we
can read.

The nodes are the real vectors pulled directly from Chroma — no re-embedding
happens here.

About the edges
---------------
ChromaDB 1.5.9 uses a **pure-Rust** HNSW engine.  Its on-disk binary format is
not compatible with the standalone Python `hnswlib` library, so the actual HNSW
graph connections (which nodes each node points to) cannot be read back in
Python.

The edges drawn here are therefore **approximate k-nearest-neighbour
connections** computed from the raw vectors with scikit-learn's NearestNeighbors.
This is a mathematically faithful representation of which chunks are close to
each other in the real 384-D space, but the exact graph wiring inside the HNSW
index (entry point, layer assignments, M connections per layer) is not available.
The label says "approx. k-NN" so viewers are never misled.

Run
---
    cd ~/RAG/company_rag
    .venv/bin/python -m visualization.visualize_hnsw
    .venv/bin/python -m visualization.visualize_hnsw --k 4 --out my_graph.html
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from storage.chroma_store import open_client, open_collection

# ── optional heavy imports (error early with a clear message) ──────────────────
try:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    sys.exit("scikit-learn is required.  Run:  .venv/bin/pip install scikit-learn")

try:
    import plotly.graph_objects as go
except ImportError:
    sys.exit("plotly is required.  Run:  .venv/bin/pip install plotly")


# ── colour palette — one colour per source document ───────────────────────────
_PALETTE = [
    "#4C9BE8",  # blue
    "#F28B30",  # orange
    "#53C28B",  # green
    "#E8625A",  # red
    "#A97FC4",  # purple
    "#F0C040",  # yellow
]


def _short_id(chunk_id: str) -> str:
    """Return a display label: last two dash-separated tokens, e.g. 'byom-005'."""
    parts = chunk_id.split("-")
    return "-".join(parts[-2:]) if len(parts) >= 2 else chunk_id


def _doc_label(source_file: str) -> str:
    """Trim '.pdf' for legend entries."""
    return source_file.removesuffix(".pdf") if source_file else "unknown"


def build_figure(vectors: np.ndarray, ids: list[str], metadatas: list[dict],
                 k: int) -> go.Figure:
    """Return a Plotly Figure with nodes + approximate k-NN edges."""

    n = len(vectors)

    # ── 1. PCA: 384-D → 2-D ──────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(vectors)
    var_explained = pca.explained_variance_ratio_.sum() * 100

    # ── 2. Approximate k-NN edges (cosine, i.e. angular, distance) ──────────
    # k+1 because the closest neighbour of a vector is itself
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine")
    nbrs.fit(vectors)
    distances, indices = nbrs.kneighbors(vectors)

    # Build a set of undirected edges {(min_i, max_i)} to avoid duplicates
    edges: set[tuple[int, int]] = set()
    for src, neighbours in enumerate(indices):
        for dst in neighbours[1:]:   # skip self (index 0)
            edges.add((min(src, dst), max(src, dst)))

    # ── 3. Colour nodes by source document ───────────────────────────────────
    source_files = [m.get("source_file", "unknown") for m in metadatas]
    unique_sources = sorted(set(source_files))
    colour_map = {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(unique_sources)}
    node_colours = [colour_map[s] for s in source_files]

    # ── 4. Hover text ─────────────────────────────────────────────────────────
    hover_texts = []
    for i, (cid, meta) in enumerate(zip(ids, metadatas)):
        section = meta.get("section", "—")
        src = meta.get("source_file", "—")
        eff = meta.get("effective_date", "—")
        page = meta.get("page_start", "—")
        hover_texts.append(
            f"<b>{cid}</b><br>"
            f"Source: {src}<br>"
            f"Section: {section}<br>"
            f"Page: {page}  |  Effective: {eff}"
        )

    # ── 5. Edge traces ────────────────────────────────────────────────────────
    edge_x, edge_y = [], []
    for src, dst in edges:
        edge_x += [xy[src, 0], xy[dst, 0], None]
        edge_y += [xy[src, 1], xy[dst, 1], None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="rgba(150,150,150,0.35)", width=1),
        hoverinfo="none",
        name="approx. k-NN edge",
        showlegend=True,
    )

    # ── 6. One node trace per source doc (for a clean legend) ─────────────────
    node_traces = []
    for src in unique_sources:
        mask = [i for i, s in enumerate(source_files) if s == src]
        trace = go.Scatter(
            x=xy[mask, 0], y=xy[mask, 1],
            mode="markers+text",
            marker=dict(
                size=14,
                color=colour_map[src],
                line=dict(color="white", width=1.5),
                opacity=0.9,
            ),
            text=[_short_id(ids[i]) for i in mask],
            textposition="top center",
            textfont=dict(size=9, color="#333333"),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=[hover_texts[i] for i in mask],
            name=_doc_label(src),
        )
        node_traces.append(trace)

    # ── 7. Assemble figure ────────────────────────────────────────────────────
    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        title=dict(
            text=(
                f"<b>ChromaDB collection · {config.CHROMA_COLLECTION}</b>  "
                f"({n} chunks · {config.EMBEDDING_DIM}-D → 2-D PCA)<br>"
                f"<sup>Edges = approximate {k}-nearest-neighbour connections "
                f"(NOT the real HNSW graph — see explanation below)  |  "
                f"PCA variance captured: {var_explained:.1f}%</sup>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=15),
        ),
        xaxis=dict(title="PCA component 1", showgrid=True, zeroline=False),
        yaxis=dict(title="PCA component 2", showgrid=True, zeroline=False),
        legend=dict(
            title="Source document",
            bordercolor="#cccccc", borderwidth=1,
        ),
        hovermode="closest",
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        margin=dict(t=120, b=80, l=60, r=40),
        annotations=[
            dict(
                text=(
                    "<b>How to read this chart</b><br>"
                    "Each dot = one chunk vector (384 numbers compressed to 2 numbers by PCA).<br>"
                    "Dots that are close together represent chunks with similar meaning.<br>"
                    "Lines connect each chunk to its k nearest neighbours in the real 384-D space.<br>"
                    "⚠️ The 2-D view loses information — two close dots may not be close in 384-D."
                ),
                xref="paper", yref="paper",
                x=0.01, y=-0.18,
                showarrow=False,
                align="left",
                font=dict(size=10, color="#555555"),
                bgcolor="rgba(240,240,240,0.7)",
                bordercolor="#cccccc",
                borderwidth=1,
            )
        ],
    )
    return fig


def print_summary(n: int, var_explained: float, k: int) -> None:
    sep = "─" * 60
    print(sep)
    print("  HNSW / Embedding Space Visualizer")
    print(sep)
    print(f"  Collection:        {config.CHROMA_COLLECTION}")
    print(f"  Vectors loaded:    {n}")
    print(f"  Embedding dim:     {config.EMBEDDING_DIM}")
    print(f"  Distance metric:   {config.CHROMA_SPACE}")
    print(f"  Embedding model:   {config.EMBEDDING_MODEL}")
    print(f"  PCA variance:      {var_explained:.1f}%  (how much 2-D keeps from 384-D)")
    print(f"  k-NN per node:     {k}")
    print()
    print("  Edge type:  *** APPROXIMATE k-nearest-neighbour (NOT real HNSW graph) ***")
    print()
    print("  Why?  ChromaDB 1.5.9 uses a pure-Rust HNSW engine.  Its binary index")
    print("  format cannot be read by the Python hnswlib library, so the actual")
    print("  graph connections (entry point, layer assignments) are inaccessible.")
    print("  The edges here are computed directly from the stored vectors and are")
    print("  mathematically equivalent to what a nearest-neighbour search returns.")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the ChromaDB embedding space as an interactive 2-D plot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--k", type=int, default=3,
        help="Neighbours per node for the edge overlay (default: 3)",
    )
    parser.add_argument(
        "--out", type=str, default="visualization/hnsw_visualization.html",
        help="Output HTML file path (default: visualization/hnsw_visualization.html)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Save the HTML file but do not open it in a browser",
    )
    args = parser.parse_args()

    # ── Load vectors from Chroma ──────────────────────────────────────────────
    print("Loading vectors from Chroma …")
    collection = open_collection(open_client())
    n = collection.count()
    if n == 0:
        sys.exit("The collection is empty.  Run:  .venv/bin/python -m storage.ingest_chroma")

    result = collection.get(include=["embeddings", "metadatas", "documents"])
    ids: list[str] = result["ids"]
    vectors = np.array(result["embeddings"], dtype=np.float32)   # (n, 384)
    metadatas: list[dict] = result["metadatas"]

    # ── PCA summary stat (for the printed report) ─────────────────────────────
    pca_check = PCA(n_components=2, random_state=42)
    pca_check.fit(vectors)
    var_explained = pca_check.explained_variance_ratio_.sum() * 100

    print_summary(n, var_explained, args.k)

    # ── Build and save figure ─────────────────────────────────────────────────
    print("Building figure …")
    fig = build_figure(vectors, ids, metadatas, k=args.k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved → {out_path.resolve()}")

    if not args.no_browser:
        import webbrowser
        webbrowser.open(out_path.resolve().as_uri())
        print("Opened in browser.")


if __name__ == "__main__":
    main()
