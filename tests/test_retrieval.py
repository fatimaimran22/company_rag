"""Tests for brute-force retrieval (retrieval/retrieve.py).

Run from the project root:
    .venv/bin/python -m unittest tests.test_retrieval -v

The math and validation tests need neither the model nor the data. The
retrieval tests use the real model and data/processed/embeddings.jsonl, and
are skipped if that file has not been generated yet.
"""

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import config
from embedding.embeddings import cosine_similarity, euclidean_distance, load_model
from retrieval.retrieve import RetrievalError, load_stored_embeddings, retrieve, retrieve_all_methods

DIM = config.EMBEDDING_DIM


class CosineSimilarityTest(unittest.TestCase):
    """cosine_similarity(A, B) = (A . B) / (|A| x |B|) on tiny vectors with known answers."""

    def test_known_values(self):
        cases = [
            ([1, 0], [0, 1], 0.0),  # perpendicular
            ([1, 2, 3], [2, 4, 6], 1.0),  # same direction, different length
            ([1, 0], [-1, 0], -1.0),  # opposite
            ([1, 1], [1, 0], 1 / math.sqrt(2)),  # 45 degrees
        ]
        for a, b, expected in cases:
            with self.subTest(a=a, b=b):
                self.assertAlmostEqual(cosine_similarity(np.array(a, float), np.array(b, float)), expected, places=6)


class EuclideanDistanceTest(unittest.TestCase):
    """distance(A, B) = sqrt((a1-b1)^2 + (a2-b2)^2 + ...) on tiny vectors with known answers."""

    def test_known_values(self):
        cases = [
            ([0, 0], [3, 4], 5.0),  # the 3-4-5 triangle
            ([1, 2, 3], [1, 2, 3], 0.0),  # identical vectors
            ([1, 2, 3], [4, 6, 3], 5.0),  # differences (-3, -4, 0)
            ([1, 0], [0, 1], math.sqrt(2)),
        ]
        for a, b, expected in cases:
            with self.subTest(a=a, b=b):
                self.assertAlmostEqual(euclidean_distance(np.array(a, float), np.array(b, float)), expected, places=6)

    def test_unit_vectors_relation_to_cosine(self):
        # For length-1 vectors: distance = sqrt(2 - 2 * cosine), so both give the same ranking.
        a = np.array([0.6, 0.8])
        b = np.array([1.0, 0.0])
        self.assertAlmostEqual(euclidean_distance(a, b), math.sqrt(2 - 2 * cosine_similarity(a, b)), places=6)


class InputValidationTest(unittest.TestCase):
    """Bad stored embeddings and bad requests give a clear RetrievalError."""

    def _write(self, lines: list[str]) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        tmp.write("\n".join(lines))
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return Path(tmp.name)

    def _record(self, embedding, **extra) -> str:
        return json.dumps({"chunk_id": "c1", "text": "t", "embedding": embedding, **extra})

    def test_missing_file(self):
        with self.assertRaisesRegex(RetrievalError, "not found"):
            load_stored_embeddings(Path("/nonexistent/embeddings.jsonl"))

    def test_empty_file(self):
        with self.assertRaisesRegex(RetrievalError, "no embeddings"):
            load_stored_embeddings(self._write([""]))

    def test_invalid_json(self):
        with self.assertRaisesRegex(RetrievalError, "not valid JSON"):
            load_stored_embeddings(self._write(["{not json"]))

    def test_malformed_embedding(self):
        for bad in [None, "0.1, 0.2", ["a"] * DIM]:
            with self.subTest(embedding=bad):
                with self.assertRaisesRegex(RetrievalError, "list of numbers"):
                    load_stored_embeddings(self._write([self._record(bad)]))

    def test_wrong_dimension(self):
        with self.assertRaisesRegex(RetrievalError, f"expected {DIM}"):
            load_stored_embeddings(self._write([self._record([0.1] * 10)]))

    def test_zero_vector(self):
        with self.assertRaisesRegex(RetrievalError, "all zeros"):
            load_stored_embeddings(self._write([self._record([0.0] * DIM)]))

    def test_different_model(self):
        with self.assertRaisesRegex(RetrievalError, "different models"):
            load_stored_embeddings(self._write([self._record([0.1] * DIM, embedding_model="other-model")]))

    def test_valid_record_gets_vector(self):
        records = load_stored_embeddings(self._write([self._record([0.1] * DIM)]))
        self.assertEqual(records[0]["vector"].shape, (DIM,))

    def test_missing_query(self):
        for query in ["", "   ", None]:
            with self.subTest(query=query):
                with self.assertRaisesRegex(RetrievalError, "empty"):
                    retrieve(query, top_k=5, model=object(), records=[])

    def test_invalid_method(self):
        with self.assertRaisesRegex(RetrievalError, "method"):
            retrieve("question", top_k=5, model=object(), records=[], method="dot_product")

    def test_invalid_top_k(self):
        for top_k in [0, -1, 2.5, "5", True]:
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(RetrievalError, "top_k"):
                    retrieve("question", top_k=top_k, model=object(), records=[])


class CountingModel:
    """Wraps the real model and records every list of texts sent to encode()."""

    def __init__(self, model):
        self.model = model
        self.encoded: list[list[str]] = []

    def encode(self, texts, **kwargs):
        self.encoded.append(list(texts))
        return self.model.encode(texts, **kwargs)


_MODEL = None


def shared_model():
    """Load the real model once for all test classes (loading takes several seconds)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model(config.EMBEDDING_MODEL)
    return _MODEL


QUERY = "What is the monthly allowance for a used MacBook Pro M3?"


@unittest.skipUnless(config.EMBEDDINGS_FILE.exists(), "run embedding.embed first")
class BruteForceRetrievalTest(unittest.TestCase):
    """Cosine similarity (the default method): unchanged behaviour."""

    QUERY = QUERY

    @classmethod
    def setUpClass(cls):
        cls.file_hash_before = hashlib.sha256(config.EMBEDDINGS_FILE.read_bytes()).hexdigest()
        cls.records = load_stored_embeddings()
        cls.stored_vectors_before = np.stack([r["vector"] for r in cls.records]).copy()
        cls.model = CountingModel(shared_model())
        cls.result = retrieve(cls.QUERY, top_k=5, model=cls.model, records=cls.records)

    def test_query_embedding_has_384_dimensions(self):
        self.assertEqual(self.result.query_vector.shape, (DIM,))

    def test_every_stored_chunk_is_scored(self):
        self.assertEqual(self.result.searched, len(self.records))
        self.assertEqual({cid for cid, _ in self.result.all_scores}, {r["chunk_id"] for r in self.records})

    def test_scores_are_cosine_with_stored_vectors(self):
        by_id = {r["chunk_id"]: r["vector"] for r in self.records}
        for chunk_id, score in self.result.all_scores:
            self.assertAlmostEqual(score, cosine_similarity(self.result.query_vector, by_id[chunk_id]), places=6)

    def test_all_scores_sorted_descending(self):
        scores = [s for _, s in self.result.all_scores]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_returns_at_most_k(self):
        self.assertEqual(len(self.result.results), 5)
        self.assertEqual([r.rank for r in self.result.results], [1, 2, 3, 4, 5])
        everything = retrieve(self.QUERY, top_k=1000, model=self.model, records=self.records)
        self.assertEqual(len(everything.results), len(self.records))  # never more than exist

    def test_first_result_scores_highest(self):
        top = self.result.results
        for later in top[1:]:
            self.assertGreaterEqual(top[0].score, later.score)
        self.assertEqual(top[0].score, self.result.all_scores[0][1])

    def test_only_the_query_is_embedded(self):
        # Every call to the model must be exactly one text: the question. No chunk text is re-embedded.
        self.assertTrue(self.model.encoded)
        for texts in self.model.encoded:
            self.assertEqual(len(texts), 1)
            self.assertEqual(texts[0], self.QUERY)

    def test_stored_embeddings_unchanged(self):
        after = np.stack([r["vector"] for r in self.records])
        self.assertTrue(np.array_equal(after, self.stored_vectors_before))
        self.assertEqual(hashlib.sha256(config.EMBEDDINGS_FILE.read_bytes()).hexdigest(), self.file_hash_before)



@unittest.skipUnless(config.EMBEDDINGS_FILE.exists(), "run embedding.embed first")
class EuclideanRetrievalTest(unittest.TestCase):
    """Euclidean distance: same brute force, lower = closer, sorted ascending."""

    @classmethod
    def setUpClass(cls):
        cls.file_hash_before = hashlib.sha256(config.EMBEDDINGS_FILE.read_bytes()).hexdigest()
        cls.records = load_stored_embeddings()
        cls.stored_vectors_before = np.stack([r["vector"] for r in cls.records]).copy()
        cls.model = CountingModel(shared_model())
        cls.result = retrieve(QUERY, top_k=5, model=cls.model, records=cls.records, method="euclidean")

    def test_method_recorded(self):
        self.assertEqual(self.result.method, "euclidean")

    def test_query_embedding_has_384_dimensions(self):
        self.assertEqual(self.result.query_vector.shape, (DIM,))

    def test_distance_calculated_for_every_stored_chunk(self):
        self.assertEqual(self.result.searched, len(self.records))
        by_id = {r["chunk_id"]: r["vector"] for r in self.records}
        self.assertEqual({cid for cid, _ in self.result.all_scores}, set(by_id))
        for chunk_id, distance in self.result.all_scores:
            self.assertAlmostEqual(distance, euclidean_distance(self.result.query_vector, by_id[chunk_id]), places=6)

    def test_distances_sorted_ascending(self):
        distances = [d for _, d in self.result.all_scores]
        self.assertEqual(distances, sorted(distances))

    def test_top_k_returns_at_most_k(self):
        self.assertEqual(len(self.result.results), 5)
        everything = retrieve(QUERY, top_k=1000, model=self.model, records=self.records, method="euclidean")
        self.assertEqual(len(everything.results), len(self.records))

    def test_first_result_is_closest(self):
        top = self.result.results
        for later in top[1:]:
            self.assertLessEqual(top[0].score, later.score)
        self.assertEqual(top[0].score, self.result.all_scores[0][1])

    def test_only_the_query_is_embedded(self):
        self.assertTrue(self.model.encoded)
        for texts in self.model.encoded:
            self.assertEqual(texts, [QUERY])

    def test_stored_embeddings_unchanged(self):
        after = np.stack([r["vector"] for r in self.records])
        self.assertTrue(np.array_equal(after, self.stored_vectors_before))
        self.assertEqual(hashlib.sha256(config.EMBEDDINGS_FILE.read_bytes()).hexdigest(), self.file_hash_before)


@unittest.skipUnless(config.EMBEDDINGS_FILE.exists(), "run embedding.embed first")
class BothMethodsTest(unittest.TestCase):
    """Running both methods embeds the question once and reuses that vector."""

    @classmethod
    def setUpClass(cls):
        cls.records = load_stored_embeddings()
        cls.model = CountingModel(shared_model())
        cls.cosine, cls.euclidean = retrieve_all_methods(QUERY, top_k=5, model=cls.model, records=cls.records)

    def test_query_embedded_once(self):
        self.assertEqual(self.model.encoded, [[QUERY]])
        self.assertIs(self.cosine.query_vector, self.euclidean.query_vector)

    def test_cosine_part_matches_plain_cosine_retrieval(self):
        alone = retrieve(QUERY, top_k=5, model=shared_model(), records=self.records)
        self.assertEqual([c for c, _ in self.cosine.all_scores], [c for c, _ in alone.all_scores])

    def test_same_ranking_for_normalized_vectors(self):
        # Our vectors have length 1, so distance = sqrt(2 - 2 * cosine): the order is identical.
        self.assertEqual([c for c, _ in self.cosine.all_scores], [c for c, _ in self.euclidean.all_scores])
        cos = dict(self.cosine.all_scores)
        for chunk_id, distance in self.euclidean.all_scores:
            self.assertAlmostEqual(distance, math.sqrt(2 - 2 * cos[chunk_id]), places=5)


if __name__ == "__main__":
    unittest.main()
