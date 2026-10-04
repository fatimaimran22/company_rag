"""Tests for Chroma storage (storage/chroma_store.py) and Chroma retrieval (retrieval/chroma_retrieve.py).

Run from the project root:
    .venv/bin/python -m unittest tests.test_chroma -v

Every test uses a temporary Chroma directory, so data/chroma/ is never touched.
Needs data/processed/embeddings.jsonl (skipped otherwise).
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import config
from retrieval.chroma_retrieve import retrieve as chroma_retrieve
from retrieval.retrieve import RetrievalError, load_stored_embeddings
from retrieval.retrieve import retrieve as brute_force_retrieve
from storage.chroma_store import ingest, open_client, open_collection, record_from_chroma, reset_collection
from tests.test_retrieval import QUERY, CountingModel, shared_model

COLLECTION = "test_company_chunks"


class TempChromaTestCase(unittest.TestCase):
    """Gives each test class its own empty Chroma directory and the stored records."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="chroma_test_"))
        cls.records = load_stored_embeddings()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def fresh_collection(self):
        return reset_collection(open_client(self.tmp), COLLECTION)


@unittest.skipUnless(config.EMBEDDINGS_FILE.exists(), "run embedding.embed first")
class ChromaStorageTest(TempChromaTestCase):

    def test_collection_created_and_reopened_from_disk(self):
        collection = self.fresh_collection()
        ingest(collection, self.records)
        reopened = open_collection(open_client(self.tmp), COLLECTION)  # a new client, same directory
        self.assertEqual(reopened.count(), len(self.records))
        self.assertEqual(reopened.configuration["hnsw"]["space"], "cosine")

    def test_no_hidden_embedding_model(self):
        # Without an embedding function, Chroma cannot embed text itself: we must pass vectors.
        collection = self.fresh_collection()
        ingest(collection, self.records)
        with self.assertRaises(ValueError):
            collection.query(query_texts=["monthly allowance"], n_results=1)

    def test_ingest_all_39(self):
        collection = self.fresh_collection()
        report = ingest(collection, self.records)
        self.assertEqual((report.loaded, report.inserted, report.skipped), (39, 39, 0))
        self.assertEqual(collection.count(), 39)

    def test_stored_embeddings_are_the_existing_384d_vectors(self):
        collection = self.fresh_collection()
        ingest(collection, self.records)
        got = collection.get(include=["embeddings"])
        stored = dict(zip(got["ids"], got["embeddings"]))
        for r in self.records:
            vec = np.asarray(stored[r["chunk_id"]])
            self.assertEqual(vec.shape, (config.EMBEDDING_DIM,))
            np.testing.assert_allclose(vec, r["vector"], atol=1e-6)

    def test_ids_text_and_metadata_preserved(self):
        collection = self.fresh_collection()
        ingest(collection, self.records)
        got = collection.get(include=["documents", "metadatas"])
        self.assertEqual(set(got["ids"]), {r["chunk_id"] for r in self.records})
        by_id = {r["chunk_id"]: r for r in self.records}
        for chunk_id, document, metadata in zip(got["ids"], got["documents"], got["metadatas"]):
            original = by_id[chunk_id]
            rebuilt = record_from_chroma(chunk_id, document, metadata)
            self.assertEqual(rebuilt["text"], original["text"])
            self.assertEqual(rebuilt["metadata"], original["metadata"])  # incl. None values, pages, tables
            for key in ["doc", "section", "page_start", "page_end", "embedding_text", "embedding_model", "token_info"]:
                self.assertEqual(rebuilt[key], original[key], key)

    def test_duplicate_ingestion_adds_nothing(self):
        collection = self.fresh_collection()
        ingest(collection, self.records)
        again = ingest(collection, self.records)
        self.assertEqual((again.inserted, again.skipped, again.count_after), (0, 39, 39))

    def test_only_missing_chunks_are_added(self):
        collection = self.fresh_collection()
        ingest(collection, self.records[:-1])
        report = ingest(collection, self.records)
        self.assertEqual((report.inserted, report.skipped, report.count_after), (1, 38, 39))

    def test_reset_empties_the_collection(self):
        client = open_client(self.tmp)
        ingest(reset_collection(client, COLLECTION), self.records)
        self.assertEqual(reset_collection(client, COLLECTION).count(), 0)


@unittest.skipUnless(config.EMBEDDINGS_FILE.exists(), "run embedding.embed first")
class ChromaRetrievalTest(TempChromaTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.collection = reset_collection(open_client(cls.tmp), COLLECTION)
        ingest(cls.collection, cls.records)
        cls.model = CountingModel(shared_model())
        cls.result = chroma_retrieve(QUERY, top_k=5, model=cls.model, collection=cls.collection)

    def test_query_embedding_has_384_dimensions(self):
        self.assertEqual(self.result.query_vector.shape, (config.EMBEDDING_DIM,))

    def test_only_the_query_is_embedded(self):
        for texts in self.model.encoded:
            self.assertEqual(texts, [QUERY])

    def test_returns_top_k(self):
        self.assertEqual(len(self.result.results), 5)
        self.assertEqual([r.rank for r in self.result.results], [1, 2, 3, 4, 5])
        everything = chroma_retrieve(QUERY, top_k=1000, model=self.model, collection=self.collection)
        self.assertEqual(len(everything.results), 39)  # never more than are stored

    def test_results_carry_chunk_information(self):
        for res in self.result.results:
            rec = res.record
            self.assertTrue(rec["chunk_id"] and rec["text"] and rec["section"])
            self.assertIn("source_file", rec["metadata"])
            self.assertLessEqual(rec["page_start"], rec["page_end"])

    def test_distances_ascending(self):
        distances = [r.distance for r in self.result.results]
        self.assertEqual(distances, sorted(distances))

    def test_matches_brute_force_cosine(self):
        # Same query vector, same stored vectors, same metric: same top-5 and 1 - distance == cosine.
        brute = brute_force_retrieve(QUERY, top_k=5, model=shared_model(), records=self.records)
        self.assertEqual([r.chunk_id for r in self.result.results], [r.chunk_id for r in brute.results])
        for chroma_res, brute_res in zip(self.result.results, brute.results):
            self.assertAlmostEqual(chroma_res.cosine_similarity, brute_res.score, places=5)

    def test_empty_collection_is_an_error(self):
        empty = reset_collection(open_client(self.tmp), "test_empty_chunks")
        with self.assertRaisesRegex(RetrievalError, "empty"):
            chroma_retrieve(QUERY, top_k=5, model=self.model, collection=empty)

    def test_invalid_requests(self):
        with self.assertRaisesRegex(RetrievalError, "top_k"):
            chroma_retrieve(QUERY, top_k=0, model=self.model, collection=self.collection)
        with self.assertRaisesRegex(RetrievalError, "empty"):
            chroma_retrieve("  ", top_k=5, model=self.model, collection=self.collection)


if __name__ == "__main__":
    unittest.main()
