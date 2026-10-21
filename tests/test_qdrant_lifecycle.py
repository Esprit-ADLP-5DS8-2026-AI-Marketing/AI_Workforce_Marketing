"""Régression du nettoyage Qdrant/SQLite entre threads, sans modèle ni API."""
from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from local_qdrant import LocalQdrantClient
from qdrant_client import models
from qdrant_client.local.persistence import CollectionPersistence


class QdrantLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.strict = patch.object(CollectionPersistence, 'CHECK_SAME_THREAD', True)
        self.strict.start()
        self.addCleanup(self.strict.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def new_client(self):
        client = LocalQdrantClient(path=self.tmp.name)
        if not client.collection_exists('lifecycle'):
            client.create_collection('lifecycle', vectors_config=models.VectorParams(
                size=2, distance=models.Distance.COSINE))
        return client

    def test_second_close_after_worker_cleanup_is_harmless_from_another_thread(self):
        with ThreadPoolExecutor(max_workers=1) as worker:
            def create_and_close():
                client = self.new_client()
                client.close()
                return client
            client = worker.submit(create_and_close).result()
        client.close()  # Main thread, après une fermeture effective dans le worker.
        self.assertTrue(client._rag_cleanup_done)

    def test_finalizer_in_another_thread_does_not_reclose_sqlite(self):
        gc.collect()
        errors = []
        with patch.object(sys, 'unraisablehook', errors.append):
            with ThreadPoolExecutor(max_workers=1) as worker:
                def create_closed_cycle():
                    client = self.new_client()
                    client.close()
                    client._test_cycle = client
                    return weakref.ref(client)
                reference = worker.submit(create_closed_cycle).result()
            gc.collect()  # __del__ se produit sur le thread principal.
        self.assertIsNone(reference())
        self.assertEqual(errors, [])

    def test_active_connection_keeps_sqlite_thread_protection(self):
        client = self.new_client()
        try:
            with ThreadPoolExecutor(max_workers=1) as worker:
                with self.assertRaises(sqlite3.ProgrammingError):
                    worker.submit(client.close).result()
            self.assertFalse(client._rag_cleanup_done)
        finally:
            client.close()  # Nettoyage sur le thread créateur.

    def test_close_releases_storage_for_next_client(self):
        first = self.new_client()
        first.upsert('lifecycle', points=[models.PointStruct(id=1, vector=[1., 0.],
                                                            payload={'doc_id': 'test'})])
        first.close()
        second = self.new_client()
        try:
            self.assertEqual(second.count('lifecycle', exact=True).count, 1)
        finally:
            second.close()
        first.close()

    def test_constructor_failure_does_not_cause_another_finalizer_error(self):
        errors = []
        with patch.object(sys, 'unraisablehook', errors.append):
            with patch('qdrant_client.QdrantClient.__init__', side_effect=RuntimeError('init failed')):
                with self.assertRaises(RuntimeError):
                    LocalQdrantClient(path=self.tmp.name)
                gc.collect()
        self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()
