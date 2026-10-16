import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from repair_scipy import bad_sections
from evaluate_retrieval import score_question
from rag_common import retrieve


class RetrievalTests(unittest.TestCase):
    def test_macho_changes_only_offset(self):
        header = struct.pack('<8I', 0xFEEDFACF, 0x0100000C, 0, 6, 1, 152, 0, 0)
        segment = struct.pack('<II16sQQQQIIII', 0x19, 152, b'__DATA', 0, 0, 0, 0, 7, 3, 1, 0)
        section = struct.pack('<16s16sQQIIIIIIII', b'__thread_bss', b'__DATA', 0, 64, 4096, 3, 0, 0, 0x12, 0, 0, 0)
        original = header + segment + section
        changes = bad_sections(original)
        self.assertEqual(changes, [{'field_position': 152, 'old_offset': 4096}])
        repaired = bytearray(original)
        struct.pack_into('<I', repaired, 152, 0)
        self.assertEqual(bad_sections(repaired), [])
        self.assertEqual(repaired[:152], original[:152])
        self.assertEqual(repaired[156:], original[156:])

    def test_malformed_macho_is_refused(self):
        with self.assertRaises(ValueError):
            bad_sections(b'not a valid Mach-O')

    def test_all_evidence_groups_are_required(self):
        q = {'question_id': 'CUSTOM', 'evidence_groups': [{'any_unit_ids': ['product']}, {'any_unit_ids': ['policy']}]}
        hit = {'rank': 1, 'payload': {'unit_ids': ['product'], 'text': 'Produit.'}}
        score = score_question(q, [hit])
        self.assertTrue(score['hit'])
        self.assertFalse(score['all_groups'])
        self.assertEqual(score['group_recall'], 0.5)

    def test_missing_information_is_not_given_a_gold_score(self):
        q = {'question_id': 'CUSTOM', 'evidence_groups': []}
        score = score_question(q, [])
        self.assertIsNone(score['hit'])
        self.assertTrue(score['requires_generation_abstention_test'])

    def test_anchor_prevents_wrong_passage_on_same_page(self):
        q = {'question_id': 'Q02', 'evidence_groups': [{'any_unit_ids': ['lumera_p07']}]}
        hits = [{'rank': 1, 'payload': {'unit_ids': ['lumera_p07'], 'text': 'Livraison offerte.'}}]
        score = score_question(q, hits)
        self.assertTrue(score['all_groups'])
        self.assertFalse(score['anchored_all_groups'])

    def test_qdrant_cosine_and_filter_with_real_local_client(self):
        import numpy as np
        from qdrant_client import QdrantClient, models
        with tempfile.TemporaryDirectory() as directory:
            client = QdrantClient(path=directory)
            try:
                client.create_collection('adlp_fixed', vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
                client.upsert('adlp_fixed', points=[
                    models.PointStruct(id=1, vector=[1.0, 0.0], payload={'doc_id': 'lumera'}),
                    models.PointStruct(id=2, vector=[0.0, 1.0], payload={'doc_id': 'arpp'}),
                ], wait=True)
                hits, _ = retrieve(client, 'fixed', np.array([1.0, 0.0]), top_k=2)
                self.assertEqual(hits[0]['payload']['doc_id'], 'lumera')
                self.assertAlmostEqual(hits[0]['score'], 1.0, places=5)
                filtered, _ = retrieve(client, 'fixed', np.array([1.0, 0.0]), top_k=2, doc_id='arpp')
                self.assertEqual(len(filtered), 1)
                self.assertEqual(filtered[0]['payload']['doc_id'], 'arpp')
            finally:
                client.close()


if __name__ == '__main__':
    unittest.main()
