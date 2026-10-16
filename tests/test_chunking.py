"""Contrôles du texte, des limites réelles et des références de pages.

python tests/test_chunking.py
python tests/test_chunking.py --tokenizer-path /chemin/tokenizer
"""
import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import chunk_corpus as chunker


class ChunkingTests(unittest.TestCase):
    tokenizer = None

    def check_windows(self, text, sections=False):
        coverage = bytearray(len(text))
        ranges = list(chunker.token_windows(self.tokenizer, text, 64, 10, sections))
        self.assertGreater(len(ranges), 1)
        for start, end in ranges:
            self.assertLessEqual(chunker.count_tokens(self.tokenizer, text[start:end]), 64)
            coverage[start:end] = b"\1" * (end - start)
        self.assertTrue(all(coverage[i] or c.isspace() for i, c in enumerate(text)))

    def test_unicode_and_percentages_are_not_lost(self):
        text = ("Niacinamide à 4 %, acide hyaluronique à 0,8 %.\n"
                "Peau déshydratée — ingrédients et précautions.\n\n") * 25
        self.check_windows(text)
        self.check_windows(text, sections=True)

    def test_rule_continues_across_pages(self):
        base = {"doc_id": "arpp", "source": "source.pdf", "doc_type": "cosmetics_guidance",
                "source_type": "professional_reference", "scope": "FR", "synthetic": False,
                "section": "Page", "language": "fr"}
        units = [
            {**base, "unit_id": "arpp_p02", "page_start": 2, "page_end": 2,
             "text": "1.3 Présentation des performances\nDébut de la règle qui continue"},
            {**base, "unit_id": "arpp_p03", "page_start": 3, "page_end": 3,
             "text": "sur la page suivante.\n1.4 Cautions\nRègle suivante."},
        ]
        doc = chunker.build_documents(units)[0]
        regions = chunker.section_ranges(doc)
        start, end, title = next(r for r in regions if r[2].startswith("1.3"))
        self.assertIn("sur la page suivante", doc['text'][start:end])
        chunk = chunker.make_chunk(doc, start, end, "sections", 1, title, self.tokenizer)
        self.assertEqual(chunk['page_start'], 2)
        self.assertEqual(chunk['page_end'], 3)
        self.assertEqual(chunk['unit_ids'], ['arpp_p02', 'arpp_p03'])

    def test_two_sources_never_share_a_chunk(self):
        docs = chunker.build_documents(chunker.read_jsonl(ROOT / 'data/processed/corpus_units.jsonl'))
        for doc in docs:
            for start, end in chunker.token_windows(self.tokenizer, doc['text'], 300, 50):
                c = chunker.make_chunk(doc, start, end, 'fixed', 1, 'Fenêtre', self.tokenizer)
                self.assertEqual({x['source'] for x in c['citations']}, {c['source']})
                self.assertLessEqual(c['embedding_token_count'], 512)

    def test_evaluation_references_exist(self):
        units = chunker.read_jsonl(ROOT / 'data/processed/corpus_units.jsonl')
        ids = {u['unit_id'] for u in units}
        questions = chunker.read_jsonl(ROOT / 'data/evaluation/questions.jsonl')
        self.assertEqual(len(questions), 14)
        for q in questions:
            for group in q['evidence_groups']:
                self.assertTrue(set(group['any_unit_ids']) <= ids)
            self.assertTrue(set(q.get('supporting_unit_ids', [])) <= ids)
            if q['kind'] == 'unanswerable':
                self.assertEqual(q['evidence_groups'], [])

    def test_parent_title_attaches_to_child(self):
        doc = {'doc_id': 'arpp', 'text': '1. PRINCIPES GÉNÉRAUX\n1.2 Preuves\nUne allégation exige une preuve.', 'spans': []}
        regions = chunker.section_ranges(doc)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0][0], 0)
        self.assertIn('Une allégation', doc['text'][regions[0][0]:regions[0][1]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer-path', type=Path)
    parser.add_argument('--offline', action='store_true')
    args, unittest_args = parser.parse_known_args()
    ChunkingTests.tokenizer = chunker.load_tokenizer(args.tokenizer_path, args.offline)
    unittest.main(argv=[sys.argv[0]] + unittest_args)
