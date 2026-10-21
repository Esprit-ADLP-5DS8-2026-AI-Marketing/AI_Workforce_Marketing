"""Contrôles des preuves et du protocole, sans téléchargement ni appel LLM."""
import copy
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from validate_evaluation import (file_sha, read_jsonl, validate_dataset,
                                 validate_question)
from evaluate_retrieval import score_question

DATA = ROOT / 'data/evaluation/v2'
CORPUS = ROOT / 'data/processed/corpus_units.jsonl'


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units = {u['unit_id']: u for u in read_jsonl(CORPUS)}
        cls.dev = read_jsonl(DATA / 'questions_dev.jsonl')

    def test_all_annotations_match_frozen_corpus(self):
        report = validate_dataset(DATA, CORPUS)
        self.assertEqual(report['splits']['dev']['count'], 20)
        self.assertEqual(report['splits']['test']['count'], 10)
        self.assertEqual(report['proof_quotes_checked'], 36)

    def test_unknown_unit_is_rejected(self):
        q = copy.deepcopy(self.dev[0])
        q['evidence_groups'][0]['any_unit_ids'] = ['missing_page']
        with self.assertRaisesRegex(ValueError, 'absente du corpus'):
            validate_question(q, self.units, 'dev')

    def test_invented_proof_is_rejected(self):
        q = copy.deepcopy(self.dev[0])
        q['evidence_groups'][0]['evidence_quotes'][0]['quote'] = 'Niacinamide à 99 %'
        with self.assertRaisesRegex(ValueError, 'Extrait absent'):
            validate_question(q, self.units, 'dev')

    def test_quote_from_another_unit_is_rejected(self):
        q = copy.deepcopy(self.dev[0])
        q['evidence_groups'][0]['evidence_quotes'][0]['unit_id'] = 'arpp_p02'
        with self.assertRaisesRegex(ValueError, 'autre unité'):
            validate_question(q, self.units, 'dev')

    def test_unit_reference_without_focused_anchor_is_rejected(self):
        q = copy.deepcopy(self.dev[0])
        q['evidence_groups'][0]['evidence_quotes'][0]['quote'] = 'Texture : gel-creme, fini non gras'
        with self.assertRaisesRegex(ValueError, 'ancre manque'):
            validate_question(q, self.units, 'dev')

    def test_unanswerable_cannot_have_positive_gold(self):
        q = copy.deepcopy(self.dev[0])
        q.update(kind='unanswerable', missing_information='X', absence_note='Non fourni.')
        with self.assertRaisesRegex(ValueError, 'preuve positive'):
            validate_question(q, self.units, 'dev')

    def test_unanswerable_requires_manual_absence_explanation(self):
        q = copy.deepcopy(self.dev[11])
        del q['absence_note']
        with self.assertRaisesRegex(ValueError, 'Justification'):
            validate_question(q, self.units, 'dev')

    def test_changed_corpus_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / 'corpus.jsonl'
            corpus.write_bytes(CORPUS.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'corpus diffère'):
                validate_dataset(DATA, corpus)

    def test_unreviewed_dataset_edit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'v2'
            shutil.copytree(DATA, data)
            path = data / 'questions_dev.jsonl'
            path.write_bytes(path.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'empreinte différente'):
                validate_dataset(data, CORPUS)

    def test_seen_legacy_question_cannot_be_moved_to_test(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'v2'
            shutil.copytree(DATA, data)
            dev, test = read_jsonl(data / 'questions_dev.jsonl'), read_jsonl(data / 'questions_test.jsonl')
            dev[0], test[0] = test[0], dev[0]
            manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
            for split, rows in [('dev', dev), ('test', test)]:
                for q in rows:
                    q['split'] = split
                path = data / f'questions_{split}.jsonl'
                path.write_text(''.join(json.dumps(q, ensure_ascii=False) + '\n' for q in rows), encoding='utf-8')
                manifest['splits'][split].update(sha256=file_sha(path),
                                                 question_ids=[q['question_id'] for q in rows])
            (data / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'ancienne question'):
                validate_dataset(data, CORPUS)

    def test_wrong_fragment_of_correct_page_fails_anchors(self):
        question = self.dev[0]
        hits = [{'rank': 1, 'payload': {'unit_ids': ['lumera_p04'],
                                       'text': 'Texture : gel-creme, fini non gras'}}]
        result = score_question(question, hits)
        self.assertTrue(result['hit'])
        self.assertFalse(result['anchored_all_groups'])

    def test_correct_alternative_page_is_accepted(self):
        hits = [{'rank': 2, 'payload': {'unit_ids': ['lumera_p04'],
                                       'text': 'Niacinamide a 4 % pour unifier le teint'}}]
        result = score_question(self.dev[0], hits)
        self.assertTrue(result['anchored_all_groups'])
        self.assertEqual(result['reciprocal_rank'], 0.5)


class EvaluatorIntegrationTests(unittest.TestCase):
    def test_dev_and_test_paths_are_separate_and_only_selected_questions_are_encoded(self):
        import evaluate_retrieval as evaluation
        # Faux modèles/recherche : aucun résultat réel du test final n'est calculé.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(DATA, root / 'data/evaluation/v2')
            (root / 'data/processed').mkdir(parents=True)
            shutil.copyfile(CORPUS, root / 'data/processed/corpus_units.jsonl')
            for split, count in [('dev', 20), ('test', 10)]:
                client = Mock()
                manifest = {'collections': {s: {'name': f'adlp_{s}', 'points': n}
                                           for s, n in [('fixed', 79), ('sections', 144)]}}
                details = {'retrieval_ms': 1., 'rerank_ms': 0., 'total_search_ms': 1., 'reranker': None}
                with patch.object(evaluation, 'ROOT', root), \
                     patch.object(evaluation, 'load_encoder'), \
                     patch.object(evaluation, 'encode', return_value=[[]] * count) as encode, \
                     patch.object(evaluation, 'load_reranker'), \
                     patch.object(evaluation, 'open_index', return_value=(client, manifest)), \
                     patch.object(evaluation, 'search_passages', return_value=([], [], details)), \
                     patch.object(sys, 'argv', ['evaluate_retrieval.py', '--split', split, '--compare-reranking']), \
                     patch('sys.stdout', new_callable=io.StringIO):
                    evaluation.main()
                expected = read_jsonl(DATA / f'questions_{split}.jsonl')
                self.assertEqual(encode.call_args.args[1], ['query: ' + q['question'] for q in expected])
                out = root / 'data/evaluation/results/v2' / split / 'reranking'
                report = json.loads((out / 'retrieval_report.json').read_text(encoding='utf-8'))
                self.assertEqual(report['split'], split)
                self.assertEqual(report['questions_sha256'], file_sha(DATA / f'questions_{split}.jsonl'))
                self.assertFalse(report['generation_evaluated'])
                rows = read_jsonl(out / 'retrieved_contexts.jsonl')
                self.assertEqual(len(rows), count * 4)
                self.assertEqual({r['question_id'] for r in rows}, {q['question_id'] for q in expected})
                client.close.assert_called_once()
            self.assertTrue((root / 'data/evaluation/results/v2/dev/reranking/comparison.csv').is_file())
            self.assertTrue((root / 'data/evaluation/results/v2/test/reranking/comparison.csv').is_file())


if __name__ == '__main__':
    unittest.main()
