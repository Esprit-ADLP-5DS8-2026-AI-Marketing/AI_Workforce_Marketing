"""Métadonnées et traces du reranker : aucun poids téléchargé, aucun appel API."""
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rerank_rag import (build_rerank_passage, document_title, rerank_hits,
                        search_passages)


def hit(index, rank=None):
    return {'rank': rank or index, 'score': 1 - index / 100,
            'payload': {'chunk_id': f'chunk_{index}', 'text': f'Preuve {index}.',
                        'source': 'source.pdf', 'doc_id': 'reference',
                        'embedding_text': f'passage: Titre documentaire\nPreuve {index}.',
                        'section': 'Section documentaire', 'unit_ids': [f'unit_{index}'],
                        'token_count': 8}}


class FakeReranker:
    device = 'cpu'

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score_pairs(self, pairs):
        self.calls.append(pairs)
        return self.scores


class MetadataInputTests(unittest.TestCase):
    def test_existing_index_header_supplies_missing_document_title(self):
        payload = hit(1)['payload']
        self.assertEqual(document_title(payload), 'Titre documentaire')
        self.assertEqual(build_rerank_passage(payload),
                         'Titre documentaire\nSection documentaire\nPreuve 1.')

    def test_explicit_title_has_priority_over_embedding_header(self):
        payload = hit(1)['payload']
        payload['doc_title'] = 'Titre explicite'
        self.assertEqual(document_title(payload), 'Titre explicite')

    def test_source_filename_is_fallback_when_no_title_exists(self):
        payload = {'source': 'politique.pdf', 'section': 'Règle', 'text': 'Texte original.'}
        self.assertEqual(build_rerank_passage(payload), 'politique.pdf\nRègle\nTexte original.')

    def test_body_is_not_interpreted_as_an_embedding_title_without_header_line(self):
        payload = {'source': 'source.pdf', 'embedding_text': 'passage: Un corps sans en-tête',
                   'text': 'Preuve originale.'}
        self.assertEqual(document_title(payload), 'source.pdf')

    def test_section_already_at_start_of_body_is_not_duplicated(self):
        payload = hit(1)['payload']
        payload['text'] = 'Section documentaire\nPreuve 1.'
        result = build_rerank_passage(payload)
        self.assertEqual(result.count('Section documentaire'), 1)
        self.assertTrue(result.endswith(payload['text']))

    def test_missing_metadata_keeps_body_and_never_invents_a_source(self):
        payload = {'text': '  Corps original\navec sa mise en forme.  '}
        self.assertEqual(build_rerank_passage(payload), payload['text'])

    def test_legacy_mode_reproduces_old_input_exactly(self):
        payload = hit(1)['payload']
        self.assertEqual(build_rerank_passage(payload, 'legacy'), payload['text'])
        payload['doc_title'] = 'Titre explicite'
        self.assertEqual(build_rerank_passage(payload, 'legacy'),
                         'Titre explicite\n' + payload['text'])

    def test_reranker_gets_metadata_but_returned_evidence_is_unchanged(self):
        candidates = [hit(1), hit(2)]
        original = copy.deepcopy(candidates)
        model = FakeReranker([0., 2.])
        final = rerank_hits('Question ?', candidates, model, 1)
        self.assertIn('Titre documentaire', model.calls[0][0][1])
        self.assertNotIn('passage:', model.calls[0][0][1])
        self.assertEqual(final[0]['payload'], original[1]['payload'])
        self.assertEqual(final[0]['score'], original[1]['score'])
        self.assertEqual(candidates, original)

    def test_evaluation_annotations_are_never_added_to_reranker_input(self):
        payload = hit(1)['payload']
        payload['expected_answer_notes'] = 'GOLD ANSWER MUST NOT BE USED'
        payload['evidence_groups'] = [{'answer': 'SECRET_GOLD'}]
        result = build_rerank_passage(payload)
        self.assertNotIn('GOLD', result)

    def test_invalid_mode_fails_before_inference(self):
        model = FakeReranker([0.])
        for value in ['other', None, []]:
            with self.subTest(mode=value), self.assertRaises(ValueError):
                rerank_hits('Q', [hit(1)], model, input_mode=value)
        self.assertEqual(model.calls, [])


class RankingTraceTests(unittest.TestCase):
    def test_all_twenty_candidates_are_scored_once_and_traced_without_changing_pool(self):
        candidates = [hit(i) for i in range(1, 21)]
        original = copy.deepcopy(candidates)
        model = FakeReranker([float(i) for i in range(20)])
        with patch('rag_common.retrieve', return_value=(candidates, 1.)):
            final, pool, details = search_passages('client', 'sections', 'vector', 'Q',
                                                  top_k=4, reranker=model, candidate_k=20)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(model.calls[0]), 20)
        self.assertEqual([h['payload']['chunk_id'] for h in final],
                         ['chunk_20', 'chunk_19', 'chunk_18', 'chunk_17'])
        ranking = details['candidate_ranking']
        self.assertEqual(len(ranking), 20)
        self.assertEqual([r['rerank_rank'] for r in ranking], list(range(1, 21)))
        self.assertEqual(ranking[-1]['retrieval_rank'], 1)
        self.assertEqual(ranking[-1]['rerank_score'], 0.)
        self.assertFalse(ranking[-1]['selected'])
        self.assertEqual(sum(r['selected'] for r in ranking), 4)
        self.assertEqual(details['reranker']['input_version'], 'title_section_v1')
        self.assertEqual(pool, original)
        self.assertEqual(candidates, original)

    def test_legacy_trace_records_the_old_input_version(self):
        candidates = [hit(1)]
        model = FakeReranker([1.])
        with patch('rag_common.retrieve', return_value=(candidates, 1.)):
            _, _, details = search_passages('client', 'sections', 'vector', 'Q',
                                            reranker=model, rerank_input='legacy')
        self.assertEqual(model.calls[0][0], ('Q', 'Preuve 1.'))
        self.assertEqual(details['reranker']['input_version'], 'legacy_v1')

    def test_empty_pool_does_not_run_the_model(self):
        model = FakeReranker([])
        with patch('rag_common.retrieve', return_value=([], 1.)):
            final, pool, details = search_passages('client', 'sections', 'vector', 'Q', reranker=model)
        self.assertEqual((final, pool, details['candidate_ranking']), ([], [], []))
        self.assertEqual(model.calls, [])

    def test_dense_search_keeps_original_results_and_has_no_rerank_trace(self):
        candidates = [hit(1)]
        with patch('rag_common.retrieve', return_value=(candidates, 1.)):
            final, pool, details = search_passages('client', 'sections', 'vector', 'Q')
        self.assertIs(final, candidates)
        self.assertIs(pool, candidates)
        self.assertEqual(details['candidate_ranking'], [])


class EvaluationIntegrationTests(unittest.TestCase):
    def test_both_input_modes_keep_complete_trace_and_preserve_previous_results(self):
        import evaluate_retrieval as evaluation
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data/processed').mkdir(parents=True)
            (root / 'data/processed/corpus_units.jsonl').write_text(
                json.dumps({'unit_id': 'unit_18'}) + '\n', encoding='utf-8')
            question = {'question_id': 'CUSTOM', 'question': 'Question ?', 'kind': 'answerable',
                        'evidence_groups': [{'any_unit_ids': ['unit_18']}]}
            path = root / 'questions.jsonl'
            path.write_text(json.dumps(question) + '\n', encoding='utf-8')
            previous = root / 'data/evaluation/results/reranking/comparison.csv'
            previous.parent.mkdir(parents=True)
            previous.write_text('ORIGINAL CSV', encoding='utf-8')
            manifest = {'collections': {s: {'name': f'adlp_{s}', 'points': 20}
                                       for s in ('fixed', 'sections')}}
            candidates = [hit(i) for i in range(1, 21)]

            def retrieve(_client, _strategy, _vector, limit, _filter):
                return candidates[:limit], 1.

            for mode in ('legacy', 'metadata'):
                out = root / f'results_{mode}'
                model, client = FakeReranker(list(range(20))), Mock()
                with patch.object(evaluation, 'ROOT', root), \
                     patch.object(evaluation, 'load_encoder'), \
                     patch.object(evaluation, 'encode', return_value=[[]]), \
                     patch.object(evaluation, 'load_reranker', return_value=model), \
                     patch.object(evaluation, 'open_index', return_value=(client, manifest)), \
                     patch('rag_common.retrieve', side_effect=retrieve), \
                     patch.object(sys, 'argv', ['evaluate_retrieval.py', '--compare-reranking',
                                               '--questions', str(path), '--rerank-input', mode,
                                               '--output-dir', str(out)]), \
                     patch('sys.stdout', new_callable=io.StringIO):
                    evaluation.main()
                report = json.loads((out / 'retrieval_report.json').read_text(encoding='utf-8'))
                self.assertEqual(report['rerank_input'], mode)
                self.assertEqual(report['reranker']['input_mode'], mode)
                rows = [json.loads(line) for line in (out / 'retrieved_contexts.jsonl').read_text().splitlines()]
                reranked = [r for r in rows if r['mode'] == 'rerank']
                self.assertEqual(len(reranked), 2)
                for row in reranked:
                    self.assertEqual(len(row['retrieval']['candidate_ranking']), 20)
                    self.assertEqual(len(row['candidate_hits']), 20)
                    self.assertEqual(len(row['hits']), 4)
                    self.assertTrue(row['scores']['hit'])
                expected_passage = 'Preuve 1.' if mode == 'legacy' else (
                    'Titre documentaire\nSection documentaire\nPreuve 1.')
                self.assertEqual(model.calls[0][0][1], expected_passage)
                self.assertEqual(previous.read_text(encoding='utf-8'), 'ORIGINAL CSV')
                client.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
