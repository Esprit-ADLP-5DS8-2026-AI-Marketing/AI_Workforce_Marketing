"""Régressions locales : aucun poids de modèle téléchargé, aucun appel Gemini."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from answer_rag import build_request, get_sources, validate_answer
from evaluate_retrieval import score_question
from rerank_rag import (MultilingualReranker, rerank_hits, search_passages,
                        validate_search_options)


def hit(index, rank, text=None):
    return {'rank': rank, 'score': 1.0 - rank / 100,
            'payload': {'chunk_id': f'chunk_{index}', 'text': text or f'Passage {index}',
                        'doc_id': 'lumera', 'source': 'produit.pdf', 'doc_title': 'Produit',
                        'page_start': index, 'page_end': index, 'doc_type': 'product',
                        'unit_ids': [f'unit_{index}'], 'token_count': 12,
                        'section': 'Ingrédients'}}


class FakeReranker:
    device = 'cpu'

    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def score_pairs(self, pairs):
        self.pairs = pairs
        return self.scores


class RerankingTests(unittest.TestCase):
    def test_lower_dense_rank_can_become_first_without_changing_evidence(self):
        hits = [hit(1, 1), hit(2, 2), hit(3, 3)]
        original = copy.deepcopy(hits)
        model = FakeReranker([-2.0, 5.0, 1.0])
        result = rerank_hits('Question ?', hits, model, top_k=2)
        self.assertEqual([h['payload']['chunk_id'] for h in result], ['chunk_2', 'chunk_3'])
        self.assertEqual([h['rank'] for h in result], [1, 2])
        self.assertEqual(result[0]['retrieval_rank'], 2)
        self.assertEqual(result[0]['score'], original[1]['score'])
        self.assertEqual(result[0]['payload'], original[1]['payload'])
        self.assertEqual(hits, original)
        self.assertEqual(model.pairs[0], ('Question ?', 'Produit\nIngrédients\nPassage 1'))

    def test_equal_logits_preserve_dense_order(self):
        hits = [hit(1, 1), hit(2, 2)]
        result = rerank_hits('Q', hits, FakeReranker([0.1, 0.1]))
        self.assertEqual([h['retrieval_rank'] for h in result], [1, 2])

    def test_fewer_candidates_than_requested_are_valid(self):
        result = rerank_hits('Q', [hit(1, 1)], FakeReranker([0.5]), 4)
        self.assertEqual(len(result), 1)

    def test_empty_pool_does_not_call_model(self):
        model = FakeReranker([])
        self.assertEqual(rerank_hits('Q', [], model), [])
        self.assertIsNone(model.pairs)

    def test_missing_title_uses_source_and_section_without_modifying_body(self):
        h = hit(1, 1)
        del h['payload']['doc_title']
        model = FakeReranker([1.0])
        rerank_hits('Q', [h], model)
        self.assertEqual(model.pairs, [('Q', 'produit.pdf\nIngrédients\nPassage 1')])

    def test_invalid_scores_fail_instead_of_silent_fallback(self):
        for scores in [[], [float('nan')], [float('inf')], [1.0, 2.0]]:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                rerank_hits('Q', [hit(1, 1)], FakeReranker(scores))

    def test_duplicate_or_empty_candidates_are_refused(self):
        invalid = hit(1, 2)
        for hits in [[hit(1, 1), invalid], [hit(2, 1, ' ')]]:
            with self.assertRaises(ValueError):
                rerank_hits('Q', hits, FakeReranker([1.0] * len(hits)))

    def test_candidate_budget_is_validated(self):
        for top, pool in [(4, 3), (4, 101), (0, 20), (True, 20), (4, True)]:
            with self.subTest(top=top, pool=pool), self.assertRaises(ValueError):
                validate_search_options(top, pool, True)

    def test_dense_path_keeps_existing_results(self):
        hits = [hit(1, 1)]
        with patch('rag_common.retrieve', return_value=(hits, 1.5)) as retrieve:
            result, pool, details = search_passages('client', 'sections', 'vector', 'Q', top_k=4)
        retrieve.assert_called_once_with('client', 'sections', 'vector', 4, None)
        self.assertIs(result, hits)
        self.assertIs(pool, hits)
        self.assertEqual(details['mode'], 'dense')
        self.assertEqual(details['rerank_ms'], 0.0)
        self.assertIsNone(details['reranker'])

    def test_rerank_path_uses_larger_pool_and_preserves_filter(self):
        hits = [hit(1, 1), hit(2, 2)]
        with patch('rag_common.retrieve', return_value=(hits, 1.5)) as retrieve:
            result, pool, details = search_passages(
                'client', 'sections', 'vector', 'Q', top_k=1, doc_id='lumera',
                reranker=FakeReranker([-2.0, 3.0]), candidate_k=20,
            )
        retrieve.assert_called_once_with('client', 'sections', 'vector', 20, 'lumera')
        self.assertEqual(result[0]['payload']['chunk_id'], 'chunk_2')
        self.assertEqual(pool, hits)
        self.assertEqual(details['candidates_returned'], 2)
        self.assertEqual(details['final_passages'], 1)
        self.assertEqual(details['reranker']['score_type'], 'raw_logit')

    def test_metrics_use_final_rank_not_original_dense_rank(self):
        result = rerank_hits('Q', [hit(1, 1), hit(2, 2)], FakeReranker([0, 1]), 1)
        q = {'question_id': 'CUSTOM', 'evidence_groups': [{'any_unit_ids': ['unit_2']}]}
        self.assertEqual(score_question(q, result)['reciprocal_rank'], 1.0)

    def test_no_gold_question_remains_outside_positive_metrics(self):
        q = {'question_id': 'UNKNOWN', 'evidence_groups': []}
        self.assertIsNone(score_question(q, [hit(1, 1)])['hit'])

    def test_pair_overflow_is_refused_before_inference(self):
        model = MultilingualReranker.__new__(MultilingualReranker)
        model.tokenizer = Mock(return_value={'input_ids': [[1] * 513]})
        model.model = Mock()
        with self.assertRaisesRegex(ValueError, '513 tokens'):
            model.score_pairs([('Q', 'long passage')])
        model.model.assert_not_called()
        self.assertFalse(model.tokenizer.call_args.kwargs['truncation'])


class AnswerIntegrationTests(unittest.TestCase):
    def test_citation_ids_are_assigned_after_reranking(self):
        import numpy as np
        client = Mock()
        diagnostics = {}
        with patch('rag_common.load_encoder'), patch('rag_common.encode', return_value=np.array([[1, 0]])), \
             patch('rag_common.open_index', return_value=(client, {})), \
             patch('rag_common.retrieve', return_value=([hit(1, 1), hit(2, 2)], 0.5)), \
             patch('rerank_rag.load_reranker', return_value=FakeReranker([0, 5])):
            sources, _ = get_sources('Q', 'sections', 1, True, use_reranker=True, diagnostics=diagnostics)
        self.assertEqual(sources[0]['source_id'], 'S1')
        self.assertEqual(sources[0]['chunk_id'], 'chunk_2')
        self.assertEqual(sources[0]['page_start'], 2)
        self.assertEqual(sources[0]['retrieval_rank'], 2)
        self.assertEqual(diagnostics['mode'], 'rerank')
        client.close.assert_called_once()

    def test_dense_answer_does_not_load_reranker(self):
        import numpy as np
        client = Mock()
        with patch('rag_common.load_encoder'), patch('rag_common.encode', return_value=np.array([[1, 0]])), \
             patch('rag_common.open_index', return_value=(client, {})), \
             patch('rag_common.retrieve', return_value=([hit(1, 1)], 0.5)), \
             patch('rerank_rag.load_reranker') as load:
            sources, latency = get_sources('Q', 'sections', 4, True)
        load.assert_not_called()
        self.assertEqual(latency, 0.5)
        self.assertNotIn('rerank_score', sources[0])

    def test_numeric_scores_and_dense_ranks_are_not_sent_to_gemini(self):
        source = {'source_id': 'S1', 'text': 'Niacinamide 4 %', 'page_start': 5,
                  'retrieval_score': 0.88, 'rerank_score': 9.41, 'retrieval_rank': 12}
        body = build_request('Q', [source], 'Instructions')
        context = json.loads(body['contents'][0]['parts'][0]['text'])['sources'][0]
        self.assertEqual(context, {'source_id': 'S1', 'text': 'Niacinamide 4 %', 'page_start': 5})

    def test_grouped_citations_remain_supported(self):
        sources = [{'source_id': 'S1', 'text': 'Niacinamide 4 %'},
                   {'source_id': 'S2', 'text': 'Niacinamide a 4 %'}]
        for answer in ['4 % [S1, S2].', '4 % [S1,S2].', '4 % [S1] [S2].']:
            response = {'status': 'answered', 'answer': answer, 'limitations': [],
                        'citations': [{'source_id': 'S1', 'quote': 'Niacinamide 4 %'},
                                      {'source_id': 'S2', 'quote': 'Niacinamide a 4 %'}]}
            self.assertEqual(validate_answer(response, sources), response)
        response['answer'] = '4 % [S1, S99].'
        with self.assertRaises(ValueError):
            validate_answer(response, sources)

    def test_reranking_does_not_make_invented_quotes_acceptable(self):
        response = {'status': 'answered', 'answer': '10 % [S1].', 'limitations': [],
                    'citations': [{'source_id': 'S1', 'quote': 'Niacinamide 10 %'}]}
        with self.assertRaises(ValueError):
            validate_answer(response, [{'source_id': 'S1', 'text': 'Niacinamide 4 %'}])


class QdrantIntegrationTests(unittest.TestCase):
    def test_real_qdrant_filter_applies_before_reranking(self):
        import numpy as np
        from qdrant_client import QdrantClient, models
        with tempfile.TemporaryDirectory() as directory:
            client = QdrantClient(path=directory)
            try:
                client.create_collection('adlp_sections', vectors_config=models.VectorParams(
                    size=2, distance=models.Distance.COSINE))
                points = [models.PointStruct(id=i, vector=[1.0, float(i) / 10],
                                             payload=hit(i, i)['payload']) for i in (1, 2, 3)]
                points[0].payload['doc_id'] = 'meta'
                client.upsert('adlp_sections', points=points, wait=True)
                model = FakeReranker([0.0, 4.0])
                final, pool, details = search_passages(
                    client, 'sections', np.array([1.0, 0.0]), 'Q', top_k=1,
                    doc_id='lumera', reranker=model, candidate_k=20,
                )
                self.assertEqual(len(pool), 2)
                self.assertTrue(all(h['payload']['doc_id'] == 'lumera' for h in pool))
                self.assertEqual(final[0]['payload']['chunk_id'], 'chunk_3')
                self.assertEqual(details['candidates_returned'], 2)
            finally:
                client.close()


class EvaluationIntegrationTests(unittest.TestCase):
    def test_four_comparisons_keep_pool_and_final_metrics_distinct(self):
        import numpy as np
        import evaluate_retrieval as evaluation

        candidates = [hit(i, i) for i in range(1, 7)]
        question = {'question_id': 'CUSTOM', 'question': 'Q', 'kind': 'answerable',
                    'evidence_groups': [{'any_unit_ids': ['unit_6']}]}
        unknown = {'question_id': 'UNKNOWN', 'question': 'Q inconnu', 'kind': 'unanswerable',
                   'evidence_groups': []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data/processed').mkdir(parents=True)
            (root / 'data/processed/corpus_units.jsonl').write_text(
                json.dumps({'unit_id': 'unit_6'}) + '\n')
            questions = root / 'questions.jsonl'
            questions.write_text('\n'.join(json.dumps(q) for q in [question, unknown]) + '\n')
            manifest = {'collections': {strategy: {'name': 'adlp_' + strategy, 'points': 6}
                                        for strategy in ('fixed', 'sections')}}
            client = Mock()

            def retrieve(_client, _strategy, _vector, limit, _filter):
                return candidates[:limit], 0.5

            with patch.object(evaluation, 'ROOT', root), \
                 patch.object(evaluation, 'load_encoder'), \
                 patch.object(evaluation, 'encode', return_value=np.array([[1, 0], [0, 1]])), \
                 patch.object(evaluation, 'load_reranker', return_value=FakeReranker([0, 0, 0, 0, 0, 5])), \
                 patch.object(evaluation, 'open_index', return_value=(client, manifest)), \
                 patch('rag_common.retrieve', side_effect=retrieve), \
                 patch.object(sys, 'argv', ['evaluate_retrieval.py', '--compare-reranking',
                                           '--questions', str(questions)]), \
                 patch('sys.stdout', new_callable=io.StringIO):
                evaluation.main()
            folder = root / 'data/evaluation/results/reranking'
            report = json.loads((folder / 'retrieval_report.json').read_text())
            self.assertEqual(set(report['strategies']), {
                'fixed_dense', 'fixed_rerank', 'sections_dense', 'sections_rerank'})
            self.assertEqual(report['strategies']['sections_dense']['unit_hit_rate'], 0)
            self.assertEqual(report['strategies']['sections_rerank']['unit_hit_rate'], 1)
            self.assertEqual(report['strategies']['sections_rerank']['candidate_unit_hit_rate'], 1)
            self.assertEqual(report['strategies']['sections_rerank']['questions_with_gold'], 1)
            self.assertEqual(report['strategies']['sections_rerank']['unanswerable_questions'], 1)
            self.assertFalse(report['generation_evaluated'])
            self.assertTrue((folder / 'comparison.csv').exists())
            client.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
