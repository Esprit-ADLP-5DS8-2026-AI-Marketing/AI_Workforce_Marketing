"""Sélection et traçabilité du prompt, sans clé ni appel LLM réel."""
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import answer_rag


class PromptVersionTests(unittest.TestCase):
    def test_prompt_selection_changes_only_instructions_and_trace_metadata(self):
        sources = [{'source_id': 'S1', 'chunk_id': 'c1', 'source': 'notice.pdf',
                    'page_start': 1, 'page_end': 1, 'text': 'Cet appareil est réservé à un usage domestique.',
                    'retrieval_score': .8}]
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'prompts').mkdir()
            for version in answer_rag.PROMPT_VERSIONS:
                (root / 'prompts' / f'{version}.txt').write_bytes(
                    (ROOT / 'prompts' / f'{version}.txt').read_bytes())
            # v1.3 par défaut ; les versions précédentes restent disponibles.
            # Le chemin de recherche et le contexte restent identiques.
            for args, expected in [([], 'rag_answer_v1_3'),
                                   (['--prompt-version', 'rag_answer_v1_2'], 'rag_answer_v1_2'),
                                   (['--prompt-version', 'rag_answer_v1_1'], 'rag_answer_v1_1')]:
                def capture(record):
                    calls.append(copy.deepcopy(record))
                    return root / 'trace.json'
                with patch.object(answer_rag, 'ROOT', root), \
                     patch.object(answer_rag, 'load_project_env'), \
                     patch.object(answer_rag, 'get_sources', return_value=(copy.deepcopy(sources), 1.)) as retrieve, \
                     patch.object(answer_rag, 'save_run', side_effect=capture), \
                     patch.object(answer_rag, 'call_gemini') as api, \
                     patch.object(sys, 'argv', ['answer_rag.py', 'Quelle est la puissance ?',
                                               '--strategy', 'sections', '--rerank', '--candidate-k', '20',
                                               '--top-k', '8', '--offline', '--dry-run', *args]), \
                     patch('sys.stdout', new_callable=io.StringIO):
                    answer_rag.main()
                api.assert_not_called()
                retrieve.assert_called_once_with('Quelle est la puissance ?', 'sections', 8, True,
                                                  use_reranker=True, candidate_k=20, doc_id=None,
                                                  device='cpu', diagnostics={})
                record = calls[-1]
                text = (root / 'prompts' / f'{expected}.txt').read_text(encoding='utf-8')
                self.assertEqual(record['prompt_version'], expected)
                self.assertEqual(record['prompt_sha256'], hashlib.sha256(text.encode('utf-8')).hexdigest())
                self.assertEqual(record['sources'], sources)
                self.assertFalse(record['live_api_called'])
            self.assertNotEqual(calls[0]['prompt_sha256'], calls[1]['prompt_sha256'])

    def test_selected_prompt_is_sent_to_gemini_without_leaking_examples_into_sources(self):
        source = {'source_id': 'S1', 'text': 'Texte documentaire original.'}
        prompt = (ROOT / 'prompts/rag_answer_v1_3.txt').read_text(encoding='utf-8')
        body = answer_rag.build_request('Question réelle ?', [source], prompt)
        self.assertEqual(body['systemInstruction']['parts'][0]['text'], prompt)
        data = json.loads(body['contents'][0]['parts'][0]['text'])
        self.assertEqual(data, {'question': 'Question réelle ?', 'sources': [source]})

    def test_missing_prompt_stops_before_retrieval_or_api(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(answer_rag, 'ROOT', Path(folder)), \
             patch.object(answer_rag, 'load_project_env'), \
             patch.object(answer_rag, 'get_sources') as retrieve, \
             patch.object(answer_rag, 'call_gemini') as api, \
             patch.object(sys, 'argv', ['answer_rag.py', 'Question ?', '--dry-run']):
            with self.assertRaisesRegex(SystemExit, 'Prompt absent'):
                answer_rag.main()
        retrieve.assert_not_called()
        api.assert_not_called()

    def test_prompt_filename_cannot_be_supplied_as_a_path(self):
        with patch.object(answer_rag, 'load_project_env'), \
             patch.object(answer_rag, 'get_sources') as retrieve, \
             patch.object(sys, 'argv', ['answer_rag.py', 'Q', '--prompt-version', '../other']), \
             patch('sys.stderr', new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as caught:
                answer_rag.main()
        self.assertEqual(caught.exception.code, 2)
        retrieve.assert_not_called()


if __name__ == '__main__':
    unittest.main()
