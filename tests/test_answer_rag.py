"""Tests locaux sans clé, sans accès réseau et sans chargement E5."""
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from answer_rag import validate_answer, parse_response, call_gemini, build_request, load_project_env


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.sources = [{'source_id': 'S1', 'text': 'Niacinamide\n4 %', 'retrieval_score': 0.9}]
        self.result = {'status': 'answered', 'answer': 'La fiche indique 4 % [S1].',
                       'citations': [{'source_id': 'S1', 'quote': 'Niacinamide 4 %'}], 'limitations': []}

    def test_exact_quote_with_normalized_whitespace(self):
        self.assertEqual(validate_answer(self.result, self.sources), self.result)

    def test_added_table_separator_is_rejected(self):
        self.result['citations'][0]['quote'] = 'Niacinamide | 4 %'
        with self.assertRaises(ValueError):
            validate_answer(self.result, self.sources)

    def test_invented_source_is_rejected(self):
        self.result['citations'][0]['source_id'] = 'S9'
        with self.assertRaises(ValueError):
            validate_answer(self.result, self.sources)

    def test_invented_quote_is_rejected(self):
        self.result['citations'][0]['quote'] = 'Niacinamide 10 %'
        with self.assertRaises(ValueError):
            validate_answer(self.result, self.sources)

    def test_missing_information_can_have_no_citation(self):
        result = {'status': 'insufficient', 'answer': 'SPF absent des passages.',
                  'citations': [], 'limitations': ['Indice SPF non documenté dans le contexte.']}
        validate_answer(result, self.sources)
        result['status'] = 'answered'
        with self.assertRaises(ValueError):
            validate_answer(result, self.sources)

    def test_inline_reference_must_match_citations(self):
        self.result['answer'] = 'La fiche indique 4 % [S2].'
        with self.assertRaises(ValueError):
            validate_answer(self.result, self.sources)

    def test_truncated_response_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_response({'candidates': [{'finishReason': 'MAX_TOKENS'}]})

    def test_request_uses_header_and_parses_response(self):
        body = build_request('Concentration ?', self.sources, 'Instructions système')
        self.assertNotIn('retrieval_score', body['contents'][0]['parts'][0]['text'])
        raw = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
            {'text': 'raisonnement caché', 'thought': True},
            {'text': json.dumps(self.result)},
        ]}}]}
        def fake_open(request, timeout):
            self.assertNotIn('FAKE_KEY', request.full_url)
            self.assertEqual(request.get_header('X-goog-api-key'), 'FAKE_KEY')
            self.assertEqual(json.loads(request.data), body)
            return io.BytesIO(json.dumps(raw).encode())
        with patch('urllib.request.urlopen', fake_open):
            result = parse_response(call_gemini('gemini-3.7-flash', 'FAKE_KEY', body))
        validate_answer(result, self.sources)

    def test_quota_error_does_not_expose_response_or_key(self):
        error = urllib.error.HTTPError('https://example.invalid', 429, 'Quota', {},
                                       io.BytesIO(b'FAKE_KEY secret error'))
        with patch('urllib.request.urlopen', side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                call_gemini('gemini-3.7-flash', 'FAKE_KEY', {})
        self.assertIn('429', str(caught.exception))
        self.assertNotIn('FAKE_KEY', str(caught.exception))

    def test_google_diagnostic_is_shown_with_key_redacted(self):
        body = json.dumps({'error': {'status': 'INVALID_ARGUMENT',
            'message': 'Invalid responseSchema. key=FAKE_KEY other=AIzaFakeExample123'}}).encode()
        error = urllib.error.HTTPError('https://example.invalid', 400, 'Bad request', {}, io.BytesIO(body))
        with patch('urllib.request.urlopen', side_effect=error) as opener:
            with self.assertRaises(RuntimeError) as caught:
                call_gemini('gemini-3.7-flash', 'FAKE_KEY', {})
        message = str(caught.exception)
        self.assertIn('INVALID_ARGUMENT', message)
        self.assertIn('responseSchema', message)
        self.assertNotIn('FAKE_KEY', message)
        self.assertNotIn('AIzaFakeExample123', message)
        self.assertEqual(opener.call_count, 1)  # Pas de relance sur un HTTP 400.

    def test_503_is_retried_with_bounded_attempts(self):
        errors = [urllib.error.HTTPError('https://example.invalid', 503, 'Busy', {}, io.BytesIO(b'{}'))
                  for _ in range(3)]
        with patch('urllib.request.urlopen', side_effect=errors) as opener:
            with patch('answer_rag.time.sleep') as sleeper:
                with self.assertRaises(RuntimeError) as caught:
                    call_gemini('gemini-3.7-flash', 'FAKE_KEY', {})
        self.assertIn('503', str(caught.exception))
        self.assertEqual(opener.call_count, 3)
        self.assertEqual(sleeper.call_count, 2)

    def test_env_loads_credentials_and_model_without_shell_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / '.env'
            path.write_text('GEMINI_API_KEY=FAKE_KEY\nGEMINI_MODEL=gemini-3.5-flash-lite\nUNRELATED=value\n')
            with patch.dict(os.environ, {}, clear=True):
                load_project_env(path)
                self.assertEqual(os.environ['GEMINI_API_KEY'], 'FAKE_KEY')
                self.assertEqual(os.environ['GEMINI_MODEL'], 'gemini-3.5-flash-lite')
                self.assertNotIn('UNRELATED', os.environ)

    def test_env_does_not_overwrite_explicit_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / '.env'
            path.write_text('GEMINI_API_KEY=FILE_KEY\nGEMINI_MODEL=file_model\n')
            with patch.dict(os.environ, {'GEMINI_API_KEY': 'SHELL_KEY', 'GEMINI_MODEL': 'shell_model'}, clear=True):
                load_project_env(path)
                self.assertEqual(os.environ['GEMINI_API_KEY'], 'SHELL_KEY')
                self.assertEqual(os.environ['GEMINI_MODEL'], 'shell_model')


if __name__ == '__main__':
    unittest.main()
