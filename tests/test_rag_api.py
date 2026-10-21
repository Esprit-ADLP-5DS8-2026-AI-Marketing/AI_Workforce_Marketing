"""Vrais appels FastAPI, PDF et Qdrant ; modèles et Gemini simulés, sans réseau."""
import asyncio
from contextlib import closing
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

import pymupdf
from fastapi.testclient import TestClient
from qdrant_client import models
from qdrant_client.local.persistence import CollectionPersistence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
from test_product_ingestion import FakeEncoder, make_pdf, vector
from local_qdrant import LocalQdrantClient as QdrantClient
from configure_rag_api import configure
from product_rag import DIMENSION
from rag_api import BoundaryMiddleware, Query, create_app
from rag_service import RagService, ServiceError, Settings, load_settings

TOKEN = 'test-service-token-' + 'x' * 32


class FakeReranker:
    def score_pairs(self, pairs):
        return [float(len(pairs) - i) for i in range(len(pairs))]


class ApiTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        self.unraisable_errors = []
        unraisable = patch.object(sys, 'unraisablehook',
            lambda error: self.unraisable_errors.append((error.exc_type.__name__, str(error.exc_value))))
        unraisable.start()
        self.addCleanup(unraisable.stop)
        # Reproduire la contrainte macOS, même lors d'un test sous Linux.
        strict_sqlite = patch.object(CollectionPersistence, 'CHECK_SAME_THREAD', True)
        strict_sqlite.start()
        self.addCleanup(strict_sqlite.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.common_path = self.root / 'common'
        with closing(QdrantClient(path=str(self.common_path))) as client:
            client.create_collection('adlp_sections', vectors_config=models.VectorParams(
                size=DIMENSION, distance=models.Distance.COSINE))
            points = []
            for doc_id, text in [('meta', 'Les publicités doivent respecter les règles Meta.'),
                                  ('arpp', 'Référentiel ARPP : allégations hydratation.'),
                                  ('methods', 'Guide de rédaction et création visuelle.'),
                                  ('lumera', 'Produit fictif à exclure du corpus commun.')]:
                points.append(models.PointStruct(id=str(uuid.uuid4()), vector=vector().tolist(),
                    payload={'chunk_id': 'common_' + doc_id, 'doc_id': doc_id,
                             'source': doc_id + '.pdf', 'text': text, 'section': 'Référence',
                             'page_start': 1, 'page_end': 1, 'scope': 'FR', 'synthetic': False}))
            client.upsert('adlp_sections', points=points)
        self.pdf = self.root / 'test.pdf'
        make_pdf(self.pdf, [('1. Ingredients', 18), ('Niacinamide : 4 %.', 11)])
        self.settings = Settings(service_token=TOKEN, store_path=self.root / 'products',
                                 gemini_api_key='dummy-gemini-key', gemini_model='test-model', gate_timeout=.01)
        self.encoder = FakeEncoder()
        self.loader = patch('rag_service.load_encoder', return_value=self.encoder).start()
        self.rerank_loader = patch('rag_service.load_reranker', return_value=FakeReranker()).start()
        self.index_patch = patch('rag_service.open_index', side_effect=self.common_index).start()
        self.addCleanup(patch.stopall)
        self.service = RagService(self.settings)
        self.app = create_app(self.settings, self.service)
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        gc.collect()
        self.tmp.cleanup()
        self.assertEqual(self.unraisable_errors, [], 'Exception ignorée pendant le nettoyage Qdrant')

    def common_index(self):
        return QdrantClient(path=str(self.common_path)), {'collections': {'sections': {'points': 4}}}

    def headers(self, user='alice'):
        return {'Authorization': 'Bearer ' + TOKEN, 'X-User-ID': user}

    def upload(self, user='alice', product='creme', title='Document produit', **extra):
        return self.client.post(f'/products/{product}/documents', headers=self.headers(user),
            data={'title': title, 'synthetic': 'true', **extra},
            files={'file': ('test.pdf', self.pdf.read_bytes(), 'application/pdf')})

    def query(self, scope='common', **kwargs):
        return {'question': 'Quelle information ?', 'scope': scope, **kwargs}

    def assert_error(self, response, status, code):
        self.assertEqual(response.status_code, status, response.text)
        self.assertEqual(response.json()['detail']['code'], code)
        self.assertIn('request_id', response.json()['detail'])

    def test_health_and_models_loaded_once(self):
        self.assertEqual(self.client.get('/health').json()['status'], 'ready')
        for _ in range(2):
            response = self.client.post('/retrieve', headers=self.headers(), json=self.query())
            self.assertEqual(response.status_code, 200, response.text)
        self.loader.assert_called_once()
        self.rerank_loader.assert_called_once()

    def test_bearer_token_and_user_context_are_required(self):
        self.assert_error(self.client.get('/products'), 401, 'unauthorized')
        self.assert_error(self.client.get('/products', headers={'Authorization': 'Bearer wrong'}), 401, 'unauthorized')
        self.assert_error(self.client.get('/products', headers={'Authorization': 'Bearer ' + TOKEN}), 422, 'invalid_user_context')
        self.assert_error(self.client.get('/products', headers=self.headers('../alice')), 422, 'invalid_user_context')

    def test_upload_requires_authentication_before_processing(self):
        with patch.object(self.service, 'ingest') as ingest:
            response = self.client.post('/products/x/documents', files={'file': ('x.pdf', b'invalid')})
        self.assert_error(response, 401, 'unauthorized')
        ingest.assert_not_called()

    def test_upload_current_and_reuse(self):
        first = self.upload()
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()['status'], 'ready')
        self.assertFalse(first.json()['reused'])
        second = self.upload()
        self.assertTrue(second.json()['reused'])
        self.assertEqual(first.json()['document_id'], second.json()['document_id'])
        current = self.client.get('/products/creme/documents/current', headers=self.headers())
        self.assertEqual(current.json()['document_id'], first.json()['document_id'])

    def test_other_user_cannot_read_or_query_private_product(self):
        self.upload()
        self.assert_error(self.client.get('/products/creme/documents/current', headers=self.headers('bob')), 404, 'product_not_ready')
        response = self.client.post('/retrieve', headers=self.headers('bob'),
            json=self.query('product', product_id='creme'))
        self.assert_error(response, 404, 'product_not_ready')
        self.assertEqual(self.client.get('/products', headers=self.headers('bob')).json()['products'], [])

    def test_same_product_id_can_exist_for_two_users_without_mixing(self):
        a, b = self.upload().json(), self.upload(user='bob').json()
        self.assertNotEqual(a['document_id'], b['document_id'])
        response = self.client.post('/retrieve', headers=self.headers(), json=self.query('product', product_id='creme'))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(all(s['document_id'] == a['document_id'] for s in response.json()['sources']))

    def test_common_queries_work_without_product_and_exclude_demo_product(self):
        response = self.client.post('/retrieve', headers=self.headers(), json=self.query())
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertIsNone(data['document_id'])
        self.assertEqual({s['doc_id'] for s in data['sources']}, {'meta', 'arpp', 'methods'})

    def test_common_doc_filter_and_mixed_retrieval(self):
        response = self.client.post('/retrieve', headers=self.headers(), json=self.query(doc_id='meta'))
        self.assertEqual({s['doc_id'] for s in response.json()['sources']}, {'meta'})
        self.upload()
        response = self.client.post('/retrieve', headers=self.headers(), json=self.query('both', product_id='creme'))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual({s['origin'] for s in response.json()['sources']}, {'product', 'common'})
        self.assertNotIn('lumera', {s['doc_id'] for s in response.json()['sources']})

    def test_product_context_on_common_query_is_also_verified(self):
        self.upload()
        response = self.client.post('/retrieve', headers=self.headers('bob'),
                                    json=self.query('common', product_id='creme'))
        self.assert_error(response, 404, 'product_not_ready')

    def test_expired_campaign_document_version_is_refused(self):
        old = self.upload().json()
        self.upload(title='Version suivante')
        response = self.client.post('/retrieve', headers=self.headers(),
            json=self.query('product', product_id='creme', document_id=old['document_id']))
        self.assert_error(response, 409, 'document_changed')

    def test_request_validation_rejects_identity_override_and_invalid_budgets(self):
        for body in [self.query(user_id='bob'), self.query('both'), self.query(doc_id='lumera'),
                     self.query(top_k=9), self.query(top_k=True), self.query(top_k=8, candidate_k=4),
                     self.query(question=' '), self.query(document_id='prd_old'),
                     self.query('both', product_id='x', doc_id='meta')]:
            with self.subTest(body=body):
                self.assert_error(self.client.post('/retrieve', headers=self.headers(), json=body), 422, 'invalid_request')

    def test_upload_rejects_unknown_owner_form_field(self):
        self.assert_error(self.upload(user_id='bob'), 422, 'invalid_form')

    def test_upload_pdf_validation(self):
        for name, data, status, code in [('x.txt', b'hello', 415, 'unsupported_file'),
                                          ('x.pdf', b'not a pdf', 422, 'invalid_pdf')]:
            with self.subTest(name=name):
                result = self.client.post('/products/x/documents', headers=self.headers(),
                    files={'file': (name, data, 'application/pdf')})
                self.assert_error(result, status, code)

    def test_upload_filename_cannot_escape_temporary_directory(self):
        response = self.client.post('/products/x/documents', headers=self.headers(),
            files={'file': ('../../escaped.pdf', self.pdf.read_bytes(), 'application/pdf')})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['source'], 'escaped.pdf')

    def test_oversized_content_length_is_rejected(self):
        headers = {**self.headers(), 'Content-Length': str(22 * 1024 * 1024)}
        self.assert_error(self.client.post('/retrieve', headers=headers, content=b'x'), 413, 'request_too_large')

    def test_busy_service_has_retry_after(self):
        self.service._gate.acquire()
        try:
            response = self.client.post('/retrieve', headers=self.headers(), json=self.query())
        finally:
            self.service._gate.release()
        self.assert_error(response, 503, 'rag_busy')
        self.assertEqual(response.headers['Retry-After'], '3')

    def test_second_worker_on_same_product_store_is_refused(self):
        another = RagService(self.settings)
        with self.assertRaisesRegex(RuntimeError, 'seul worker'):
            another.start()

    def test_changed_common_index_is_http_503_not_system_exit(self):
        for error in (SystemExit('Index modifié'), ValueError('Comptage incohérent')):
            with self.subTest(error=type(error).__name__):
                self.index_patch.side_effect = error
                result = self.client.post('/retrieve', headers=self.headers(), json=self.query())
                self.assert_error(result, 503, 'common_index_unavailable')

    def test_answer_uses_gemini_schema_validates_quotes_and_writes_trace(self):
        self.upload()
        raw = {'modelVersion': 'fake-model', 'candidates': [{'finishReason': 'STOP', 'content': {
            'parts': [{'text': json.dumps({'status': 'answered', 'answer': '4 % [S1].',
                'citations': [{'source_id': 'S1', 'quote': 'Niacinamide : 4 %.'}], 'limitations': []})}]}}]}
        with patch('rag_service.call_gemini', return_value=raw) as gemini, \
             patch('rag_service.save_run', return_value=Path('api_answer.json')) as save:
            response = self.client.post('/answer', headers=self.headers(),
                json=self.query('product', product_id='creme', question='Concentration ?'))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()['validation_passed'])
        self.assertEqual(response.json()['trace_id'], 'api_answer.json')
        self.assertNotIn('dummy-gemini-key', response.text)
        record = save.call_args.args[0]
        self.assertEqual(record['user_id'], 'alice')
        self.assertEqual(record['request_id'], response.json()['request_id'])
        self.assertEqual(response.headers['X-Request-ID'], record['request_id'])
        contexts = json.loads(gemini.call_args.args[2]['contents'][0]['parts'][0]['text'])
        self.assertTrue(all('retrieval_score' not in s and 'rerank_score' not in s for s in contexts['sources']))

    def test_invalid_llm_answer_is_502_with_trace_id(self):
        raw = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': '{}'}]}}]}
        with patch('rag_service.call_gemini', return_value=raw), \
             patch('rag_service.save_run', return_value=Path('failed.json')):
            response = self.client.post('/answer', headers=self.headers(), json=self.query())
        self.assert_error(response, 502, 'invalid_llm_answer')
        self.assertEqual(response.json()['detail']['trace_id'], 'failed.json')

    def test_insufficient_is_a_valid_200_not_a_gateway_failure(self):
        result = {'status': 'insufficient', 'answer': 'Information indisponible.', 'citations': [], 'limitations': ['Valeur absente.']}
        raw = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': json.dumps(result)}]}}]}
        with patch('rag_service.call_gemini', return_value=raw), \
             patch('rag_service.save_run', return_value=Path('insufficient.json')):
            response = self.client.post('/answer', headers=self.headers(), json=self.query())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'insufficient')

    def test_gemini_errors_and_missing_key_are_explicit(self):
        for message, status, code in [('Gemini HTTP 503', 503, 'gemini_unavailable'),
                                      ('Gemini HTTP 429', 429, 'gemini_quota_exceeded'),
                                      ('Gemini HTTP 400', 502, 'gemini_configuration_error')]:
            with self.subTest(message=message), patch('rag_service.call_gemini', side_effect=RuntimeError(message)), \
                 patch('rag_service.save_run', return_value=Path('failed_api.json')):
                response = self.client.post('/answer', headers=self.headers(), json=self.query())
                self.assert_error(response, status, code)
        self.service.settings = replace(self.settings, gemini_api_key='')
        self.assert_error(self.client.post('/answer', headers=self.headers(), json=self.query()),
                          503, 'gemini_not_configured')

    def test_openapi_documents_bearer_header_and_actual_routes(self):
        schema = self.client.get('/openapi.json').json()
        self.assertIn('HTTPBearer', schema['components']['securitySchemes'])
        self.assertIn('/products/{product_id}/documents', schema['paths'])
        params = schema['paths']['/retrieve']['post']['parameters']
        self.assertIn('X-User-ID', {p['name'] for p in params})


class ConfigurationTests(unittest.TestCase):
    def test_setup_preserves_existing_gemini_settings_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            path = Path(folder) / '.env'
            original = 'GEMINI_API_KEY="existing-key"\nGEMINI_MODEL=existing-model\n'
            path.write_text(original)
            self.assertIn('RAG_SERVICE_TOKEN', configure(path))
            before = path.read_bytes()
            self.assertEqual(configure(path), [])
            self.assertEqual(path.read_bytes(), before)
            self.assertTrue(path.read_text().startswith(original))
            settings = load_settings(path)
            self.assertEqual(settings.gemini_api_key, 'existing-key')
            self.assertEqual(settings.gemini_model, 'existing-model')
            self.assertGreaterEqual(len(settings.service_token), 32)

    def test_missing_or_short_service_token_is_refused(self):
        for token in ['', 'short']:
            with self.assertRaises(ValueError):
                Settings(service_token=token)

    def test_streamed_body_limit_applies_without_content_length(self):
        called, sent = [], []
        async def downstream(*args):
            called.append(True)
        middleware = BoundaryMiddleware(downstream, TOKEN, body_limit=5)
        scope = {'type': 'http', 'method': 'POST', 'path': '/retrieve', 'headers': [
            (b'authorization', ('Bearer ' + TOKEN).encode()), (b'x-user-id', b'alice')]}
        messages = iter([{'type': 'http.request', 'body': b'123', 'more_body': True},
                         {'type': 'http.request', 'body': b'456', 'more_body': False}])
        async def receive():
            return next(messages)
        async def send(message):
            sent.append(message)
        asyncio.run(middleware(scope, receive, send))
        self.assertEqual(called, [])
        self.assertEqual(sent[0]['status'], 413)


if __name__ == '__main__':
    unittest.main()
