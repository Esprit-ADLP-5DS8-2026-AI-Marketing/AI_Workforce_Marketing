"""PDF réels temporaires et vrai Qdrant local ; modèles simulés, aucune API.

python tests/test_product_ingestion.py
"""
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import pymupdf
from qdrant_client import QdrantClient, models

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from product_rag import (COLLECTION, COMMON_DOCS, DIMENSION, ProductError, ProductStore,
    extract_pages, get_product_sources, ingest_product_pdf, make_product_chunks,
    owner_product_key, page_sections, read_pdf_bytes, search_scoped)
from query_product_rag import answer_from_sources


class WordTokenizer:
    """Offsets déterministes pour tester la couverture, sans modèle téléchargé."""
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        matches = list(re.finditer(r'\S+', text))
        result = {'input_ids': list(range(len(matches) + (2 if add_special_tokens else 0)))}
        if return_offsets_mapping:
            result['offset_mapping'] = [(m.start(), m.end()) for m in matches]
        return result


class FakeEncoder:
    tokenizer = WordTokenizer()
    def encode(self, texts, **kwargs):
        result = np.zeros((len(texts), DIMENSION), dtype=np.float32)
        result[:, 0] = 1.
        return result


def make_pdf(path, lines, *, blank_page=False):
    with pymupdf.open() as doc:
        page = doc.new_page()
        y = 60
        for text, size in lines:
            page.insert_text((50, y), text, fontsize=size)
            y += size + 12
        if blank_page:
            doc.new_page()
        doc.save(path)


def vector():
    result = np.zeros(DIMENSION, dtype=np.float32)
    result[0] = 1.
    return result


class ProductTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store_path = self.root / 'products'
        self.pdf = self.root / 'test.pdf'
        make_pdf(self.pdf, [('1. Ingredients', 18), ('Niacinamide : 4 %.', 11),
                            ('2. Utilisation', 18), ('Appliquer le soir.', 11)])
        self.encoder = FakeEncoder()

    def tearDown(self):
        self.tmp.cleanup()

    def ingest(self, user='alice', product='creme', **kwargs):
        return ingest_product_pdf(self.pdf, user, product, store_path=self.store_path,
                                  encoder=self.encoder, **kwargs)

    def expect_code(self, code, callback):
        with self.assertRaises(ProductError) as caught:
            callback()
        self.assertEqual(caught.exception.code, code)

    def test_text_pdf_provenance_and_ready_after_complete_index(self):
        entry = self.ingest(synthetic=True)
        self.assertEqual(entry['status'], 'ready')
        folder = self.store_path / 'documents' / entry['document_id']
        self.assertEqual((folder / 'source.pdf').read_bytes(), self.pdf.read_bytes())
        chunks = [json.loads(line) for line in (folder / 'chunks.jsonl').read_text().splitlines()]
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(c['user_id'] == 'alice' and c['product_id'] == 'creme' for c in chunks))
        self.assertTrue(all(c['source_sha256'] == entry['source_sha256'] for c in chunks))
        self.assertTrue(all(c['synthetic'] and c['page_start'] == c['page_end'] == 1 for c in chunks))
        with ProductStore(self.store_path) as store:
            store.verify(entry)
        self.assertFalse(list(folder.parent.glob('.ingest-*')))

    def test_idempotence_skips_encoder_and_has_same_number_of_points(self):
        first = self.ingest()
        with patch.object(self.encoder, 'encode', side_effect=AssertionError('Ne pas recalculer')):
            second = self.ingest()
        self.assertEqual(first['document_id'], second['document_id'])
        self.assertTrue(second['reused'])
        with ProductStore(self.store_path) as store:
            self.assertEqual(store.client.count(COLLECTION, exact=True).count, first['chunks'])

    def test_private_queries_exclude_other_owner_and_other_product(self):
        a = self.ingest()
        b = self.ingest('bob')
        c = self.ingest(product='serum')
        self.assertEqual(len({a['document_id'], b['document_id'], c['document_id']}), 3)
        with ProductStore(self.store_path) as store:
            hits, _ = search_scoped(store, vector(), 'Composition ?', 'alice', 'creme', scope='product')
        self.assertTrue(hits)
        self.assertTrue(all(h['payload']['user_id'] == 'alice' and
                            h['payload']['product_id'] == 'creme' and
                            h['payload']['document_id'] == a['document_id'] for h in hits))

    def test_unknown_user_fails_even_when_another_user_has_same_product_id(self):
        self.ingest()
        with ProductStore(self.store_path) as store:
            self.expect_code('product_not_ready', lambda: search_scoped(
                store, vector(), 'Q', 'bob', 'creme', scope='product'))

    def test_other_users_document_id_cannot_bypass_owner_filter(self):
        self.ingest()
        b = self.ingest('bob')
        with ProductStore(self.store_path) as store:
            self.expect_code('document_changed', lambda: search_scoped(store, vector(), 'Q',
                'alice', 'creme', document_id=b['document_id'], scope='product'))

    def test_new_version_is_active_old_version_stays_archived_and_is_not_searched(self):
        old = self.ingest(title='Version une')
        new = self.ingest(title='Version deux')
        self.assertNotEqual(old['document_id'], new['document_id'])
        self.assertTrue((self.store_path / 'documents' / old['document_id']).exists())
        with ProductStore(self.store_path) as store:
            hits, _ = search_scoped(store, vector(), 'Q', 'alice', 'creme', scope='product')
            self.assertTrue(all(h['payload']['document_id'] == new['document_id'] for h in hits))
            self.expect_code('document_changed', lambda: store.current('alice', 'creme', old['document_id']))

    def test_failed_embeddings_do_not_publish_new_version(self):
        old = self.ingest()
        with patch.object(self.encoder, 'encode', side_effect=RuntimeError('indisponible')):
            with self.assertRaises(RuntimeError):
                self.ingest(title='Nouvelle version')
        with ProductStore(self.store_path) as store:
            self.assertEqual(store.current('alice', 'creme')['document_id'], old['document_id'])
            store.verify(old)

    def test_failed_activation_keeps_old_version_and_retry_finishes(self):
        old = self.ingest()
        with patch.object(ProductStore, 'activate', side_effect=OSError('disque indisponible')):
            with self.assertRaises(OSError):
                self.ingest(title='Version suivante')
        with ProductStore(self.store_path) as store:
            self.assertEqual(store.current('alice', 'creme')['document_id'], old['document_id'])
        new = self.ingest(title='Version suivante')
        with ProductStore(self.store_path) as store:
            self.assertEqual(store.current('alice', 'creme')['document_id'], new['document_id'])
            store.verify(new)

    def test_failed_first_indexing_does_not_make_a_product_ready(self):
        with patch('qdrant_client.QdrantClient.upsert', side_effect=RuntimeError('index indisponible')):
            with self.assertRaises(RuntimeError):
                self.ingest()
        with ProductStore(self.store_path) as store:
            self.expect_code('product_not_ready', lambda: store.current('alice', 'creme'))
        self.assertEqual(list((self.store_path / 'documents').glob('.ingest-*')), [])

    def test_blank_or_scanned_page_is_refused_not_silently_omitted(self):
        blank = self.root / 'blank.pdf'
        make_pdf(blank, [('Contenu page un', 11)], blank_page=True)
        self.expect_code('text_required', lambda: ingest_product_pdf(blank, 'alice', 'creme',
            encoder=self.encoder, store_path=self.store_path))

    def test_extension_header_and_file_size_are_checked(self):
        wrong = self.root / 'wrong.txt'
        wrong.write_bytes(self.pdf.read_bytes())
        self.expect_code('unsupported_file', lambda: read_pdf_bytes(wrong))
        wrong = self.root / 'wrong.pdf'
        wrong.write_bytes(b'pas un pdf')
        self.expect_code('invalid_pdf', lambda: read_pdf_bytes(wrong))
        with patch('product_rag.MAX_FILE_BYTES', 10):
            self.expect_code('file_too_large', lambda: read_pdf_bytes(self.pdf))

    def test_corrupt_pdf_password_and_page_limit_are_explicit_errors(self):
        self.expect_code('invalid_pdf', lambda: extract_pages(b'%PDF-1.7\ninvalid'))
        with patch('product_rag.MAX_PAGES', 0):
            self.expect_code('page_limit', lambda: extract_pages(self.pdf.read_bytes()))
        encrypted = self.root / 'protected.pdf'
        with pymupdf.open(self.pdf) as doc:
            doc.save(encrypted, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw='owner', user_pw='secret')
        self.expect_code('encrypted_pdf', lambda: extract_pages(encrypted.read_bytes()))

    def test_product_and_owner_identifiers_are_mandatory_and_not_paths(self):
        for user, product in [('', 'x'), ('../alice', 'x'), ('alice', ''), ('alice', '../x')]:
            with self.subTest(user=user, product=product), self.assertRaises(ProductError):
                owner_product_key(user, product)

    def test_percentages_and_table_numbers_are_not_numbered_headings(self):
        page = {'page': 1, 'text': 'Ingredients\nNiacinamide\n4 %\nUtilisation\nMatin et soir.',
                'heading_candidates': []}
        self.assertEqual(len(page_sections(page)), 1)
        self.assertIn('Niacinamide\n4 %', page['text'])

    def test_generic_chunking_covers_unicode_text_and_keeps_page_offsets(self):
        text = ('Éclat à 4 %, niacinamide — application matin et soir.\n\n' * 100).strip()
        pages = [{'page': 3, 'text': text, 'heading_candidates': []}]
        metadata = {'doc_id': 'prd_test', 'doc_title': 'Produit démo'}
        chunks = make_product_chunks(pages, WordTokenizer(), metadata, size=30, overlap=5)
        self.assertGreater(len(chunks), 5)
        coverage = bytearray(len(text))
        for chunk in chunks:
            a, b = chunk['char_start'], chunk['char_end']
            self.assertEqual(chunk['text'], text[a:b])
            self.assertEqual(chunk['page_start'], 3)
            self.assertLessEqual(chunk['token_count'], 30)
            coverage[a:b] = b'\1' * (b - a)
        self.assertTrue(all(coverage[i] or char.isspace() for i, char in enumerate(text)))

    def test_title_only_at_end_is_kept(self):
        page = {'page': 1, 'text': 'Corps du texte.\n2. Annexe', 'heading_candidates': []}
        regions = page_sections(page)
        self.assertEqual(regions[0][1], len(page['text']))

    def test_missing_points_fail_closed(self):
        entry = self.ingest()
        with ProductStore(self.store_path) as store:
            store.client.delete(COLLECTION, points_selector=models.FilterSelector(
                filter=store.private_filter('alice', 'creme', entry['document_id'])), wait=True)
            self.expect_code('incomplete_store', lambda: search_scoped(
                store, vector(), 'Q', 'alice', 'creme', scope='product'))
        repaired = self.ingest()
        self.assertFalse(repaired['reused'])
        self.assertEqual(repaired['document_id'], entry['document_id'])
        with ProductStore(self.store_path) as store:
            store.verify(repaired)

    def test_common_allowlist_excludes_lumera_and_any_other_product(self):
        self.ingest()
        common = QdrantClient(':memory:')
        try:
            common.create_collection('adlp_sections', vectors_config=models.VectorParams(
                size=DIMENSION, distance=models.Distance.COSINE))
            points = []
            for doc_id in [*COMMON_DOCS, 'lumera', 'other_private_product']:
                points.append(models.PointStruct(id=str(uuid.uuid4()), vector=vector().tolist(),
                    payload={'chunk_id': 'common_' + doc_id, 'doc_id': doc_id, 'text': 'Règle documentaire.',
                             'source': doc_id + '.pdf', 'page_start': 1, 'page_end': 1}))
            common.upsert('adlp_sections', points=points)
            with ProductStore(self.store_path) as store:
                hits, _ = search_scoped(store, vector(), 'Q', 'alice', 'creme',
                    common_client=common, scope='both', top_k=8)
                ids = {h['payload']['doc_id'] for h in hits if h['origin'] == 'common'}
                self.assertEqual(ids, set(COMMON_DOCS))
                self.assertTrue(any(h['origin'] == 'product' for h in hits))
                hits, _ = search_scoped(store, vector(), 'Q', 'alice', 'creme',
                    common_client=common, scope='common', top_k=8)
                self.assertTrue(all(h['origin'] == 'common' for h in hits))
        finally:
            common.close()

    def test_merged_candidates_are_reranked_once_and_version_is_reported(self):
        entry = self.ingest()
        class ReverseReranker:
            calls = 0
            def score_pairs(self, pairs):
                self.calls += 1
                return list(range(len(pairs)))
        reranker = ReverseReranker()
        with ProductStore(self.store_path) as store:
            hits, details = search_scoped(store, vector(), 'Q', 'alice', 'creme',
                scope='product', reranker=reranker, candidate_k=20, top_k=1)
        self.assertEqual(reranker.calls, 1)
        self.assertEqual(hits[0]['retrieval_rank'], entry['chunks'])
        self.assertEqual(details['document_id'], entry['document_id'])
        self.assertEqual(len(details['candidate_ranking']), entry['chunks'])
        self.assertEqual(sum(r['selected'] for r in details['candidate_ranking']), 1)

    def test_invalid_context_is_rejected_before_model_loading(self):
        with patch('rag_common.load_encoder') as load:
            self.expect_code('product_not_ready', lambda: get_product_sources(
                'Q', 'unknown', 'x', store_path=self.store_path, scope='product'))
        load.assert_not_called()

    def test_product_sources_keep_provenance_and_use_pinned_active_version(self):
        entry = self.ingest()
        class FakeReranker:
            def score_pairs(self, pairs):
                return [float(i) for i in range(len(pairs))]
        with patch('rag_common.load_encoder', return_value=self.encoder), \
             patch('rerank_rag.load_reranker', return_value=FakeReranker()), \
             patch('rag_common.open_index') as common:
            sources, details = get_product_sources('Composition ?', 'alice', 'creme',
                document_id=entry['document_id'], scope='product', store_path=self.store_path)
        common.assert_not_called()
        self.assertEqual(details['document_id'], entry['document_id'])
        self.assertEqual([s['source_id'] for s in sources], [f'S{i}' for i in range(1, len(sources) + 1)])
        self.assertTrue(all(s['origin'] == 'product' and s['document_id'] == entry['document_id']
                            and s['source_sha256'] == entry['source_sha256'] for s in sources))
        self.assertTrue(all('user_id' not in s for s in sources))

    def test_scope_violation_in_backend_response_is_refused_before_reranking(self):
        self.ingest()
        class BadCommon:
            def query_points(self, *args, **kwargs):
                class Result:
                    points = [type('Point', (), {'score': .9, 'payload': {'doc_id': 'lumera'}})()]
                return Result()
        with ProductStore(self.store_path) as store:
            self.expect_code('scope_violation', lambda: search_scoped(store, vector(), 'Q',
                'alice', 'creme', scope='common', common_client=BadCommon()))

    def test_gemini_uses_existing_schema_and_validator_with_scoped_sources(self):
        sources = [{'source_id': 'S1', 'text': 'Niacinamide : 4 %.', 'retrieval_score': .9}]
        raw = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': json.dumps({
            'status': 'answered', 'answer': '4 % [S1].', 'citations': [
                {'source_id': 'S1', 'quote': 'Niacinamide : 4 %.'}], 'limitations': []})}]}}]}
        with patch('answer_rag.call_gemini', return_value=raw) as call, \
             patch('answer_rag.get_api_key', return_value='not-a-real-key'), \
             patch('answer_rag.save_run', return_value=Path('fake_trace.json')):
            result = answer_from_sources('Concentration ?', sources, {'document_id': 'prd_test'},
                user_id='alice', product_id='creme', model='test-model', prompt_version='rag_answer_v1_3')
        self.assertTrue(result['validation_passed'])
        contexts = json.loads(call.call_args.args[2]['contents'][0]['parts'][0]['text'])
        self.assertNotIn('retrieval_score', contexts['sources'][0])
        self.assertEqual(sources[0]['text'], contexts['sources'][0]['text'])


if __name__ == '__main__':
    unittest.main()
