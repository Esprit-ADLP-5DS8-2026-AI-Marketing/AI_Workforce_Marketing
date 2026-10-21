"""Service interne partagé par les routes : modèles en mémoire, Qdrant coordonné."""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
import time

from answer_rag import (DEFAULT_MODEL, PROMPT_VERSION, ROOT, build_request,
                        call_gemini, parse_response, save_run, validate_answer)
from product_rag import (COMMON_DOCS, STORE_PATH, ProductError, ProductStore,
                         common_filter, ingest_product_pdf, search_scoped)
from rag_common import MODEL, REVISION, COLLECTIONS, encode, load_encoder, open_index
from rerank_rag import (RERANK_MODEL, RERANK_REVISION, INPUT_VERSIONS,
                        load_reranker, rerank_hits)

LOG = logging.getLogger('adlp.rag')


class ServiceError(Exception):
    def __init__(self, status, code, message, *, trace_id=None):
        super().__init__(message)
        self.status, self.code, self.message, self.trace_id = status, code, message, trace_id


@dataclass(frozen=True)
class Settings:
    service_token: str
    store_path: Path = STORE_PATH
    offline: bool = True
    device: str = 'cpu'
    gemini_model: str = DEFAULT_MODEL
    gemini_api_key: str = ''
    gate_timeout: float = 5.0

    def __post_init__(self):
        if not 32 <= len(self.service_token) <= 256 or not self.service_token.isascii():
            raise ValueError('RAG_SERVICE_TOKEN doit contenir de 32 à 256 caractères ASCII. Lancer configure_rag_api.py.')
        if self.device not in ('cpu', 'mps', 'cuda'):
            raise ValueError('RAG_DEVICE doit valoir cpu, mps ou cuda.')
        if not re.fullmatch(r'[A-Za-z0-9._-]+', self.gemini_model):
            raise ValueError('GEMINI_MODEL contient un identifiant invalide.')


def load_settings(path=None):
    from dotenv import dotenv_values
    path = Path(path) if path is not None else ROOT / '.env'
    values = dotenv_values(path, interpolate=False) if path.exists() else {}
    def get(name, default=''):
        return os.environ.get(name, values.get(name) or default)
    offline = get('RAG_OFFLINE', 'true').strip().lower()
    if offline not in ('true', 'false', '1', '0'):
        raise ValueError('RAG_OFFLINE doit valoir true ou false.')
    return Settings(service_token=get('RAG_SERVICE_TOKEN'), offline=offline in ('true', '1'),
                    device=get('RAG_DEVICE', 'cpu'), gemini_model=get('GEMINI_MODEL', DEFAULT_MODEL),
                    gemini_api_key=get('GEMINI_API_KEY'))


def sources_from_hits(hits):
    sources = []
    for i, hit in enumerate(hits, 1):
        p = hit['payload']
        source = {key: p.get(key) for key in ('chunk_id', 'doc_id', 'document_id', 'product_id',
                  'doc_title', 'source', 'source_sha256', 'page_start', 'page_end',
                  'doc_type', 'scope', 'synthetic', 'section', 'text')}
        source.update(source_id=f'S{i}', origin=hit['origin'], retrieval_score=hit['score'])
        if 'rerank_score' in hit:
            source.update(rerank_score=hit['rerank_score'], retrieval_rank=hit['retrieval_rank'])
        sources.append(source)
    return sources


class RagService:
    def __init__(self, settings):
        self.settings = settings
        self.encoder = self.reranker = None
        self.ready = False
        self.counts = {}
        self._gate = threading.Lock()
        self._process_lock = None

    def start(self):
        # portalocker est déjà une dépendance de qdrant-client. Le verrou évite
        # deux workers API indépendants sur le même stockage produit local.
        import portalocker
        self.settings.store_path.mkdir(parents=True, exist_ok=True)
        self._process_lock = portalocker.Lock(str(self.settings.store_path / '.api.lock'),
                                             mode='a', timeout=0)
        try:
            self._process_lock.acquire()
        except portalocker.exceptions.LockException:
            self._process_lock = None
            raise RuntimeError('Une API utilise déjà ce stockage. Lancer un seul worker.') from None
        try:
            client, manifest = open_index()
            try:
                self.counts = {key: entry['points'] for key, entry in manifest['collections'].items()}
            finally:
                client.close()
            with ProductStore(self.settings.store_path):
                pass
            self.encoder = load_encoder(self.settings.device, self.settings.offline)
            self.reranker = load_reranker(self.settings.device, self.settings.offline)
            self.ready = True
        except BaseException as error:
            self.close()
            if isinstance(error, SystemExit):
                raise RuntimeError(str(error)) from error
            raise

    def close(self):
        self.ready = False
        self.encoder = self.reranker = None
        if self._process_lock is not None:
            self._process_lock.release()
            self._process_lock = None

    @contextmanager
    def access(self):
        if not self.ready:
            raise ServiceError(503, 'service_not_ready', 'Le RAG n’est pas prêt.')
        if not self._gate.acquire(timeout=self.settings.gate_timeout):
            raise ServiceError(503, 'rag_busy', 'Le RAG traite une autre opération. Réessayer dans quelques secondes.')
        try:
            yield
        finally:
            self._gate.release()

    def health(self):
        return {'status': 'ready' if self.ready else 'starting',
                'api_version': '1.0.0', 'embedding_model': MODEL, 'embedding_revision': REVISION,
                'reranker_model': RERANK_MODEL, 'collections': self.counts,
                'gemini_configured': bool(self.settings.gemini_api_key),
                'gemini_model': self.settings.gemini_model}

    def ingest(self, path, user_id, product_id, *, title=None, synthetic=False):
        with self.access():
            return ingest_product_pdf(path, user_id, product_id, title=title, synthetic=synthetic,
                encoder=self.encoder, store_path=self.settings.store_path)

    def current(self, user_id, product_id):
        with self.access(), ProductStore(self.settings.store_path) as store:
            entry = store.current(user_id, product_id)
            store.verify(entry)
            return entry

    def list_products(self, user_id):
        with self.access(), ProductStore(self.settings.store_path) as store:
            entries = [dict(e) for e in store.catalog['active'].values() if e.get('user_id') == user_id]
            for entry in entries:
                store.verify(entry)
            return {'products': sorted(entries, key=lambda e: e['product_id'])}

    def retrieve(self, query, user_id):
        with self.access():
            expected_document = query.document_id
            # Vérification avant encodage, y compris si un contexte produit est
            # fourni avec une question sur les seules références communes.
            if query.product_id is not None:
                with ProductStore(self.settings.store_path) as store:
                    entry = store.current(user_id, query.product_id, expected_document)
                    store.verify(entry)
                    expected_document = entry['document_id']
            started = time.perf_counter()
            vector = encode(self.encoder, ['query: ' + query.question])[0]
            encoding_ms = (time.perf_counter() - started) * 1000
            common = None
            try:
                if query.scope != 'product':
                    try:
                        common, _ = open_index()
                    except SystemExit as error:
                        raise ServiceError(503, 'common_index_unavailable', 'Index commun absent ou modifié ; vérifier build_indexes.py.') from error
                    except RuntimeError as error:
                        raise ServiceError(503, 'common_index_busy', 'Index commun occupé par un autre processus.') from error
                    except (ValueError, OSError) as error:
                        raise ServiceError(503, 'common_index_unavailable', 'État de l’index commun invalide ; vérifier le serveur.') from error
                if query.scope in ('product', 'both'):
                    with ProductStore(self.settings.store_path) as store:
                        hits, details = search_scoped(store, vector, query.question, user_id, query.product_id,
                            document_id=expected_document, common_client=common, scope=query.scope,
                            top_k=query.top_k, candidate_k=query.candidate_k, reranker=self.reranker)
                else:
                    hits, details = self._search_common(common, query, vector)
                    details['document_id'] = expected_document
            finally:
                if common is not None:
                    common.close()
            details['encoding_ms'] = encoding_ms
            return {'question': query.question, 'user_id': user_id, 'product_id': query.product_id,
                    'document_id': expected_document, 'scope': query.scope,
                    'strategy': 'sections', 'sources': sources_from_hits(hits), 'retrieval': details}

    def _search_common(self, client, query, vector):
        from qdrant_client import models
        query_filter = common_filter()
        if query.doc_id is not None:
            query_filter = models.Filter(must=[models.FieldCondition(key='doc_id',
                                      match=models.MatchValue(value=query.doc_id))])
        started = time.perf_counter()
        points = client.query_points(COLLECTIONS['sections'], query=vector.tolist(),
            query_filter=query_filter, limit=query.candidate_k, with_payload=True).points
        candidates = [{'rank': i, 'score': float(p.score), 'payload': p.payload, 'origin': 'common'}
                      for i, p in enumerate(points, 1)]
        if any(h['payload'].get('doc_id') not in COMMON_DOCS for h in candidates):
            raise ProductError('scope_violation', 'Résultat hors périmètre commun : recherche interrompue.')
        retrieval_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        ranked = rerank_hits(query.question, candidates, self.reranker, max(1, len(candidates)))
        details = {'mode': 'rerank', 'scope': 'common', 'doc_id': query.doc_id,
                   'candidate_k': query.candidate_k, 'candidates_returned': len(candidates),
                   'top_k': query.top_k, 'final_passages': min(len(ranked), query.top_k),
                   'retrieval_ms': retrieval_ms, 'rerank_ms': (time.perf_counter() - started) * 1000,
                   'candidate_ranking': [{'chunk_id': h['payload']['chunk_id'],
                       'retrieval_rank': h['retrieval_rank'], 'rank': h['rank'],
                       'rerank_score': h['rerank_score'], 'selected': h['rank'] <= query.top_k} for h in ranked],
                   'reranker': {'model': RERANK_MODEL, 'revision': RERANK_REVISION,
                                'input_version': INPUT_VERSIONS['metadata']}}
        return ranked[:query.top_k], details

    def answer(self, query, user_id, request_id):
        if not self.settings.gemini_api_key:
            raise ServiceError(503, 'gemini_not_configured', 'GEMINI_API_KEY manque dans la configuration serveur.')
        retrieved = self.retrieve(query, user_id)
        sources = retrieved['sources']
        if not sources:
            raise ServiceError(404, 'no_sources', 'Aucun passage retrouvé ; aucun appel Gemini effectué.')
        prompt = (ROOT / 'prompts' / f'{PROMPT_VERSION}.txt').read_text(encoding='utf-8')
        record = {**retrieved, 'request_id': request_id,
                  'model_requested': self.settings.gemini_model, 'prompt_version': PROMPT_VERSION,
                  'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                  'created_at': datetime.now(timezone.utc).isoformat(), 'live_api_called': False}
        # Aucune connexion Qdrant ni verrou ML pendant l'appel réseau Gemini.
        started = time.perf_counter()
        try:
            raw = call_gemini(self.settings.gemini_model, self.settings.gemini_api_key,
                              build_request(query.question, sources, prompt))
        except RuntimeError as error:
            diagnostic = str(error).replace(self.settings.gemini_api_key, '[CLE_MASQUEE]')
            record.update(live_api_called=True, generation_seconds=time.perf_counter() - started,
                          error='gemini_request_failed', generation_error=diagnostic)
            trace = save_run(record)
            if 'HTTP 429' in str(error):
                raise ServiceError(429, 'gemini_quota_exceeded', 'Quota Gemini atteint.', trace_id=trace.name) from error
            if any(f'HTTP {code}' in str(error) for code in (400, 401, 403, 404)):
                raise ServiceError(502, 'gemini_configuration_error',
                                   'Vérifier la clé Gemini et le modèle configurés sur le serveur.', trace_id=trace.name) from error
            raise ServiceError(503, 'gemini_unavailable',
                               'Appel Gemini en échec. Vérifier le service et sa configuration serveur.', trace_id=trace.name) from error
        record.update(live_api_called=True, generation_seconds=time.perf_counter() - started,
                      model_version=raw.get('modelVersion'), usage=raw.get('usageMetadata'))
        try:
            result = validate_answer(parse_response(raw), sources)
        except (ValueError, TypeError, KeyError) as error:
            record.update(validation_passed=False, validation_error=str(error), raw_response=raw)
            trace = save_run(record)
            raise ServiceError(502, 'invalid_llm_answer',
                               'La réponse du modèle a échoué aux contrôles JSON ou de citations.', trace_id=trace.name) from error
        record.update(validation_passed=True, result=result)
        trace = save_run(record)
        return {**retrieved, **result, 'validation_passed': True, 'trace_id': trace.name,
                'model': raw.get('modelVersion', self.settings.gemini_model),
                'prompt_version': PROMPT_VERSION, 'usage': raw.get('usageMetadata'),
                'generation_ms': record['generation_seconds'] * 1000}
