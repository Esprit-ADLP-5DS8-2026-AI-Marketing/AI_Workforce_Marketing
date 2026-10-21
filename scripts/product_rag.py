"""PDF produits privés : préparation, version active et recherche filtrée.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import tempfile
import time
import unicodedata
import uuid
from datetime import datetime, timezone

from rag_common import ROOT, MODEL, REVISION, DIMENSION, COLLECTIONS
from chunk_corpus import count_tokens, token_windows
from rerank_rag import (rerank_hits, validate_search_options, RERANK_MODEL,
                        RERANK_REVISION, INPUT_VERSIONS)

STORE_PATH = ROOT / 'data/products'
COLLECTION = 'adlp_product_documents_v1'
PIPELINE_VERSION = 'product_pdf_sections_v1'
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PAGES = 100
MAX_TEXT_CHARS = 2_000_000
COMMON_DOCS = ('meta', 'arpp', 'methods')


class ProductError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def digest(value):
    return hashlib.sha256(value).hexdigest()


def identifier(value, label):
    # Identifiants techniques, pas les noms affichés du produit/utilisateur.
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value):
        raise ProductError('invalid_identifier', f'{label} invalide : 1 à 128 caractères alphanumériques, _, . ou -.')
    return value


def owner_product_key(user_id, product_id):
    identifier(user_id, 'user_id')
    identifier(product_id, 'product_id')
    return digest(json.dumps([user_id, product_id], separators=(',', ':')).encode())


def clean_text(text):
    # Nettoyage conservateur : ni suppression des accents ni réécriture du PDF.
    text = unicodedata.normalize('NFC', text).replace('\u00ad', '')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(line.rstrip() for line in text.splitlines())).strip()


def read_pdf_bytes(path):
    path = Path(path)
    if path.suffix.lower() != '.pdf':
        raise ProductError('unsupported_file', 'Seuls les fichiers PDF sont acceptés.')
    try:
        with path.open('rb') as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
    except OSError as error:
        raise ProductError('unreadable_file', 'Impossible de lire le PDF fourni.') from error
    if len(data) > MAX_FILE_BYTES:
        raise ProductError('file_too_large', 'Le PDF dépasse 20 Mio.')
    if b'%PDF-' not in data[:1024]:
        raise ProductError('invalid_pdf', 'Le fichier ne possède pas un en-tête PDF valide.')
    return data


def extract_pages(data):
    import pymupdf
    try:
        doc = pymupdf.open(stream=data, filetype='pdf')
    except Exception as error:
        raise ProductError('invalid_pdf', 'PDF illisible ou endommagé.') from error
    with doc:
        if doc.needs_pass:
            raise ProductError('encrypted_pdf', 'Le PDF est protégé par un mot de passe.')
        if not 1 <= len(doc) <= MAX_PAGES:
            raise ProductError('page_limit', 'Le PDF doit contenir entre 1 et 100 pages.')
        pages, total_chars = [], 0
        for number, page in enumerate(doc, 1):
            raw = page.get_text('text', sort=False)
            text = clean_text(raw)
            if not text:
                raise ProductError('text_required', f'Page {number} sans texte extractible : PDF scanné ou page vide. OCR non inclus.')
            total_chars += len(text)
            if total_chars > MAX_TEXT_CHARS:
                raise ProductError('text_limit', 'Le volume de texte extrait est trop important.')
            # Les grandes polices fournissent des candidats de titres. Aucune
            # décision sémantique ni liste adaptée aux quatre PDF historiques.
            lines = []
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines', []):
                    spans = [s for s in line.get('spans', []) if s.get('text', '').strip()]
                    if spans:
                        lines.append((''.join(s['text'] for s in spans).strip(),
                                      max(float(s['size']) for s in spans)))
            body_size = statistics.median([size for _, size in lines]) if lines else 0
            headings = {clean_text(value) for value, size in lines
                        if body_size and size >= body_size * 1.15 and 1 < len(value) <= 120
                        and re.search(r'[^\W\d_]', value)}
            pages.append({'page': number, 'raw_text': raw, 'text': text,
                          'heading_candidates': sorted(headings)})
        return pages


def page_sections(page):
    """Titres numérotés/grandes polices ; sinon une section par page.

    Les sections restent dans la page pour garantir la provenance. Les tableaux
    sont conservés tels qu'extraits ; leur reconstruction n'est pas un OCR.
    """
    text, headings = page['text'], set(page['heading_candidates'])
    starts, cursor = [], 0
    for line in text.splitlines(keepends=True):
        title = line.strip()
        numbered = re.match(r'^(?:\d{1,2}[.)]|\d{1,2}(?:\.\d{1,2})+\.?)\s+[^\W\d_]', title)
        if title and len(title) <= 120 and (title in headings or numbered):
            starts.append((cursor, title))
        cursor += len(line)
    if not starts or starts[0][0] > 0:
        starts.insert(0, (0, f"Page {page['page']}"))
    result, pending = [], None
    for i, (start, title) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        if text[start:end].strip() == title:
            pending = start if pending is None else pending
            continue
        result.append((pending if pending is not None else start, end, title))
        pending = None
    if pending is not None:
        if result:
            a, _, title = result[-1]
            result[-1] = (a, len(text), title)
        else:
            result.append((0, len(text), f"Page {page['page']}"))
    return result


def make_product_chunks(pages, tokenizer, metadata, size=300, overlap=50):
    chunks = []
    for page in pages:
        coverage = bytearray(len(page['text']))
        for start, end, section in page_sections(page):
            body = page['text'][start:end]
            for a, b in token_windows(tokenizer, body, size, overlap, prefer_paragraphs=True):
                a, b = start + a, start + b
                text = page['text'][a:b]
                embedding_text = f"passage: {metadata['doc_title']}\n{text}"
                embedding_size = count_tokens(tokenizer, embedding_text, special=True)
                if embedding_size > 512:
                    raise ProductError('token_limit', 'Titre et passage trop longs pour E5.')
                coverage[a:b] = b'\1' * (b - a)
                unit_id = f"{metadata['doc_id']}_p{page['page']:03d}"
                index = len(chunks) + 1
                chunks.append({**metadata,
                    'chunk_id': f"{metadata['doc_id']}_{index:05d}",
                    'unit_ids': [unit_id], 'strategy': 'product_sections_v1',
                    'page_start': page['page'], 'page_end': page['page'],
                    'section': section, 'text': text, 'embedding_text': embedding_text,
                    'token_count': count_tokens(tokenizer, text),
                    'embedding_token_count': embedding_size,
                    'char_start': a, 'char_end': b,
                    'tokenizer_model': MODEL, 'tokenizer_revision': REVISION,
                })
        if not all(coverage[i] or char.isspace() for i, char in enumerate(page['text'])):
            raise ProductError('incomplete_chunking', 'Une partie du texte serait perdue : ingestion interrompue.')
    if not chunks:
        raise ProductError('empty_document', 'Aucun passage exploitable.')
    return chunks


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


class ProductStore:
    """Un seul processus à la fois, verrou Qdrant local, catalogue atomique.

    Une seule documentation active par couple utilisateur/produit. Les anciennes
    versions restent archivées mais ne sont jamais recherchées par ce module.
    """
    def __init__(self, path=STORE_PATH):
        from qdrant_client import models
        from local_qdrant import LocalQdrantClient as QdrantClient
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.path / 'catalog.json'
        try:
            self.client = QdrantClient(path=str(self.path / 'qdrant'))
        except RuntimeError as error:
            raise ProductError('store_busy', 'Stockage produit occupé : exécuter les commandes successivement.') from error
        try:
            self.catalog = (json.loads(self.catalog_path.read_text(encoding='utf-8'))
                            if self.catalog_path.exists() else
                            {'schema_version': 1, 'model': MODEL, 'revision': REVISION,
                             'dimension': DIMENSION, 'active': {}})
            if (self.catalog.get('schema_version') != 1 or self.catalog.get('model') != MODEL
                    or self.catalog.get('revision') != REVISION or self.catalog.get('dimension') != DIMENSION):
                raise ProductError('incompatible_store', 'Le stockage produit utilise un autre modèle ou format.')
            if not self.client.collection_exists(COLLECTION):
                if self.catalog['active']:
                    raise ProductError('incomplete_store', 'Catalogue présent mais collection produit absente.')
                self.client.create_collection(COLLECTION, vectors_config=models.VectorParams(
                    size=DIMENSION, distance=models.Distance.COSINE))
            params = self.client.get_collection(COLLECTION).config.params.vectors
            if isinstance(params, dict) or params.size != DIMENSION or params.distance != models.Distance.COSINE:
                raise ProductError('incompatible_store', 'Configuration vectorielle produit incompatible.')
        except BaseException:
            self.client.close()
            raise

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def current(self, user_id, product_id, document_id=None):
        key = owner_product_key(user_id, product_id)
        entry = self.catalog['active'].get(key)
        if not entry or entry.get('status') != 'ready':
            raise ProductError('product_not_ready', 'Aucune documentation prête pour ce produit et cet utilisateur.')
        if entry['user_id'] != user_id or entry['product_id'] != product_id:
            raise ProductError('invalid_catalog', 'Identité incohérente dans le catalogue.')
        if document_id is not None and entry['document_id'] != document_id:
            raise ProductError('document_changed', 'La documentation active a changé : relancer la campagne avec sa nouvelle version.')
        return dict(entry)

    @staticmethod
    def private_filter(user_id, product_id, document_id):
        from qdrant_client import models
        owner_product_key(user_id, product_id)
        identifier(document_id, 'document_id')
        return models.Filter(must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in [('user_id', user_id), ('product_id', product_id),
                               ('document_id', document_id), ('doc_type', 'product')]])

    def verify(self, entry):
        query_filter = self.private_filter(entry['user_id'], entry['product_id'], entry['document_id'])
        count = self.client.count(COLLECTION, count_filter=query_filter, exact=True).count
        if count != entry['chunks'] or not (self.path / 'documents' / entry['document_id'] / 'report.json').exists():
            raise ProductError('incomplete_store', 'La documentation active est incomplète : réingérer le PDF.')

    def activate(self, entry):
        updated = {**self.catalog, 'active': dict(self.catalog['active'])}
        updated['active'][owner_product_key(entry['user_id'], entry['product_id'])] = entry
        fd, name = tempfile.mkstemp(prefix='.catalog-', dir=self.path)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(updated, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.catalog_path)
            self.catalog = updated
        finally:
            Path(name).unlink(missing_ok=True)


def ingest_product_pdf(path, user_id, product_id, *, title=None, synthetic=False,
                       store_path=STORE_PATH, encoder=None, offline=False, device='cpu'):
    """Retourner ready uniquement après fichiers + vecteurs + catalogue complet.

    Une erreur ne remplace jamais l'ancienne documentation prête. Une nouvelle
    version est identifiée par le PDF, le titre, le marquage synthétique et le pipeline.
    """
    from rag_common import load_encoder, encode, write_jsonl
    from qdrant_client import models
    owner_product_key(user_id, product_id)
    path = Path(path)
    title = unicodedata.normalize('NFC', title if title is not None else path.stem).strip()
    if not title or len(title) > 160 or '\n' in title or '\r' in title:
        raise ProductError('invalid_title', 'Le titre doit contenir de 1 à 160 caractères sur une ligne.')
    data = read_pdf_bytes(path)
    source_hash = digest(data)
    fingerprint = json.dumps([user_id, product_id, source_hash, title, bool(synthetic),
                              PIPELINE_VERSION, MODEL, REVISION], ensure_ascii=False, separators=(',', ':'))
    document_id = 'prd_' + digest(fingerprint.encode())
    with ProductStore(store_path) as store:
        try:
            existing = store.current(user_id, product_id, document_id)
        except ProductError as error:
            if error.code not in ('product_not_ready', 'document_changed'):
                raise
        else:
            try:
                store.verify(existing)
            except ProductError as error:
                if error.code != 'incomplete_store':
                    raise
                # Une réingestion explicite permet de réparer cette version.
            else:
                return {**existing, 'reused': True}
    pages = extract_pages(data)
    if encoder is None:
        try:
            encoder = load_encoder(device, offline)
        except SystemExit as error:
            raise ProductError('embedding_model_unavailable', str(error)) from error
    metadata = {'doc_id': document_id, 'document_id': document_id,
                'user_id': user_id, 'product_id': product_id,
                'source': path.name, 'source_sha256': source_hash,
                'doc_title': title, 'doc_type': 'product', 'source_type': 'user_upload',
                'scope': 'private_product', 'synthetic': bool(synthetic),
                'pipeline_version': PIPELINE_VERSION}
    chunks = make_product_chunks(pages, encoder.tokenizer, metadata)
    vectors = encode(encoder, [c['embedding_text'] for c in chunks])
    entry = {'status': 'ready', 'user_id': user_id, 'product_id': product_id,
             'document_id': document_id, 'doc_id': document_id, 'title': title,
             'source': path.name, 'source_sha256': source_hash, 'synthetic': bool(synthetic),
             'pages': len(pages), 'chunks': len(chunks), 'pipeline_version': PIPELINE_VERSION,
             'model': MODEL, 'revision': REVISION, 'dimension': DIMENSION,
             'created_at': datetime.now(timezone.utc).isoformat()}
    with ProductStore(store_path) as store:
        docs = store.path / 'documents'
        docs.mkdir(exist_ok=True)
        destination = docs / document_id
        staging = Path(tempfile.mkdtemp(prefix='.ingest-', dir=docs))
        try:
            (staging / 'source.pdf').write_bytes(data)
            write_jsonl(staging / 'pages.jsonl', pages)
            write_jsonl(staging / 'chunks.jsonl', chunks)
            write_json(staging / 'report.json', entry)
            # Une ancienne ingestion interrompue du même hash peut avoir laissé
            # des points : les IDs déterministes rendent la relance idempotente.
            for start in range(0, len(chunks), 64):
                points = [models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, c['chunk_id'])),
                    vector=v.tolist(), payload=c)
                    for c, v in zip(chunks[start:start + 64], vectors[start:start + 64])]
                store.client.upsert(COLLECTION, points=points, wait=True)
            count = store.client.count(COLLECTION,
                count_filter=store.private_filter(user_id, product_id, document_id), exact=True).count
            if count != len(chunks):
                raise ProductError('incomplete_index', 'Nombre de vecteurs incohérent : documentation non activée.')
            if destination.exists():
                # Les lecteurs sont exclus par le verrou Qdrant. Une version
                # incomplète reste inaccessible et pourra être réparée à nouveau.
                shutil.rmtree(destination)
            os.replace(staging, destination)
            store.verify(entry)
            store.activate(entry)  # Seule publication : dernière opération atomique.
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return {**entry, 'reused': False}


def common_filter():
    from qdrant_client import models
    return models.Filter(must=[models.FieldCondition(key='doc_id', match=models.MatchAny(any=list(COMMON_DOCS)))])


def search_scoped(store, vector, question, user_id, product_id, *, document_id=None,
                  common_client=None, scope='both', top_k=8, candidate_k=20, reranker=None):
    """Top dense global parmi privé/commun, puis reranking du même pool.

    On prend jusqu'à candidate_k dans chaque collection, fusionne par cosine E5
    et garde candidate_k au total. Cela ne garantit pas un quota par document.
    """
    validate_search_options(top_k, candidate_k, True)
    if scope not in ('product', 'common', 'both'):
        raise ProductError('invalid_scope', 'scope doit valoir product, common ou both.')
    if not isinstance(question, str) or not question.strip():
        raise ProductError('invalid_question', 'Question vide.')
    entry = store.current(user_id, product_id, document_id)
    store.verify(entry)
    if scope != 'product' and common_client is None:
        raise ProductError('missing_common_index', 'L’index commun est nécessaire pour ce périmètre.')
    started = time.perf_counter()
    pool = []
    if scope != 'common':
        points = store.client.query_points(COLLECTION, query=vector.tolist(),
            query_filter=store.private_filter(user_id, product_id, entry['document_id']),
            limit=candidate_k, with_payload=True).points
        pool.extend({'score': float(p.score), 'payload': p.payload, 'origin': 'product'} for p in points)
    if scope != 'product':
        points = common_client.query_points(COLLECTIONS['sections'], query=vector.tolist(),
            query_filter=common_filter(), limit=candidate_k, with_payload=True).points
        pool.extend({'score': float(p.score), 'payload': p.payload, 'origin': 'common'} for p in points)
    # Défense complémentaire : vérifier les résultats avant tout passage au LLM.
    for hit in pool:
        payload = hit['payload']
        if hit['origin'] == 'product':
            allowed = (payload.get('user_id') == user_id and payload.get('product_id') == product_id
                       and payload.get('document_id') == entry['document_id'] and payload.get('doc_type') == 'product')
        else:
            allowed = payload.get('doc_id') in COMMON_DOCS
        if not allowed:
            raise ProductError('scope_violation', 'Résultat hors périmètre : recherche interrompue.')
        if not math.isfinite(hit['score']):
            raise ProductError('invalid_score', 'Score de recherche invalide.')
    pool.sort(key=lambda h: (-h['score'], h['payload']['chunk_id']))
    candidates = [dict(h, rank=i) for i, h in enumerate(pool[:candidate_k], 1)]
    retrieval_ms = (time.perf_counter() - started) * 1000
    rerank_started = time.perf_counter()
    ranked = (rerank_hits(question, candidates, reranker, max(1, len(candidates)))
              if reranker is not None else candidates)
    hits = ranked[:top_k]
    details = {'mode': 'rerank' if reranker is not None else 'dense', 'scope': scope,
               'document_id': entry['document_id'], 'candidate_k': candidate_k,
               'candidates_returned': len(candidates), 'top_k': top_k, 'final_passages': len(hits),
               'retrieval_ms': retrieval_ms,
               'rerank_ms': (time.perf_counter() - rerank_started) * 1000 if reranker is not None else 0.,
               'candidate_policy': 'merge_cosine_then_top_k',
               'candidate_ranking': [
                   {'chunk_id': h['payload']['chunk_id'], 'origin': h['origin'],
                    'retrieval_rank': h.get('retrieval_rank', h['rank']), 'rank': h['rank'],
                    'retrieval_score': h['score'], 'rerank_score': h.get('rerank_score'),
                    'selected': h['rank'] <= top_k} for h in ranked],
               'reranker': ({'model': RERANK_MODEL, 'revision': RERANK_REVISION,
                             'input_version': INPUT_VERSIONS['metadata']} if reranker is not None else None)}
    return hits, details


def get_product_sources(question, user_id, product_id, *, document_id=None, scope='both',
                        top_k=8, candidate_k=20, offline=False, device='cpu',
                        use_reranker=True, store_path=STORE_PATH):
    from rag_common import load_encoder, encode, open_index
    from rerank_rag import load_reranker
    owner_product_key(user_id, product_id)
    validate_search_options(top_k, candidate_k, True)
    if scope not in ('product', 'common', 'both'):
        raise ProductError('invalid_scope', 'scope doit valoir product, common ou both.')
    if not isinstance(question, str) or not question.strip():
        raise ProductError('invalid_question', 'Question vide.')
    # Valider le contexte avant les chargements de modèles.
    with ProductStore(store_path) as store:
        entry = store.current(user_id, product_id, document_id)
        store.verify(entry)
        document_id = entry['document_id']
    try:
        encoder = load_encoder(device, offline)
    except SystemExit as error:
        raise ProductError('embedding_model_unavailable', str(error)) from error
    vector = encode(encoder, ['query: ' + question])[0]
    reranker = load_reranker(device, offline) if use_reranker else None
    common_client = None
    try:
        if scope != 'product':
            common_client, _ = open_index()
        with ProductStore(store_path) as store:
            hits, details = search_scoped(store, vector, question, user_id, product_id,
                document_id=document_id, common_client=common_client, scope=scope,
                top_k=top_k, candidate_k=candidate_k, reranker=reranker)
    finally:
        if common_client is not None:
            common_client.close()
    sources = []
    for index, hit in enumerate(hits, 1):
        p = hit['payload']
        # Les clés techniques d'identité ne sont pas des instructions pour Gemini.
        source = {key: p.get(key) for key in ('chunk_id', 'doc_id', 'document_id', 'product_id',
                  'doc_title', 'source', 'source_sha256', 'page_start', 'page_end',
                  'doc_type', 'scope', 'synthetic', 'section', 'text')}
        source.update(source_id=f'S{index}', retrieval_score=hit['score'], origin=hit['origin'])
        if 'rerank_score' in hit:
            source.update(rerank_score=hit['rerank_score'], retrieval_rank=hit['retrieval_rank'])
        sources.append(source)
    return sources, details
