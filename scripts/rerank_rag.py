"""Recherche E5 puis reclassement optionnel par cross-encoder multilingue.
"""
import math
import re
import time

RERANK_MODEL = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
RERANK_REVISION = '1427fd652930e4ba29e8149678df786c240d8825'
MAX_PAIR_TOKENS = 512
MAX_CANDIDATES = 100
DEFAULT_CANDIDATES = 20
DEFAULT_BATCH_SIZE = 8
INPUT_VERSIONS = {'legacy': 'legacy_v1', 'metadata': 'title_section_v1'}


def validate_input_mode(input_mode):
    if not isinstance(input_mode, str) or input_mode not in INPUT_VERSIONS:
        raise ValueError('Entrée du reranker inconnue : choisir legacy ou metadata.')


def document_title(payload):
    """Réutiliser les métadonnées déjà indexées, sans table liée aux questions."""
    explicit = payload.get('doc_title')
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    # Les index existants ont le titre dans embedding_text, mais pas doc_title.
    # Extraire uniquement l'en-tête, jamais le corps ni le préfixe E5.
    embedding_text = payload.get('embedding_text')
    if isinstance(embedding_text, str) and '\n' in embedding_text:
        header = embedding_text.split('\n', 1)[0].strip()
        if header.startswith('passage: '):
            title = header[len('passage: '):].strip()
            if title:
                return title
    source = payload.get('source')
    return source.strip() if isinstance(source, str) else ''


def build_rerank_passage(payload, input_mode='metadata'):
    """Construire l'entrée de classement sans modifier le texte de preuve."""
    validate_input_mode(input_mode)
    text = payload['text']
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Passage candidat vide.')
    if input_mode == 'legacy':
        title = payload.get('doc_title') or ''
        return f'{title}\n{text}' if title else text
    title = document_title(payload)
    section = payload.get('section')
    section = section.strip() if isinstance(section, str) else ''
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), '')
    key = lambda value: re.sub(r'\s+', ' ', value).strip().casefold()
    headers, seen = [], {key(first_line)}
    for value in (title, section):
        if value and key(value) not in seen:
            headers.append(value)
            seen.add(key(value))
    # La section n'est pas répétée si elle est déjà le premier titre du corps.
    # Le contrôle de 512 tokens est appliqué ensuite à la paire complète.
    return '\n'.join([*headers, text])


def validate_search_options(top_k, candidate_k=DEFAULT_CANDIDATES, use_reranker=False):
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError('top-k doit être un entier positif.')
    if use_reranker and (
        isinstance(candidate_k, bool) or not isinstance(candidate_k, int)
        or not top_k <= candidate_k <= MAX_CANDIDATES
    ):
        raise ValueError(f'candidate-k doit être compris entre top-k et {MAX_CANDIDATES}.')


class MultilingualReranker:
    """Inférence locale Transformers, sans entraînement ni tronquage silencieux."""

    def __init__(self, device='cpu', offline=False, batch_size=DEFAULT_BATCH_SIZE):
        if device not in ('cpu', 'mps', 'cuda'):
            raise ValueError('Périphérique de reranking inconnu.')
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size doit être un entier positif.')
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            RERANK_MODEL, revision=RERANK_REVISION, use_fast=True,
            local_files_only=offline, trust_remote_code=False,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            RERANK_MODEL, revision=RERANK_REVISION, local_files_only=offline,
            trust_remote_code=False, use_safetensors=True,
        )
        if self.model.config.num_labels != 1:
            raise ValueError('Le reranker doit produire un score unique par paire.')
        self.device = device
        self.batch_size = batch_size
        self.model.to(device)
        self.model.eval()

    def score_pairs(self, pairs):
        if not pairs:
            return []
        import torch

        features = self.tokenizer(
            [p[0] for p in pairs], [p[1] for p in pairs],
            padding=False, truncation=False, add_special_tokens=True,
        )
        for index, token_ids in enumerate(features['input_ids']):
            if len(token_ids) > MAX_PAIR_TOKENS:
                raise ValueError(
                    f'Paire question/passage {index + 1} : {len(token_ids)} tokens, '
                    f'limite {MAX_PAIR_TOKENS}. Raccourcir la question ou le passage ; '
                    'aucun tronquage silencieux effectué.'
                )
        scores = []
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                stop = min(start + self.batch_size, len(pairs))
                rows = [{key: values[i] for key, values in features.items()}
                        for i in range(start, stop)]
                batch = self.tokenizer.pad(rows, padding=True, return_tensors='pt')
                batch = {key: value.to(self.device) for key, value in batch.items()}
                logits = self.model(**batch).logits
                if tuple(logits.shape) != (stop - start, 1):
                    raise ValueError('Dimensions de sortie inattendues pour le reranker.')
                scores.extend(logits[:, 0].detach().cpu().tolist())
        if len(scores) != len(pairs) or not all(math.isfinite(s) for s in scores):
            raise ValueError('Scores du reranker invalides.')
        return scores


def load_reranker(device='cpu', offline=False):
    try:
        return MultilingualReranker(device=device, offline=offline)
    except (ImportError, OSError) as error:
        raise RuntimeError(
            'Chargement du reranker impossible. Vérifier torch, transformers et safetensors. '
            'Au premier lancement, autoriser le téléchargement en retirant --offline. '
            f'Détail : {error}'
        ) from error


def rerank_hits(question, hits, reranker, top_k=4, *, input_mode='metadata'):
    """Conserver les preuves et les scores E5, puis recalculer le rang final."""
    validate_search_options(top_k)
    validate_input_mode(input_mode)
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Question vide.')
    if not hits:
        return []
    pairs, seen = [], set()
    for hit in hits:
        payload = hit['payload']
        chunk_id = payload['chunk_id']
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen:
            raise ValueError('Identifiant de chunk absent ou dupliqué dans les candidats.')
        seen.add(chunk_id)
        text = payload['text']
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Passage candidat vide.')
        passage = build_rerank_passage(payload, input_mode)
        pairs.append((question, passage))  # Sans préfixes E5 query:/passage:.
    scores = reranker.score_pairs(pairs)
    if len(scores) != len(hits) or not all(math.isfinite(float(s)) for s in scores):
        raise ValueError('Scores du reranker absents, non finis ou en nombre incorrect.')
    # Le tri Python est stable : une égalité conserve l'ordre de la recherche E5.
    order = sorted(range(len(hits)), key=lambda i: float(scores[i]), reverse=True)
    return [dict(hits[index], rank=rank,
                 retrieval_rank=hits[index]['rank'], rerank_score=float(scores[index]))
            for rank, index in enumerate(order[:top_k], 1)]


def search_passages(client, strategy, vector, question, top_k=4, doc_id=None,
                    reranker=None, candidate_k=DEFAULT_CANDIDATES, rerank_input='metadata'):
    """Retourner les passages finaux, le pool initial et les mesures du parcours.

    La recherche actuelle reste le comportement par défaut. Les durées excluent
    le chargement des modèles et l'encodage E5 de la question.
    """
    from rag_common import retrieve

    enabled = reranker is not None
    validate_search_options(top_k, candidate_k, enabled)
    validate_input_mode(rerank_input)
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Question vide.')
    started = time.perf_counter()
    candidates, retrieval_ms = retrieve(
        client, strategy, vector, candidate_k if enabled else top_k, doc_id,
    )
    rerank_ms = 0.0
    candidate_ranking = []
    if enabled:
        rerank_started = time.perf_counter()
        # Une seule inférence sur tous les candidats ; le top-k est une tranche.
        ranked = rerank_hits(question, candidates, reranker,
                             max(1, len(candidates)), input_mode=rerank_input)
        hits = ranked[:top_k]
        rerank_ms = (time.perf_counter() - rerank_started) * 1000
        candidate_ranking = [
            {'chunk_id': h['payload']['chunk_id'], 'retrieval_rank': h['retrieval_rank'],
             'rerank_rank': h['rank'], 'retrieval_score': h['score'],
             'rerank_score': h['rerank_score'], 'selected': h['rank'] <= top_k,
             'document_title': document_title(h['payload']),
             'section': h['payload'].get('section')}
            for h in ranked
        ]
    else:
        hits = candidates
    details = {
        'mode': 'rerank' if enabled else 'dense',
        'candidate_k': candidate_k if enabled else top_k,
        'candidates_returned': len(candidates), 'top_k': top_k,
        'final_passages': len(hits), 'retrieval_ms': retrieval_ms,
        'rerank_ms': rerank_ms,
        'total_search_ms': (time.perf_counter() - started) * 1000,
        'candidate_ranking': candidate_ranking,
        'reranker': ({'model': RERANK_MODEL, 'revision': RERANK_REVISION,
                      'score_type': 'raw_logit', 'max_pair_tokens': MAX_PAIR_TOKENS,
                      'input_mode': rerank_input, 'input_version': INPUT_VERSIONS[rerank_input],
                      'device': getattr(reranker, 'device', 'unknown')}
                     if enabled else None),
    }
    return hits, candidates, details
