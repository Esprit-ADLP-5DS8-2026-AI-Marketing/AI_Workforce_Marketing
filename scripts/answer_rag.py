#!/usr/bin/env python3
"""Recherche E5/Qdrant existante, puis réponse Gemini avec citations contrôlées."""
import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import random
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = 'rag_answer_v1_3'
PROMPT_VERSIONS = ('rag_answer_v1_1', 'rag_answer_v1_2', 'rag_answer_v1_3')
PROMPT = ROOT / f'prompts/{PROMPT_VERSION}.txt'
DEFAULT_MODEL = 'gemini-3.7-flash'
SCHEMA = {
    'type': 'OBJECT',
    'properties': {
        'status': {'type': 'STRING', 'enum': ['answered', 'partial', 'insufficient']},
        'answer': {'type': 'STRING'},
        'citations': {'type': 'ARRAY', 'items': {
            'type': 'OBJECT', 'properties': {
                'source_id': {'type': 'STRING'}, 'quote': {'type': 'STRING'}},
            'required': ['source_id', 'quote']}},
        'limitations': {'type': 'ARRAY', 'items': {'type': 'STRING'}},
    },
    'required': ['status', 'answer', 'citations', 'limitations'],
}


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def validate_answer(result, sources):
    """Valider la structure, les identifiants et les extraits, pas la vérité des faits."""
    fields = {'status', 'answer', 'citations', 'limitations'}
    if not isinstance(result, dict) or set(result) != fields:
        raise ValueError('Structure JSON de réponse invalide.')
    if result['status'] not in ('answered', 'partial', 'insufficient'):
        raise ValueError('Statut invalide.')
    if not isinstance(result['answer'], str) or not result['answer'].strip():
        raise ValueError('Réponse vide ou invalide.')
    if not isinstance(result['limitations'], list) or not all(isinstance(x, str) for x in result['limitations']):
        raise ValueError('Liste des limites invalide.')
    if not isinstance(result['citations'], list):
        raise ValueError('Liste des citations invalide.')
    lookup = {s['source_id']: s for s in sources}
    cited = set()
    for item in result['citations']:
        if not isinstance(item, dict) or set(item) != {'source_id', 'quote'}:
            raise ValueError('Structure de citation invalide.')
        sid, quote = item['source_id'], item['quote']
        if not isinstance(sid, str) or sid not in lookup or sid in cited:
            raise ValueError('Source de citation inconnue ou dupliquée.')
        if not isinstance(quote, str) or not normalized(quote):
            raise ValueError('Extrait de citation vide.')
        if len(quote) > 800 or normalized(quote) not in normalized(lookup[sid]['text']):
            raise ValueError(f'Extrait absent du passage {sid} ou trop long.')
        cited.add(sid)
    citation_groups = re.findall(r'\[(S\d+(?:\s*,\s*S\d+)*)\]', result['answer'])
    inline_ids = {sid for group in citation_groups for sid in re.findall(r'S\d+', group)}
    if inline_ids != cited:
        raise ValueError('Les citations dans la réponse et la liste ne correspondent pas.')
    if result['status'] in ('answered', 'partial') and not cited:
        raise ValueError('Réponse factuelle sans citation : refusée.')
    return result


def get_sources(question, strategy, top_k, offline, *, use_reranker=False,
                candidate_k=20, doc_id=None, device='cpu', diagnostics=None):
    from rag_common import load_encoder, encode, open_index
    from rerank_rag import load_reranker, search_passages, validate_search_options
    validate_search_options(top_k, candidate_k, use_reranker)
    encoder = load_encoder(device, offline)
    vector = encode(encoder, ['query: ' + question])[0]
    reranker = load_reranker(device, offline) if use_reranker else None
    client, _ = open_index()
    try:
        hits, _, details = search_passages(
            client, strategy, vector, question, top_k, doc_id,
            reranker=reranker, candidate_k=candidate_k,
        )
    finally:
        client.close()  # Libérer la base locale avant l'appel réseau.
    sources = []
    for i, hit in enumerate(hits, 1):
        p = hit['payload']
        sources.append({
            'source_id': f'S{i}', 'chunk_id': p['chunk_id'],
            'source': p['source'], 'page_start': p['page_start'], 'page_end': p['page_end'],
            'doc_id': p['doc_id'], 'doc_type': p.get('doc_type'), 'scope': p.get('scope'),
            'synthetic': p.get('synthetic', False), 'text': p['text'],
            'retrieval_score': hit['score'],
        })
        if 'rerank_score' in hit:
            sources[-1].update(rerank_score=hit['rerank_score'], retrieval_rank=hit['retrieval_rank'])
    if diagnostics is not None:
        diagnostics.update(details)
    return sources, details['retrieval_ms']


def build_request(question, sources, prompt):
    # La similarité n'est pas envoyée comme un score de confiance au LLM.
    contexts = [{k: v for k, v in s.items() if k not in {'retrieval_score', 'rerank_score', 'retrieval_rank'}} for s in sources]
    return {
        'systemInstruction': {'parts': [{'text': prompt}]},
        'contents': [{'role': 'user', 'parts': [{'text': json.dumps(
            {'question': question, 'sources': contexts}, ensure_ascii=False)}]}],
        'generationConfig': {
            'responseMimeType': 'application/json', 'responseSchema': SCHEMA,
            'maxOutputTokens': 4096,
        },
    }


def parse_response(response):
    candidates = response.get('candidates', [])
    if not candidates:
        raise ValueError('Aucune réponse Gemini. Requête éventuellement bloquée.')
    candidate = candidates[0]
    if candidate.get('finishReason') != 'STOP':
        raise ValueError(f"Réponse incomplète ou bloquée : {candidate.get('finishReason')}.")
    text = ''.join(p.get('text', '') for p in candidate.get('content', {}).get('parts', [])
                   if not p.get('thought', False))
    return json.loads(text)


def safe_error_detail(error, api_key):
    """Afficher seulement statut/message Google, en masquant les clés éventuelles."""
    try:
        data = json.loads(error.read(16384).decode('utf-8', errors='replace'))
        detail = data.get('error', {})
        if not isinstance(detail, dict):
            return ''
        message = str(detail.get('message', ''))
        status = str(detail.get('status', ''))
        text = status + ': ' + message if status else message
        if api_key:
            text = text.replace(api_key, '[CLE_MASQUEE]')
        text = re.sub(r'AIza[0-9A-Za-z_-]+', '[CLE_MASQUEE]', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text[:1500]
    except (ValueError, AttributeError, OSError):
        return ''


def call_gemini(model, api_key, body):
    if not re.fullmatch(r'[A-Za-z0-9._-]+', model):
        raise ValueError('Identifiant de modèle invalide.')
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
    request = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'),
                                    headers={'Content-Type': 'application/json',
                                             'x-goog-api-key': api_key}, method='POST')
    for attempt in range(3):  # Au plus deux relances, uniquement pour les erreurs serveur.
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code in (500, 502, 503, 504) and attempt < 2:
                delay = 2 ** attempt + random.uniform(0, 0.5)
                print(f'Gemini HTTP {error.code} : nouvel essai dans {delay:.1f} s '
                      f'({attempt + 1}/2 relances).', file=sys.stderr)
                error.close()
                time.sleep(delay)
                continue
            explanations = {
                400: 'Requête ou configuration non acceptée.',
                401: 'Clé API non acceptée.', 403: 'Accès refusé : vérifier clé et projet.',
                404: 'Modèle non disponible : vérifier son identifiant dans AI Studio.',
                429: 'Quota atteint. Vérifier les limites du projet avant de relancer.',
                503: 'Service temporairement indisponible malgré les relances.',
            }
            detail = safe_error_detail(error, api_key)
            error.close()
            message = f'Gemini HTTP {error.code} : ' + explanations.get(error.code, 'Erreur du service.')
            if detail:
                message += '\nDiagnostic Google : ' + detail
            raise RuntimeError(message) from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError('Connexion Gemini impossible ou délai dépassé. Vérifier Internet et les certificats Python.') from None


def get_api_key():
    key = os.getenv('GEMINI_API_KEY', '').strip()
    if not key:
        if not sys.stdin.isatty():
            raise SystemExit('Définir GEMINI_API_KEY ou lancer dans un terminal interactif.')
        key = getpass.getpass('Clé Gemini (saisie masquée) : ').strip()
    if not key:
        raise SystemExit('Clé vide.')
    return key


def load_project_env(path=None):
    """Charger seulement RAG/.env, sans écraser les variables du terminal."""
    path = Path(path) if path is not None else ROOT / '.env'
    if not path.exists():
        return
    try:
        from dotenv import dotenv_values
    except ImportError:
        raise SystemExit('Pour lire RAG/.env : python -m pip install python-dotenv')
    values = dotenv_values(path, interpolate=False, encoding='utf-8')
    for name in ('GEMINI_API_KEY', 'GEMINI_MODEL'):
        value = values.get(name)
        if value is not None and name not in os.environ:
            os.environ[name] = value


def save_run(record):
    folder = ROOT / 'data/answers'
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = folder / f'answer_{stamp}_{uuid.uuid4().hex[:8]}.json'
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


def main():
    load_project_env()  # Avant de définir le modèle par défaut dans argparse.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question')
    parser.add_argument('--strategy', choices=['fixed', 'sections'], default='sections')
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--model', default=os.getenv('GEMINI_MODEL', DEFAULT_MODEL))
    parser.add_argument('--prompt-version', choices=PROMPT_VERSIONS, default=PROMPT_VERSION,
                        help='Version du prompt : v1.3 par défaut, versions précédentes pour comparaison')
    parser.add_argument('--offline', action='store_true', help='Cache E5 et reranker uniquement ; Gemini nécessite Internet')
    parser.add_argument('--rerank', action='store_true', help='Activer le reclassement multilingue des passages')
    parser.add_argument('--candidate-k', type=int, default=20)
    parser.add_argument('--doc-id', choices=['lumera', 'meta', 'arpp', 'methods'])
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--dry-run', action='store_true', help='Préparer les passages sans appeler Gemini')
    parser.add_argument('--api-test', action='store_true', help='Appel texte minimal, sans E5, Qdrant ni schéma JSON')
    args = parser.parse_args()
    if not args.question.strip() or not 1 <= args.top_k <= 8:
        raise SystemExit('Question vide ou top-k hors de 1 à 8.')
    from rerank_rag import validate_search_options
    validate_search_options(args.top_k, args.candidate_k, args.rerank)
    if args.api_test:
        if args.dry_run:
            raise SystemExit('Choisir --api-test ou --dry-run, pas les deux.')
        raw = call_gemini(args.model, get_api_key(), {
            'contents': [{'role': 'user', 'parts': [{'text': args.question}]}],
        })
        candidates = raw.get('candidates', [])
        if not candidates or candidates[0].get('finishReason') != 'STOP':
            raise SystemExit('Le test API ne renvoie pas de réponse terminée.')
        text = ''.join(p.get('text', '') for p in candidates[0].get('content', {}).get('parts', [])
                       if not p.get('thought', False))
        if not text.strip():
            raise SystemExit('Le test API renvoie une réponse vide.')
        print('Modèle demandé :', args.model)
        print('Réponse du test API :', text)
        return
    prompt_path = ROOT / 'prompts' / f'{args.prompt_version}.txt'
    if not prompt_path.exists():
        raise SystemExit(f'Prompt absent : {prompt_path}')
    prompt = prompt_path.read_text(encoding='utf-8')
    details = {}
    sources, latency = get_sources(
        args.question, args.strategy, args.top_k, args.offline,
        use_reranker=args.rerank, candidate_k=args.candidate_k, doc_id=args.doc_id,
        device=args.device, diagnostics=details,
    )
    if not sources:
        raise SystemExit('Aucun passage retrouvé : arrêt sans appel Gemini.')
    record = {'question': args.question, 'strategy': args.strategy, 'top_k': args.top_k,
              'model_requested': args.model, 'prompt_version': args.prompt_version,
              'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
              'sources': sources, 'retrieval_ms': latency, 'retrieval': details, 'live_api_called': False}
    if args.dry_run:
        print('Passages préparés. Aucun appel Gemini. Fichier :', save_run(record))
        for s in sources:
            print(f"[{s['source_id']}] {s['source']} p.{s['page_start']}-{s['page_end']}")
        return
    key = get_api_key()
    started = time.perf_counter()
    raw = call_gemini(args.model, key, build_request(args.question, sources, prompt))
    record.update(live_api_called=True, generation_seconds=time.perf_counter() - started,
                  model_version=raw.get('modelVersion'), usage=raw.get('usageMetadata'))
    try:
        result = validate_answer(parse_response(raw), sources)
    except (ValueError, TypeError, KeyError) as error:
        record.update(validation_passed=False, validation_error=str(error), raw_response=raw)
        path = save_run(record)
        raise SystemExit(f'Réponse rejetée par les contrôles locaux : {error}\nTrace à examiner : {path}')
    record.update(validation_passed=True, result=result)
    path = save_run(record)
    print('\nStatut annoncé par le LLM :', result['status'])
    print(result['answer'])
    lookup = {s['source_id']: s for s in sources}
    for c in result['citations']:
        s = lookup[c['source_id']]
        print(f"\n[{c['source_id']}] {s['source']} p.{s['page_start']}-{s['page_end']}")
        print('Extrait :', c['quote'])
    for limitation in result['limitations']:
        print('Limite :', limitation)
    print('\nTrace enregistrée :', path)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error))
