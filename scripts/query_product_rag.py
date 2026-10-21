#!/usr/bin/env python3
"""Recherche dans un produit autorisé et les références communes ; Gemini optionnel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from product_rag import ProductError, STORE_PATH, get_product_sources


def answer_from_sources(question, sources, details, *, user_id, product_id,
                        model, prompt_version):
    from answer_rag import (ROOT, build_request, call_gemini, get_api_key, parse_response,
                            save_run, validate_answer)
    prompt = (ROOT / 'prompts' / f'{prompt_version}.txt').read_text(encoding='utf-8')
    record = {'question': question, 'user_id': user_id, 'product_id': product_id,
              'document_id': details['document_id'], 'sources': sources, 'retrieval': details,
              'model_requested': model, 'prompt_version': prompt_version,
              'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'live_api_called': False}
    if not sources:
        raise ProductError('no_sources', 'Aucun passage disponible : aucun appel Gemini effectué.')
    started = time.perf_counter()
    raw = call_gemini(model, get_api_key(), build_request(question, sources, prompt))
    record.update(live_api_called=True, generation_seconds=time.perf_counter() - started,
                  model_version=raw.get('modelVersion'), usage=raw.get('usageMetadata'))
    try:
        result = validate_answer(parse_response(raw), sources)
    except (ValueError, TypeError, KeyError) as error:
        record.update(validation_passed=False, validation_error=str(error), raw_response=raw)
        trace = save_run(record)
        raise ProductError('invalid_llm_answer', f'Réponse rejetée par les contrôles locaux. Trace : {trace}') from error
    record.update(validation_passed=True, result=result)
    trace = save_run(record)
    return {**result, 'validation_passed': True, 'trace_id': trace.name,
            'model': raw.get('modelVersion', model), 'prompt_version': prompt_version,
            'usage': raw.get('usageMetadata')}


def main():
    from answer_rag import DEFAULT_MODEL, PROMPT_VERSION, PROMPT_VERSIONS, load_project_env
    load_project_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question')
    parser.add_argument('--user-id', required=True)
    parser.add_argument('--product-id', required=True)
    parser.add_argument('--document-id', help='Version attendue ; refuse une ancienne version après remplacement')
    parser.add_argument('--scope', choices=['product', 'common', 'both'], default='both')
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--candidate-k', type=int, default=20)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--store', type=Path, default=STORE_PATH)
    parser.add_argument('--no-rerank', action='store_true', help='Diagnostic dense uniquement')
    parser.add_argument('--answer', action='store_true', help='Appeler Gemini ; sans cette option, recherche locale uniquement')
    parser.add_argument('--model', default=os.getenv('GEMINI_MODEL', DEFAULT_MODEL))
    parser.add_argument('--prompt-version', choices=PROMPT_VERSIONS, default=PROMPT_VERSION)
    args = parser.parse_args()
    if not args.question.strip() or not 1 <= args.top_k <= 8:
        raise ProductError('invalid_question', 'Question vide ou top-k hors de 1 à 8.')
    sources, details = get_product_sources(args.question, args.user_id, args.product_id,
        document_id=args.document_id, scope=args.scope, top_k=args.top_k,
        candidate_k=args.candidate_k, offline=args.offline, device=args.device,
        use_reranker=not args.no_rerank, store_path=args.store)
    result = {'question': args.question, 'user_id': args.user_id, 'product_id': args.product_id,
              'document_id': details['document_id'], 'sources': sources, 'retrieval': details}
    if args.answer:
        result.update(answer_from_sources(args.question, sources, details, user_id=args.user_id,
            product_id=args.product_id, model=args.model, prompt_version=args.prompt_version))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except ProductError as error:
        print(json.dumps({'status': 'error', 'code': error.code, 'message': str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
    except (RuntimeError, ValueError, OSError) as error:
        print(json.dumps({'status': 'error', 'code': 'query_failed', 'message': str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
