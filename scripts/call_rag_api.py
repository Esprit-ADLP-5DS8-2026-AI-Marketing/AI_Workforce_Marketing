#!/usr/bin/env python3
"""Client de test : charge le jeton de .env sans l'afficher dans le terminal."""
import argparse
import json
from pathlib import Path
import sys
import httpx
from rag_service import load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['health', 'products', 'current', 'ingest', 'retrieve', 'answer'])
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--user-id')
    parser.add_argument('--product-id')
    parser.add_argument('--document-id')
    parser.add_argument('--scope', choices=['product', 'common', 'both'], default='both')
    parser.add_argument('--doc-id', choices=['meta', 'arpp', 'methods'])
    parser.add_argument('--question')
    parser.add_argument('--file', type=Path)
    parser.add_argument('--title')
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    headers = {}
    if args.action != 'health':
        if not args.user_id:
            parser.error('--user-id est requis pour les commandes authentifiées.')
        settings = load_settings()
        headers = {'Authorization': 'Bearer ' + settings.service_token, 'X-User-ID': args.user_id}
    with httpx.Client(base_url=args.url.rstrip('/'), headers=headers, timeout=180., trust_env=False) as client:
        if args.action == 'health':
            response = client.get('/health')
        elif args.action == 'products':
            response = client.get('/products')
        elif args.action in ('current', 'ingest'):
            if not args.product_id:
                parser.error('--product-id est requis.')
            if args.action == 'current':
                response = client.get(f'/products/{args.product_id}/documents/current')
            else:
                if not args.file:
                    parser.error('--file est requis.')
                data = {'synthetic': str(args.synthetic).lower()}
                if args.title:
                    data['title'] = args.title
                with args.file.open('rb') as stream:
                    response = client.post(f'/products/{args.product_id}/documents', data=data,
                        files={'file': (args.file.name, stream, 'application/pdf')})
        else:
            if not args.question:
                parser.error('--question est requis.')
            payload = {'question': args.question, 'scope': args.scope, 'top_k': 8, 'candidate_k': 20}
            for field in ('product_id', 'document_id', 'doc_id'):
                if getattr(args, field) is not None:
                    payload[field] = getattr(args, field)
            response = client.post('/' + args.action, json=payload)
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except ValueError:
        print(f'Réponse HTTP {response.status_code} non JSON.', file=sys.stderr)
    if response.is_error:
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except httpx.RequestError:
        raise SystemExit('Connexion API impossible ou délai dépassé. Vérifier que run_rag_api.py est lancé.')
