#!/usr/bin/env python3
"""Recherche de passages avec sources ; aucune réponse générée par un LLM."""
import argparse
import json
from rag_common import load_encoder, encode, open_index, retrieve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question')
    parser.add_argument('--strategy', choices=['fixed', 'sections'], default='sections')
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--doc-id', choices=['lumera', 'meta', 'arpp', 'methods'])
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if not args.question.strip():
        raise SystemExit('Question vide.')
    encoder = load_encoder(args.device, args.offline)
    vector = encode(encoder, ['query: ' + args.question])[0]
    client, _ = open_index()
    try:
        hits, latency = retrieve(client, args.strategy, vector, args.top_k, args.doc_id)
    finally:
        client.close()
    if args.json:
        print(json.dumps({'question': args.question, 'strategy': args.strategy,
                          'retrieval_ms': latency, 'hits': hits}, ensure_ascii=False, indent=2))
    else:
        for hit in hits:
            payload = hit['payload']
            print(f"\n[{hit['rank']}] score={hit['score']:.4f} | {payload['source']} | p.{payload['page_start']}-{payload['page_end']}")
            print(f"Section : {payload['section']}\n{payload['text']}")
        print(f'\nRecherche : {latency:.2f} ms. Similarité = proximité du texte, pas confiance dans une réponse.')


if __name__ == '__main__':
    main()
