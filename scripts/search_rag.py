#!/usr/bin/env python3
"""Recherche de passages avec sources ; aucune réponse générée par un LLM."""
import argparse
import json
from rag_common import load_encoder, encode, open_index
from rerank_rag import load_reranker, search_passages, validate_search_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question')
    parser.add_argument('--strategy', choices=['fixed', 'sections'], default='sections')
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--doc-id', choices=['lumera', 'meta', 'arpp', 'methods'])
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--rerank', action='store_true', help='Reclasser un pool de candidats avec le cross-encoder')
    parser.add_argument('--candidate-k', type=int, default=20, help='Candidats avant reranking (défaut : 20)')
    args = parser.parse_args()
    validate_search_options(args.top_k, args.candidate_k, args.rerank)
    if not args.question.strip():
        raise SystemExit('Question vide.')
    encoder = load_encoder(args.device, args.offline)
    vector = encode(encoder, ['query: ' + args.question])[0]
    reranker = load_reranker(args.device, args.offline) if args.rerank else None
    client, _ = open_index()
    try:
        hits, _, details = search_passages(
            client, args.strategy, vector, args.question, args.top_k, args.doc_id,
            reranker=reranker, candidate_k=args.candidate_k,
        )
    finally:
        client.close()
    if args.json:
        print(json.dumps({'question': args.question, 'strategy': args.strategy,
                          'retrieval_ms': details['retrieval_ms'], 'retrieval': details, 'hits': hits}, ensure_ascii=False, indent=2))
    else:
        for hit in hits:
            payload = hit['payload']
            print(f"\n[{hit['rank']}] score={hit['score']:.4f} | {payload['source']} | p.{payload['page_start']}-{payload['page_end']}")
            if args.rerank:
                print(f"Rang E5 initial : {hit['retrieval_rank']} | logit reranker={hit['rerank_score']:.4f}")
            print(f"Section : {payload['section']}\n{payload['text']}")
        print(f"\nMode : {details['mode']} ; candidats : {details['candidates_returned']} ; passages finaux : {len(hits)}")
        print(f"Qdrant : {details['retrieval_ms']:.2f} ms ; reranking : {details['rerank_ms']:.2f} ms")
        print('Scores E5 et reranker : classements, pas probabilités de vérité.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error))
