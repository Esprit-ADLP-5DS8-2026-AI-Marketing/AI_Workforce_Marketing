#!/usr/bin/env python3
"""Comparer les deux collections sur les mêmes questions, sans génération LLM."""
import argparse
import csv
import json
import statistics
import time
import unicodedata
import re
from datetime import datetime, timezone
from pathlib import Path
from rag_common import (ROOT, MODEL, REVISION, COLLECTIONS, read_jsonl, write_jsonl,
                        file_sha, load_encoder, encode, open_index)
from rerank_rag import load_reranker, search_passages, validate_search_options

# Ancres de secours pour l'ancien jeu. Le jeu v2 définit ses ancres dans le JSONL.
# Elles évitent certains faux positifs de page, sans prouver l'implication logique.
ANCHORS = {
    'Q01': [['niacinamide', '4 %']], 'Q02': [['34,90']],
    'Q03': [['normales', 'mixtes', 'deshydratees']],
    'Q04': [['#96C8B9', '#3C5F55', '#C9A86A', '#F8F4EC', '#28362F']],
    'Q05': [['matin et soir', 'peau propre']],
    'Q06': [['satisfaction', 'ne peut pas etre presente', 'efficacite']],
    'Q07': [['profondeur'], ['profondeur', 'ne peut pas']],
    'Q08': [['affirmant ou insinuant', 'attributs personnels']],
    'Q10': [['Attention', 'Interest', 'Desire', 'Action']], 'Q11': [['parfum']],
    'Q13': [['premiere semaine'], ['4 semaines']],
}


def normalized(text):
    return re.sub(r'\s+', ' ', ''.join(c for c in unicodedata.normalize('NFKD', text.casefold())
                                     if not unicodedata.combining(c))).strip()


def score_question(question, hits):
    groups = question['evidence_groups']
    if not groups:
        return {'unit_groups_found': [], 'anchored_groups_found': [], 'hit': None,
                'group_recall': None, 'all_groups': None, 'anchored_all_groups': None,
                'reciprocal_rank': None, 'requires_generation_abstention_test': True}
    unit_found, anchored_found, first_rank = [], [], None
    for i, group in enumerate(groups):
        ids = set(group['any_unit_ids'])
        relevant = [hit for hit in hits if ids.intersection(hit['payload'].get('unit_ids', []))]
        unit_found.append(bool(relevant))
        if relevant:
            rank = min(hit['rank'] for hit in relevant)
            first_rank = min(first_rank, rank) if first_rank else rank
        text = normalized('\n'.join(hit['payload']['text'] for hit in relevant))
        manual_anchors = group.get('anchor_all')
        defaults = ANCHORS.get(question['question_id'], [])
        anchors = manual_anchors if manual_anchors is not None else (defaults[i] if i < len(defaults) else [])
        anchored_found.append(bool(relevant) and all(normalized(anchor) in text for anchor in anchors))
    return {'unit_groups_found': unit_found, 'anchored_groups_found': anchored_found,
            'hit': any(unit_found), 'group_recall': sum(unit_found) / len(unit_found),
            'all_groups': all(unit_found), 'anchored_all_groups': all(anchored_found),
            'reciprocal_rank': 1 / first_rank if first_rank else 0,
            'requires_generation_abstention_test': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--offline', action='store_true')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--questions', type=Path, default=ROOT / 'data/evaluation/questions.jsonl')
    source.add_argument('--split', choices=['dev', 'test'],
                        help='Utiliser le jeu v2 validé et séparer les résultats dev/test')
    parser.add_argument('--output-dir', type=Path, help='Dossier de résultats explicite (optionnel)')
    parser.add_argument('--compare-reranking', action='store_true',
                        help='Comparer fixed/sections en dense puis avec reranking')
    parser.add_argument('--candidate-k', type=int, default=20)
    parser.add_argument('--rerank-input', choices=['metadata', 'legacy'], default='metadata',
                        help='Entrée du reranker : titre+section (défaut), ou ancien format pour comparaison')
    args = parser.parse_args()
    validate_search_options(args.top_k, args.candidate_k, args.compare_reranking)
    dataset_info = None
    if args.split:
        from validate_evaluation import validate_dataset
        dataset_info = validate_dataset(ROOT / 'data/evaluation/v2',
                                        ROOT / 'data/processed/corpus_units.jsonl')
        args.questions = Path(dataset_info['splits'][args.split]['file'])
        print(f'Jeu v2 : {args.split}. Validation des annotations réussie.', flush=True)
    questions = read_jsonl(args.questions)
    if not questions or len({q['question_id'] for q in questions}) != len(questions):
        raise SystemExit('Questions absentes ou IDs dupliqués.')
    units = {u['unit_id'] for u in read_jsonl(ROOT / 'data/processed/corpus_units.jsonl')}
    for q in questions:
        for group in q['evidence_groups']:
            if not group['any_unit_ids'] or not set(group['any_unit_ids']) <= units:
                raise SystemExit('Une référence de preuve est absente du corpus.')
    encoder = load_encoder(args.device, args.offline)
    started = time.perf_counter()
    # Même vecteur de question utilisé pour A et B. Aucun filtre pour cette comparaison.
    vectors = encode(encoder, ['query: ' + q['question'] for q in questions])
    query_encoding_seconds = time.perf_counter() - started
    reranker = None
    reranker_load_seconds = 0.0
    if args.compare_reranking:
        print('Chargement du reranker multilingue (poids supplémentaires au premier lancement).', flush=True)
        started = time.perf_counter()
        reranker = load_reranker(args.device, args.offline)
        reranker_load_seconds = time.perf_counter() - started
    client, manifest = open_index()
    results, summaries = [], {}
    try:
        for strategy in COLLECTIONS:
            modes = ['dense', 'rerank'] if args.compare_reranking else ['dense']
            for mode in modes:
                rows = []
                for question, vector in zip(questions, vectors):
                    hits, candidates, details = search_passages(
                        client, strategy, vector, question['question'], args.top_k,
                        reranker=reranker if mode == 'rerank' else None,
                        candidate_k=args.candidate_k,
                        rerank_input=args.rerank_input,
                    )
                    scores = score_question(question, hits)
                    pool_scores = score_question(question, candidates)
                    row = {'strategy': strategy, 'mode': mode,
                           'question_id': question['question_id'],
                           'question': question['question'], 'kind': question['kind'],
                           'split': question.get('split'), 'family': question.get('family'),
                           'evidence_groups': question['evidence_groups'],
                           'expected_answer_notes': question.get('expected_answer_notes'),
                           'scores': scores, 'candidate_scores': pool_scores,
                           'retrieval_ms': details['retrieval_ms'], 'retrieval': details,
                           'context_body_tokens': sum(h['payload']['token_count'] for h in hits),
                           'hits': hits, 'candidate_hits': candidates}
                    rows.append(row)
                positive = [r for r in rows if r['scores']['hit'] is not None]
                if not positive:
                    raise SystemExit('Le jeu ne contient aucune question avec preuve attendue.')
                key = f'{strategy}_{mode}' if args.compare_reranking else strategy
                summaries[key] = {
                    'collection': manifest['collections'][strategy]['name'],
                    'chunks': manifest['collections'][strategy]['points'], 'mode': mode,
                    'rerank_input': args.rerank_input if mode == 'rerank' else 'none',
                    'candidate_k': args.candidate_k if mode == 'rerank' else args.top_k,
                    'questions_with_gold': len(positive), 'unanswerable_questions': len(rows) - len(positive),
                    'unit_hit_rate': statistics.mean(r['scores']['hit'] for r in positive),
                    'mean_group_recall': statistics.mean(r['scores']['group_recall'] for r in positive),
                    'all_groups_rate': statistics.mean(r['scores']['all_groups'] for r in positive),
                    'anchored_all_groups_rate': statistics.mean(r['scores']['anchored_all_groups'] for r in positive),
                    'mrr': statistics.mean(r['scores']['reciprocal_rank'] for r in positive),
                    'candidate_unit_hit_rate': statistics.mean(r['candidate_scores']['hit'] for r in positive),
                    'candidate_all_groups_rate': statistics.mean(r['candidate_scores']['all_groups'] for r in positive),
                    'mean_retrieval_ms': statistics.mean(r['retrieval_ms'] for r in rows),
                    'mean_rerank_ms': statistics.mean(r['retrieval']['rerank_ms'] for r in rows),
                    'mean_total_search_ms': statistics.mean(r['retrieval']['total_search_ms'] for r in rows),
                    'mean_context_body_tokens': statistics.mean(r['context_body_tokens'] for r in rows),
                }
                results.extend(rows)
                summary = summaries[key]
                print(f"{key}: hit@{args.top_k}={summary['unit_hit_rate']:.1%} ; "
                      f"tous groupes={summary['all_groups_rate']:.1%} ; "
                      f"avec ancres={summary['anchored_all_groups_rate']:.1%} ; "
                      f"MRR={summary['mrr']:.3f} ; "
                      f"recherche+rerank={summary['mean_total_search_ms']:.1f} ms")
    finally:
        client.close()
    if args.output_dir:
        out = args.output_dir
    else:
        out = ROOT / 'data/evaluation/results'
        if args.split:
            out = out / 'v2' / args.split
        if args.compare_reranking:
            out = out / 'reranking'
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / 'retrieved_contexts.jsonl', results)
    report = {'model': MODEL, 'revision': REVISION, 'top_k': args.top_k,
              'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
              'split': args.split, 'dataset': dataset_info,
              'questions_file': str(args.questions.resolve()),
              'questions_sha256': file_sha(args.questions), 'query_encoding_seconds': query_encoding_seconds,
              'strategies': summaries, 'generation_evaluated': False,
              'compare_reranking': args.compare_reranking,
              'rerank_input': args.rerank_input if args.compare_reranking else None,
              'reranker_load_seconds': reranker_load_seconds,
              'reranker': next((r['retrieval']['reranker'] for r in results
                                if r['mode'] == 'rerank'), None),
              'notes': ['Golds au niveau des unités/pages : proxy, pas garantie de bonne réponse.',
                        'Jeu v2 : ancres explicites du JSONL ; jeu ancien : ancres de secours Q01-Q14.',
                        'Les questions sans réponse sont exclues des métriques positives : abstention non évaluée ici.',
                        'Les annotations v2 corrigent certaines alternatives : ne pas comparer directement aux anciens scores.',
                        'Les temps excluent le chargement des modèles et l’encodage E5 de la question.',
                        'Le premier appel du reranker peut inclure une initialisation supplémentaire.',
                        'Les scores du reranker sont des logits non calibrés, pas des probabilités.',
                        'candidate_ranking, dans chaque trace retrieval, conserve tous les rangs et scores après reranking.',
                        'Mode metadata : titre documentaire et section ajoutés pour le classement ; preuves inchangées.',
                        'Aucune optimisation automatique ni sélection de stratégie sur ces résultats.',
                        'Petit jeu pédagogique : inspection humaine des contextes nécessaire.']}
    (out / 'retrieval_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                             encoding='utf-8')
    with (out / 'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['strategy', *next(iter(summaries.values())).keys()])
        writer.writeheader()
        for strategy, summary in summaries.items():
            writer.writerow({'strategy': strategy, **summary})
    print(f'Résultats dans {out}. Inspecter retrieved_contexts.jsonl avant de choisir le parcours.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error))
