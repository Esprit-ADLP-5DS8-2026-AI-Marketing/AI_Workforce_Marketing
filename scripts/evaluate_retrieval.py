#!/usr/bin/env python3
"""Comparer les deux collections sur les mêmes questions, sans génération LLM."""
import argparse
import csv
import json
import statistics
import time
import unicodedata
import re
from pathlib import Path
from rag_common import (ROOT, MODEL, REVISION, COLLECTIONS, read_jsonl, write_jsonl,
                        file_sha, load_encoder, encode, open_index, retrieve)

# Annotations simples pour éviter de compter une autre phrase de la même page
# comme preuve exacte. Elles sont à relire par l'équipe, comme les golds du JSONL.
# Q09 reste évaluée au niveau de ses fiches alternatives.
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
    parser.add_argument('--questions', type=Path, default=ROOT / 'data/evaluation/questions.jsonl')
    args = parser.parse_args()
    if args.top_k < 1:
        raise SystemExit('top-k doit être positif.')
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
    client, manifest = open_index()
    results, summaries = [], {}
    try:
        for strategy in COLLECTIONS:
            rows = []
            for question, vector in zip(questions, vectors):
                hits, latency = retrieve(client, strategy, vector, args.top_k)
                scores = score_question(question, hits)
                row = {'strategy': strategy, 'question_id': question['question_id'],
                       'question': question['question'], 'kind': question['kind'],
                       'expected_answer_notes': question.get('expected_answer_notes'),
                       'scores': scores, 'retrieval_ms': latency,
                       'context_body_tokens': sum(h['payload']['token_count'] for h in hits), 'hits': hits}
                rows.append(row)
            positive = [r for r in rows if r['scores']['hit'] is not None]
            if not positive:
                raise SystemExit('Le jeu ne contient aucune question avec preuve attendue.')
            summaries[strategy] = {
                'collection': manifest['collections'][strategy]['name'],
                'chunks': manifest['collections'][strategy]['points'],
                'questions_with_gold': len(positive), 'unanswerable_questions': len(rows) - len(positive),
                'unit_hit_rate': statistics.mean(r['scores']['hit'] for r in positive),
                'mean_group_recall': statistics.mean(r['scores']['group_recall'] for r in positive),
                'all_groups_rate': statistics.mean(r['scores']['all_groups'] for r in positive),
                'anchored_all_groups_rate': statistics.mean(r['scores']['anchored_all_groups'] for r in positive),
                'mrr': statistics.mean(r['scores']['reciprocal_rank'] for r in positive),
                'mean_retrieval_ms': statistics.mean(r['retrieval_ms'] for r in rows),
                'mean_context_body_tokens': statistics.mean(r['context_body_tokens'] for r in rows),
            }
            results.extend(rows)
            s = summaries[strategy]
            print(f"{strategy}: hit@{args.top_k}={s['unit_hit_rate']:.1%} ; tous groupes={s['all_groups_rate']:.1%} ; avec ancres={s['anchored_all_groups_rate']:.1%} ; MRR={s['mrr']:.3f}")
    finally:
        client.close()
    out = ROOT / 'data/evaluation/results'
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / 'retrieved_contexts.jsonl', results)
    report = {'model': MODEL, 'revision': REVISION, 'top_k': args.top_k,
              'questions_sha256': file_sha(args.questions), 'query_encoding_seconds': query_encoding_seconds,
              'strategies': summaries, 'generation_evaluated': False,
              'notes': ['Golds au niveau des unités/pages : proxy, pas garantie de bonne réponse.',
                        'Ancres lexicales préparées pour les IDs Q01-Q14 ; les relire avant de figer le jeu.',
                        'Les deux questions sans réponse sont exclues des métriques positives.',
                        'Le temps de recherche exclut chargement du modèle et encodage de la question.',
                        'Petit jeu pédagogique : inspection humaine des contextes nécessaire.']}
    (out / 'retrieval_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with (out / 'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['strategy', *next(iter(summaries.values())).keys()])
        writer.writeheader()
        for strategy, summary in summaries.items():
            writer.writerow({'strategy': strategy, **summary})
    print('Résultats dans data/evaluation/results. Inspecter retrieved_contexts.jsonl avant de choisir le découpage.')


if __name__ == '__main__':
    main()
