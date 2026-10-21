"""Vérifier les annotations du jeu v2 contre le corpus, sans modèle ni réseau."""
import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KINDS = {'answerable', 'multi_source', 'conflict', 'unanswerable'}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'{path}, ligne {number} : JSON invalide.') from error
            if not isinstance(value, dict):
                raise ValueError(f'{path}, ligne {number} : objet JSON attendu.')
            rows.append(value)
    return rows


def normalized(text):
    """Même normalisation que les ancres du score de retrieval."""
    return re.sub(r'\s+', ' ', ''.join(
        c for c in unicodedata.normalize('NFKD', text.casefold())
        if not unicodedata.combining(c))).strip()


def quote_normalized(text):
    # Tolérer uniquement Unicode NFC et l'espacement ; pas des mots reformulés.
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty_strings(value):
    return (isinstance(value, list) and bool(value)
            and all(isinstance(x, str) and x.strip() for x in value))


def validate_question(question, units, split):
    qid = question.get('question_id', '?')
    prefix = f'{qid} : '
    for field in ('question_id', 'question', 'family', 'expected_answer_notes', 'origin'):
        require(isinstance(question.get(field), str) and bool(question[field].strip()),
                prefix + f'Champ {field} absent ou vide.')
    require(question.get('split') == split, prefix + 'Split incohérent.')
    require(question.get('kind') in KINDS, prefix + 'Type de question inconnu.')
    groups = question.get('evidence_groups')
    require(isinstance(groups, list), prefix + 'Liste de groupes attendue.')
    support = question.get('supporting_unit_ids', [])
    require(isinstance(support, list) and all(isinstance(x, str) and x in units for x in support),
            prefix + 'Unité de contexte inconnue.')
    if question['kind'] == 'unanswerable':
        require(not groups, prefix + 'Une question sans réponse ne doit pas avoir de preuve positive.')
        for field in ('missing_information', 'absence_note'):
            require(isinstance(question.get(field), str) and bool(question[field].strip()),
                    prefix + f'Justification {field} manquante.')
        # Cette justification est humaine : on ne peut pas prouver automatiquement
        # l'absence sémantique d'une réponse en recherchant un mot-clé.
        return 0
    require(bool(groups), prefix + 'Question répondable sans groupe de preuve.')
    if question['kind'] in ('multi_source', 'conflict'):
        require(len(groups) >= 2, prefix + 'Au moins deux groupes sont nécessaires.')
    count = 0
    for number, group in enumerate(groups, 1):
        label = prefix + f'groupe {number} : '
        require(isinstance(group, dict), label + 'Objet attendu.')
        ids = group.get('any_unit_ids')
        require(nonempty_strings(ids), label + 'Références vides ou invalides.')
        require(len(set(ids)) == len(ids), label + 'Référence dupliquée.')
        require(set(ids) <= units.keys(), label + 'Référence absente du corpus.')
        anchors = group.get('anchor_all')
        require(nonempty_strings(anchors), label + 'Ancres explicites manquantes.')
        quotes = group.get('evidence_quotes')
        require(isinstance(quotes, list) and bool(quotes), label + 'Extraits de preuve manquants.')
        by_unit = {uid: [] for uid in ids}
        for proof in quotes:
            require(isinstance(proof, dict), label + 'Extrait invalide.')
            uid, quote = proof.get('unit_id'), proof.get('quote')
            require(isinstance(uid, str) and uid in by_unit, label + 'Extrait attribué à une autre unité.')
            require(isinstance(quote, str) and bool(quote_normalized(quote)), label + 'Extrait vide.')
            require(quote_normalized(quote) in quote_normalized(units[uid]['text']),
                    label + f'Extrait absent de {uid}.')
            by_unit[uid].append(quote)
            count += 1
        for uid in ids:
            require(bool(by_unit[uid]), label + f'Aucun extrait pour {uid}.')
            focus = normalized('\n'.join(by_unit[uid]))
            require(all(normalized(a) in focus for a in anchors),
                    label + f'Une ancre manque dans les extraits de {uid}.')
    return count


def validate_dataset(dataset_dir, corpus_path):
    dataset_dir, corpus_path = Path(dataset_dir), Path(corpus_path)
    manifest_path = dataset_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest.get('dataset_id') == 'adlp_rag_eval_v2' and manifest.get('schema_version') == 1,
            'Version de jeu non reconnue.')
    require(file_sha(corpus_path) == manifest['corpus']['sha256'],
            'Le corpus diffère de la version annotée. Revoir les preuves avant d’évaluer ce nouveau corpus.')
    rows = read_jsonl(corpus_path)
    units = {row['unit_id']: row for row in rows}
    require(len(units) == len(rows) == manifest['corpus']['units'],
            'Nombre d’unités incohérent ou identifiants dupliqués.')
    sources = manifest['corpus']['sources']
    require(set(sources) == {u['doc_id'] for u in rows}, 'Sources documentaires incohérentes.')
    for u in rows:
        require(isinstance(u.get('text'), str) and bool(u['text'].strip()), 'Unité sans texte.')
        require(all(u.get(key) == value for key, value in sources[u['doc_id']].items()),
                f"Provenance incohérente pour {u['unit_id']}.")
    require(set(manifest['splits']) == {'dev', 'test'}, 'Splits dev et test attendus.')
    all_ids, all_texts, families = set(), set(), {}
    result = {'dataset_id': manifest['dataset_id'], 'manifest_sha256': file_sha(manifest_path),
              'corpus_sha256': file_sha(corpus_path), 'splits': {}, 'proof_quotes_checked': 0}
    split_ids = {}
    for split, expected_count in [('dev', 20), ('test', 10)]:
        entry = manifest['splits'][split]
        require(entry['file'] == f'questions_{split}.jsonl', 'Nom de fichier de split inattendu.')
        path = dataset_dir / entry['file']
        require(file_sha(path) == entry['sha256'],
                f'{split} : empreinte différente du manifeste. Modification à relire et à versionner.')
        questions = read_jsonl(path)
        require(len(questions) == entry['count'] == expected_count, f'{split} : nombre de questions incorrect.')
        require([q.get('question_id') for q in questions] == entry['question_ids'],
                f'{split} : liste des identifiants incohérente.')
        families[split], split_ids[split] = set(), set()
        for q in questions:
            result['proof_quotes_checked'] += validate_question(q, units, split)
            qid, text = q['question_id'], normalized(q['question'])
            require(qid not in all_ids, f'Identifiant dupliqué : {qid}.')
            require(text not in all_texts, f'Question dupliquée : {qid}.')
            all_ids.add(qid)
            all_texts.add(text)
            families[split].add(q['family'])
            split_ids[split].add(qid)
        kinds = dict(Counter(q['kind'] for q in questions))
        require(kinds == entry['kinds'], f'{split} : répartition des types incohérente.')
        require(kinds.get('unanswerable', 0) > 0, f'{split} : cas sans réponse manquants.')
        result['splits'][split] = {'file': str(path.resolve()), 'sha256': entry['sha256'],
                                   'count': len(questions), 'kinds': kinds}
    require(not families['dev'].intersection(families['test']), 'Famille partagée entre dev et test.')
    legacy = manifest.get('legacy_seen_ids', [])
    require(set(legacy) == {f'Q{i:02}' for i in range(1, 15)}, 'Liste des anciennes questions incohérente.')
    require(set(legacy) <= split_ids['dev'] and not set(legacy).intersection(split_ids['test']),
            'Une ancienne question déjà vue se trouve dans le test.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / 'data/evaluation/v2')
    parser.add_argument('--corpus', type=Path, default=ROOT / 'data/processed/corpus_units.jsonl')
    args = parser.parse_args()
    report = validate_dataset(args.dataset_dir, args.corpus)
    for split, entry in report['splits'].items():
        positives = entry['count'] - entry['kinds'].get('unanswerable', 0)
        print(f"{split} : {entry['count']} questions ; {positives} avec preuves ; "
              f"{entry['count'] - positives} sans réponse.")
    print(f"{report['proof_quotes_checked']} extraits vérifiés contre le corpus. Aucun modèle ni appel API.")
    print('Familles distinctes et anciens IDs hors test vérifiés ; cela ne prouve pas une indépendance sémantique complète.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise SystemExit(f'Validation impossible : {error}')
