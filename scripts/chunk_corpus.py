#!/usr/bin/env python3
"""Créer deux corpus de chunks avec le même tokenizer et la même provenance."""
import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_LIMIT = 512
DOC_TITLES = {
    "lumera": "Lumera — Aqua Lumina — données fictives",
    "meta": "Meta — Standards publicitaires",
    "arpp": "ARPP — Produits cosmétiques V8 — référence française",
    "methods": "Guide interne de rédaction et de création visuelle",
}
# Titres explicitement identifiés dans l'export Meta fourni.
# Contrairement à une heuristique sur la longueur des lignes, cette liste
# ne transforme pas les phrases courtes du texte en faux titres.
META_HEADINGS = {
    "1. Vue d’ensemble", "2. Principes des règles publicitaires Meta",
    "Mise en application de nos politiques", "3. Le processus d’examen des publicités",
    "Ce qui est examiné", "Examen des éléments professionnels", "Nouvel examen des publicités",
    "Que faire si votre publicité est rejetée", "Standards de la communauté",
    "Contenu interdit", "Exploitation sexuelle, maltraitance et nudité des enfants",
    "Attaques coordonnées et promotion d’actions criminelles",
    "Organisations et personnes dangereuses", "Pratiques discriminatoires", "Conduite haineuse",
    "Exploitation humaine", "Contenus, produits ou services illégaux localement",
    "Fausses informations", "Dissuasion à l’égard des vaccins", "Pratiques commerciales interdites",
    "Biens et services restreints", "Boissons alcoolisées",
    "Exploitation commerciale des crises et des évènements controversés",
    "Publicités pour des services de rencontre", "Matériaux et objets dangereux",
    "Santé et bien-être", "Vente d’animaux non menacés et d’espèces menacées",
    "Produits et services financiers", "Produits et services associés à la cryptomonnaie",
    "Jeux vidéo et jeux d’argent en ligne", "Contenu répréhensible",
    "Sollicitation sexuelle chez les adultes et langage sexuellement explicite",
    "Intimidation et harcèlement", "Vulgarité",
    "Atteintes à la vie privée et caractéristiques personnelles", "Contenu violent et explicite",
    "Suicide, automutilation et troubles de l’alimentation", "Violation de la propriété intellectuelle",
    "Violation de la propriété intellectuelle d’un tiers",
    "Utilisation des licences de propriété intellectuelle de Meta",
    "Publicités d’enjeu social, électoral ou politique",
    "Publicités portant sur un enjeu social, électoral ou politique",
    "Politiques spécifiques aux produits et aux formats", "Publicités vidéo",
    "1. Contenu trompeur", "2. Restrictions relatives au divertissement",
    "Publicités à formulaire", "Ciblage", "Pertinence", "Contenu de marque",
    "Intégrité des comptes", "Comportement non authentique", "Cybersécurité", "Spam",
    "Demandes des utilisateurs", "Restrictions relatives à l’utilisation des données", "À savoir",
}
ARPP_HEADING = re.compile(r"^(?:\d+\.\d+(?:\.\d+)?|\d+\.)\s+\S")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_tokenizer(local_path=None, offline=False):
    try:
        from transformers import PreTrainedTokenizerFast
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise SystemExit(f"Import du tokenizer impossible : {error}")
    try:
        def resolve(name):
            if local_path:
                return Path(local_path) / name
            return Path(hf_hub_download(MODEL, name, revision=REVISION, local_files_only=offline))
        # Charger le backend fast directement évite l'import d'AutoTokenizer,
        # qui peut importer génération → sklearn → SciPy dans certains environnements.
        # Le tokenizer.json officiel contient la segmentation et le postprocessor.
        special_tokens = json.loads(resolve("special_tokens_map.json").read_text(encoding="utf-8"))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(resolve("tokenizer.json")), model_max_length=MODEL_LIMIT,
            **special_tokens,
        )
    except Exception as error:
        raise SystemExit(
            "Chargement du tokenizer impossible. Vérifier Internet ou utiliser le cache "
            "avec --offline. Un tokenizer local peut être donné avec --tokenizer-path.\n"
            f"Détail : {error}"
        )
    if not tokenizer.is_fast:
        raise SystemExit("Un tokenizer fast est nécessaire pour les offsets de caractères.")
    tokenizer.model_max_length = MODEL_LIMIT
    return tokenizer


def count_tokens(tokenizer, text, special=False):
    return len(tokenizer(text, add_special_tokens=special, truncation=False, verbose=False)["input_ids"])


def build_documents(units):
    grouped = {}
    for unit in units:
        grouped.setdefault(unit["doc_id"], []).append(unit)
    documents = []
    for doc_id, selected in grouped.items():
        if doc_id not in DOC_TITLES:
            raise ValueError(f"Document inconnu : {doc_id}")
        selected.sort(key=lambda u: (u["page_start"], u.get("record_id", "")))
        parts, spans, cursor = [], [], 0
        for unit in selected:
            if parts:
                parts.append("\n\n")
                cursor += 2
            text = unit["text"]
            spans.append({"start": cursor, "end": cursor + len(text), "unit": unit})
            parts.append(text)
            cursor += len(text)
        documents.append({"doc_id": doc_id, "text": "".join(parts), "spans": spans})
    return documents


def trim_range(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def section_ranges(document):
    text = document["text"]
    doc_id = document["doc_id"]
    # Le Product Book est déjà structuré par sections d'une page ; chaque
    # fiche interne a son identifiant et constitue une unité de sens.
    if doc_id in {"lumera", "methods"}:
        return [(s["start"], s["end"], s["unit"]["section"]) for s in document["spans"]]
    headings, cursor = [], 0
    lines = text.splitlines(keepends=True)
    for line_index, line in enumerate(lines):
        stripped = line.strip()
        is_heading = (ARPP_HEADING.match(stripped) or stripped in {"PRÉAMBULE", "Annexe"}) if doc_id == "arpp" else stripped in META_HEADINGS
        # Deux intitulés apparaissent aussi dans une liste récapitulative :
        # leur règle détaillée commence par « Les publicités... ».
        if doc_id == "meta" and stripped in {"Vulgarité", "Atteintes à la vie privée et caractéristiques personnelles"}:
            following = next((x.strip() for x in lines[line_index + 1:] if x.strip()), "")
            is_heading = is_heading and following.startswith("Les publicités")
        if is_heading:
            headings.append((cursor, stripped))
        cursor += len(line)
    points = [(0, "Introduction")] if not headings or headings[0][0] > 0 else []
    points.extend(headings)
    sections = []
    for i, (start, title) in enumerate(points):
        end = points[i + 1][0] if i + 1 < len(points) else len(text)
        start, end = trim_range(text, start, end)
        if end > start:
            sections.append((start, end, title))
    # Attacher un titre sans corps à la section suivante. Les titres parents
    # restent dans le texte ; ils ne deviennent pas des chunks sans connaissance.
    merged, pending_start = [], None
    for start, end, title in sections:
        body = text[start:end].strip()
        title_only = body == title or (title == "Introduction" and len(body) < 50)
        if title_only:
            pending_start = start if pending_start is None else pending_start
            continue
        merged.append((pending_start if pending_start is not None else start, end, title))
        pending_start = None
    if pending_start is not None:
        if merged:
            start, _, title = merged[-1]
            merged[-1] = (start, len(text.rstrip()), title)
        else:
            merged.append((pending_start, len(text.rstrip()), "Introduction"))
    return merged


def token_windows(tokenizer, text, size=300, overlap=50, prefer_paragraphs=False):
    """Renvoyer des intervalles dans le texte original, jamais du texte décodé.

    Un sous-mot peut partager ses offsets avec un autre token. La taille réelle
    de chaque substring est donc recomptée pour empêcher un dépassement.
    """
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)
    offsets = encoded["offset_mapping"]
    if not offsets:
        return
    start_token = 0
    while start_token < len(offsets):
        end_token = min(start_token + size, len(offsets))
        start_char = 0 if start_token == 0 else offsets[start_token][0]
        end_char = len(text) if end_token == len(offsets) else offsets[end_token - 1][1]
        if prefer_paragraphs and end_token < len(offsets):
            lower = offsets[min(start_token + int(size * 0.65), end_token - 1)][0]
            breaks = [m.end() for m in re.finditer(r"\n\s*\n", text[lower:end_char])]
            if breaks:
                end_char = lower + breaks[-1]
        while count_tokens(tokenizer, text[start_char:end_char]) > size:
            end_token -= 1
            end_char = min(end_char, offsets[end_token - 1][1])
        start_char, end_char = trim_range(text, start_char, end_char)
        if end_char <= start_char:
            raise ValueError("Fenêtre sans progression : vérifier les offsets du tokenizer.")
        yield start_char, end_char
        if end_char >= len(text.rstrip()):
            break
        # Dernier token réellement couvert, après l'éventuelle coupure au paragraphe.
        covered = start_token
        while covered < len(offsets) and offsets[covered][1] <= end_char:
            covered += 1
        next_token = max(start_token + 1, covered - overlap)
        while next_token < len(offsets) and offsets[next_token][0] <= start_char:
            next_token += 1
        start_token = next_token


def make_chunk(document, start, end, strategy, index, section, tokenizer):
    matching = [s["unit"] for s in document["spans"] if s["start"] < end and s["end"] > start]
    base = matching[0]
    body = document["text"][start:end]
    # Même ajout de contexte pour A et B, pour isoler l'effet du découpage.
    embedding_text = "passage: " + DOC_TITLES[document["doc_id"]] + "\n" + body
    embedding_tokens = count_tokens(tokenizer, embedding_text, special=True)
    if embedding_tokens > MODEL_LIMIT:
        raise ValueError(f"Chunk trop long pour E5 : {embedding_tokens} tokens.")
    retain = ["doc_id", "source", "doc_type", "source_type", "scope", "synthetic",
              "brand_id", "product_id", "source_sha256", "capture_date"]
    row = {key: base[key] for key in retain if key in base}
    record_ids = [u["record_id"] for u in matching if "record_id" in u]
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
    row.update({
        "chunk_id": f"{strategy}_{document['doc_id']}_{index:04d}_{digest}",
        "strategy": strategy, "section": section, "text": body, "embedding_text": embedding_text,
        "token_count": count_tokens(tokenizer, body), "embedding_token_count": embedding_tokens,
        "tokenizer_model": MODEL, "tokenizer_revision": REVISION,
        "unit_ids": [u["unit_id"] for u in matching], "record_ids": record_ids,
        "page_start": min(u["page_start"] for u in matching),
        "page_end": max(u["page_end"] for u in matching),
        "citations": [{"unit_id": u["unit_id"], "source": u["source"],
                       "page_start": u["page_start"], "page_end": u["page_end"]} for u in matching],
        "char_start": start, "char_end": end,
        "languages": sorted({u.get("language", "fr") for u in matching}),
    })
    return row


def validate(documents, chunks, size):
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "IDs dupliqués"
    for document in documents:
        selected = [c for c in chunks if c["doc_id"] == document["doc_id"]]
        coverage = bytearray(len(document["text"]))
        for chunk in selected:
            a, b = chunk["char_start"], chunk["char_end"]
            assert chunk["text"] == document["text"][a:b], "Texte modifié"
            assert 0 < chunk["token_count"] <= size
            assert chunk["embedding_token_count"] <= MODEL_LIMIT
            assert chunk["citations"] and chunk["unit_ids"]
            coverage[a:b] = b"\1" * (b - a)
        assert all(coverage[i] or char.isspace() for i, char in enumerate(document["text"])), "Texte source perdu"


def summary(chunks):
    counts = [c["token_count"] for c in chunks]
    return {"chunks": len(chunks), "by_document": dict(Counter(c["doc_id"] for c in chunks)),
            "min_tokens": min(counts), "mean_tokens": round(statistics.mean(counts), 1),
            "max_tokens": max(counts), "max_embedding_tokens": max(c["embedding_token_count"] for c in chunks),
            "chunks_below_30_tokens": sum(n < 30 for n in counts),
            "total_tokens_with_overlap": sum(counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/corpus_units.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/chunks")
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--tokenizer-path", type=Path)
    args = parser.parse_args()
    if not 64 <= args.size <= 400 or not 0 <= args.overlap < args.size // 2:
        raise SystemExit("Utiliser une taille de 64 à 400 tokens et overlap < size/2.")
    if not args.input.exists():
        raise SystemExit("Corpus absent : lancer d'abord scripts/prepare_corpus.py.")
    units = read_jsonl(args.input)
    if not units or len({u["unit_id"] for u in units}) != len(units):
        raise SystemExit("Corpus vide ou identifiants d'unités dupliqués.")
    print("Chargement du tokenizer E5 (première exécution : téléchargement du tokenizer uniquement)...", flush=True)
    tokenizer = load_tokenizer(args.tokenizer_path, args.offline)
    documents = build_documents(units)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"model": MODEL, "revision": REVISION, "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "units": len(units), "chunk_size": args.size, "overlap_target": args.overlap,
              "model_limit": MODEL_LIMIT, "coverage_checks_passed": True,
              "retrieval_evaluated": False, "strategies": {}}
    for strategy in ("fixed", "sections"):
        chunks = []
        for document in documents:
            regions = [(0, len(document["text"]), "Fenêtre fixe")] if strategy == "fixed" else section_ranges(document)
            for a, b, section in regions:
                for start, end in token_windows(tokenizer, document["text"][a:b], args.size,
                                                args.overlap, prefer_paragraphs=strategy == "sections"):
                    chunks.append(make_chunk(document, a + start, a + end, strategy,
                                             len(chunks) + 1, section, tokenizer))
        validate(documents, chunks, args.size)
        write_jsonl(args.output / f"chunks_{strategy}.jsonl", chunks)
        stats = summary(chunks)
        report["strategies"][strategy] = stats
        print(f"{strategy}: {stats['chunks']} chunks ; moyenne {stats['mean_tokens']} tokens ; max {stats['max_tokens']}")
        preview = "\n\n---\n\n".join(
            f"## {c['chunk_id']}\nSource : {c['source']} — p.{c['page_start']}-{c['page_end']}\n"
            f"Section : {c['section']} | Tokens : {c['token_count']}\n\n{c['text']}" for c in chunks)
        (args.output / f"preview_{strategy}.md").write_text(preview + "\n", encoding="utf-8")
    # Sert à reconstruire et auditer les offsets, sans changer les sources.
    write_jsonl(args.output / "document_texts.jsonl", [
        {"doc_id": d["doc_id"], "text": d["text"]} for d in documents])
    (args.output / "chunking_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Contrôles réussis : couverture du texte, provenance et taille réelle. Aucun embedding calculé.")


if __name__ == "__main__":
    main()
