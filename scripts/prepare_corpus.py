#!/usr/bin/env python3
"""Étape 1 : extraire les quatre PDF et sélectionner le corpus du MVP.

Les sorties sont des unités documentaires.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

try:
    import pymupdf
except ImportError:
    raise SystemExit("Dépendance manquante : python -m pip install pymupdf")


ROOT = Path(__file__).resolve().parents[1]
# Guides internes utiles au MVP ; ce ne sont pas des règles officielles.
KEEP_RECORDS = {
    "C-010", "C-011", "C-012", "C-013", "C-014", "C-020", "C-021",
    "V-002", "V-003", "V-004", "V-005", "V-010", "V-011", "V-012",
    "V-020", "V-021", "V-022", "V-023", "V-024",
    "R-003", "R-004", "R-009",
}
PRODUCT_SECTIONS = {
    1: "Présentation", 2: "Marque", 3: "Identité et ton de voix",
    4: "Fiche produit", 5: "Formulation et utilisation", 6: "Publics cibles",
    7: "Prix et offres", 8: "Règles de communication", 9: "Brief et FAQ",
}


def identify(path):
    name = path.name.lower()
    if name.startswith("lumera_aqua_lumina"):
        return "lumera", "product", "synthetic_product", "demo_fr_tn"
    if name.startswith("standards publicitaires meta"):
        return "meta", "ad_policy", "official_export", "meta_ads"
    if name.startswith("recommandation-produits-cosm"):
        return "arpp", "cosmetics_guidance", "professional_reference", "FR"
    if name == "rag_database_content.pdf":
        return "methods", "internal_guide", "authored_guidance", "project"
    return None


def clean_page(text, doc_id, page):
    text = unicodedata.normalize("NFC", text).replace("\u00ad", "")
    lines = text.splitlines()
    kept = []
    for i, line in enumerate(lines):
        s = line.strip()
        # Enlever seulement le numéro de page situé en début/fin de page.
        if s == str(page) and (i < 4 or i >= len(lines) - 4):
            continue
        if doc_id == "lumera" and (
            s.startswith("LUMERA  |")
            or s.startswith("Document fictif a usage pedagogique - donnees synthetiques - Lumera")
        ):
            continue
        if doc_id == "meta" and (
            re.fullmatch(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}", s)
            or re.fullmatch(r"\d+/29", s)
            or s == "Présentation des Standards publicitaires | Espace modération"
            or s == "https://transparency.meta.com/policies/ad-standards/"
            or s == "En savoir plus"
        ):
            continue
        if doc_id == "methods" and (
            s == "Marketing AI Workforce - RAG database content"
            or re.fullmatch(r"Page \d+", s)
        ):
            continue
        kept.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def method_units(pages, base):
    # Les marqueurs internes permettent de conserver l'origine des fiches
    # même lorsqu'une fiche s'étend sur plusieurs pages.
    joined = "\n".join(f"@@PAGE:{p['page']}@@\n{p['clean_text']}" for p in pages)
    matches = list(re.finditer(r"(?m)^id: ([CVRW]-\d+) \|[^\n]*", joined))
    result, excluded = [], []
    for index, match in enumerate(matches):
        record_id = match.group(1)
        if record_id not in KEEP_RECORDS:
            excluded.append(record_id)
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(joined)
        body = joined[match.end():end].strip()
        # Ne pas inclure l'introduction de la collection suivante.
        body = re.split(r"(?m)^Collection \d+:", body, maxsplit=1)[0].rstrip()
        preceding = list(re.finditer(r"@@PAGE:(\d+)@@", joined[:match.start()]))
        start_page = int(preceding[-1].group(1))
        more_pages = [int(x) for x in re.findall(r"@@PAGE:(\d+)@@", body)]
        body = re.sub(r"@@PAGE:\d+@@\n?", "", body).strip()
        metadata = {}
        for field in match.group(0).split(" | "):
            key, value = field.split(": ", 1)
            metadata[key] = value
        result.append({
            **base, "unit_id": f"methods_{record_id}", "record_id": record_id,
            "page_start": start_page, "page_end": max([start_page] + more_pages),
            "section": body.splitlines()[0], "text": body,
            "platforms": metadata.get("platform", "all").split(","),
            "language": metadata.get("language", "en"),
            "collection": metadata.get("collection"),
        })
    if {r['record_id'] for r in result} != KEEP_RECORDS:
        raise ValueError("Fiches attendues manquantes : vérifier la version du PDF methods.")
    return result, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    identified = {}
    ignored = []
    for path in sorted(args.input.glob("*")):
        if path.suffix.lower() != ".pdf":
            continue
        identity = identify(path)
        if identity is None:
            ignored.append(path.name)
            continue
        if identity[0] in identified:
            raise SystemExit(f"Doublon pour {identity[0]} : conserver une seule copie.")
        identified[identity[0]] = (path, identity)
    missing = {"lumera", "meta", "arpp", "methods"} - set(identified)
    if missing:
        raise SystemExit(f"PDF manquants dans {args.input}: {', '.join(sorted(missing))}")

    args.output.mkdir(parents=True, exist_ok=True)
    preview = args.output / "preview"
    preview.mkdir(exist_ok=True)
    raw_rows, units, documents, excluded_records, excluded_pages = [], [], [], [], []
    for doc_id in ("lumera", "meta", "arpp", "methods"):
        path, (_, doc_type, source_type, scope) = identified[doc_id]
        base = {
            "doc_id": doc_id, "source": path.name, "doc_type": doc_type,
            "source_type": source_type, "scope": scope,
            "synthetic": doc_id == "lumera", "language": "fr",
            "brand_id": "lumera" if doc_id == "lumera" else None,
            "product_id": "aqua_lumina" if doc_id == "lumera" else None,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if doc_id == "meta":
            base["capture_date"] = "2026-10-04"
        pages = []
        with pymupdf.open(path) as document:
            for number, page in enumerate(document, 1):
                # Ordre natif vérifié sur les quatre PDF fournis.
                original = page.get_text("text", sort=False)
                if not original.strip():
                    raise SystemExit(f"Page sans texte : {path.name}, p. {number}. OCR à examiner.")
                clean = clean_page(original, doc_id, number)
                raw_rows.append({**base, "page": number, "text": original})
                pages.append({"page": number, "clean_text": clean})
            if doc_id == "meta" and len(pages) != 29:
                raise SystemExit("Le filtre Meta vise l'export de 29 pages fourni. Vérifier ce nouveau PDF.")
        if doc_id == "methods":
            selected, excluded_records = method_units(pages, base)
        else:
            selected = []
            for item in pages:
                number = item["page"]
                if doc_id == "meta" and number >= 24:
                    excluded_pages.append({"doc_id": doc_id, "page": number, "reason": "navigation"})
                    continue
                text = item["clean_text"]
                if not text:
                    continue
                section = PRODUCT_SECTIONS[number] if doc_id == "lumera" else f"Page {number}"
                selected.append({**base, "unit_id": f"{doc_id}_p{number:02d}",
                                 "page_start": number, "page_end": number,
                                 "section": section, "text": text})
        units.extend(selected)
        documents.append({"doc_id": doc_id, "source": path.name, "pages": len(pages),
                          "units": len(selected), "characters": sum(len(x['text']) for x in selected)})
        preview_text = "\n\n".join(
            f"## {u['unit_id']} — pages {u['page_start']}-{u['page_end']}\n\n{u['text']}"
            for u in selected
        )
        (preview / f"{doc_id}.md").write_text(preview_text + "\n", encoding="utf-8")
        print(f"{doc_id}: {len(pages)} pages → {len(selected)} unités conservées")

    write_jsonl(args.output / "raw_pages.jsonl", raw_rows)
    write_jsonl(args.output / "corpus_units.jsonl", units)
    report = {
        "stage": "extraction_and_selection_before_chunking", "documents": documents,
        "total_units": len(units), "ignored_files": ignored,
        "excluded_meta_pages": excluded_pages, "excluded_method_records": excluded_records,
        "review_flags": [
            {"source": "lumera", "pages": [4, 5], "issue": "Hydratation en profondeur : conflit avec ARPP 2.8 (p.6). Garder la source ; tester le Reviewer."},
            {"source": "lumera", "pages": [4, 5, 6], "issue": "Éclat annoncé dès la première semaine / en quelques jours, mais résultat chiffré présenté après 4 semaines. Ne pas déduire un résultat à une semaine."},
            {"source": "lumera", "issue": "Prix, tests et avis fictifs : corpus de démonstration uniquement. Dates et marché à filtrer ensuite."},
            {"source": "arpp", "issue": "Référence française à identifier par scope=FR, pas une règle tunisienne universelle."},
            {"source": "methods", "issue": "Conseils internes, parfois en anglais. Aucune garantie de performance ou de préservation du packaging par un modèle."},
            {"source": "all", "issue": "Les tables restent du texte linéaire : contrôler les aperçus ingrédients/prix/tests et conserver le PDF comme référence."},
        ],
    }
    (args.output / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Terminé : {len(units)} unités. Ouvrir {preview} et preparation_report.json.")
    print("Étape suivante : chunking comparé puis embeddings ; aucun vecteur créé ici.")


if __name__ == "__main__":
    main()
