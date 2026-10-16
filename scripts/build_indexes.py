#!/usr/bin/env python3
"""Encoder les deux corpus et créer deux collections Qdrant locales."""
import argparse
import importlib.metadata
import json
import time
import uuid
import numpy as np
from qdrant_client import QdrantClient, models
from rag_common import (ROOT, MODEL, REVISION, DIMENSION, DB_PATH, INDEX_MANIFEST,
                        COLLECTIONS, read_jsonl, file_sha, encode, load_encoder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rebuild", action="store_true", help="Recréer uniquement adlp_fixed et adlp_sections")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("batch-size doit être positif.")
    datasets, paths = {}, {}
    for strategy in COLLECTIONS:
        path = ROOT / "data/chunks" / f"chunks_{strategy}.jsonl"
        rows = read_jsonl(path)
        if not rows or len({r['chunk_id'] for r in rows}) != len(rows):
            raise SystemExit("Corpus vide ou chunks dupliqués.")
        for row in rows:
            if row["strategy"] != strategy or row["tokenizer_model"] != MODEL or row["tokenizer_revision"] != REVISION:
                raise SystemExit("Le modèle ou la stratégie des chunks ne correspond pas.")
            if not row["embedding_text"].startswith("passage: "):
                raise SystemExit("Préfixe passage: manquant.")
        datasets[strategy], paths[strategy] = rows, path
    print("Chargement E5 : la première exécution télécharge les poids du modèle.", flush=True)
    encoder = load_encoder(args.device, args.offline)
    client = QdrantClient(path=str(DB_PATH))
    try:
        if any(client.collection_exists(name) for name in COLLECTIONS.values()) and not args.rebuild:
            raise SystemExit("Collections déjà présentes. Pour les reconstruire : ajouter --rebuild.")
        INDEX_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"state": "building", "model": MODEL, "revision": REVISION, "dimension": DIMENSION,
                    "distance": "cosine", "normalized": True, "device": args.device,
                    "versions": {p: importlib.metadata.version(p) for p in ['sentence-transformers', 'transformers', 'torch', 'qdrant-client']},
                    "collections": {}}
        INDEX_MANIFEST.write_text(json.dumps(manifest, indent=2))
        for strategy, rows in datasets.items():
            started = time.perf_counter()
            vectors = encode(encoder, [r['embedding_text'] for r in rows], args.batch_size, progress=True)
            np.save(INDEX_MANIFEST.parent / f"embeddings_{strategy}.npy", vectors)
            name = COLLECTIONS[strategy]
            if client.collection_exists(name):
                client.delete_collection(name)
            client.create_collection(name, vectors_config=models.VectorParams(size=DIMENSION, distance=models.Distance.COSINE))
            for start in range(0, len(rows), 64):
                points = [models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, r['chunk_id'])),
                                            vector=vectors[i].tolist(), payload=r)
                          for i, r in enumerate(rows[start:start + 64], start)]
                client.upsert(name, points=points, wait=True)
            if client.count(name, exact=True).count != len(rows):
                raise ValueError("Indexation incomplète.")
            manifest['collections'][strategy] = {"name": name, "points": len(rows),
                                                  "chunks_sha256": file_sha(paths[strategy]),
                                                  "build_seconds": round(time.perf_counter() - started, 3)}
            print(f"{name} : {len(rows)} vecteurs de {DIMENSION} dimensions enregistrés.")
        manifest['state'] = 'ready'
        INDEX_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    finally:
        client.close()
    print("Index prêts. Étape suivante : evaluate_retrieval.py")


if __name__ == '__main__':
    main()
