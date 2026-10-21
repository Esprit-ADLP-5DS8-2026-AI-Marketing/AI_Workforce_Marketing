"""Utilitaires partagés : E5, Qdrant et validation du corpus."""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DIMENSION = 384
DB_PATH = ROOT / "data/qdrant_local"
INDEX_MANIFEST = ROOT / "data/index/index_manifest.json"
COLLECTIONS = {"fixed": "adlp_fixed", "sections": "adlp_sections"}


def read_jsonl(path):
    if not path.exists():
        raise SystemExit(f"Fichier absent : {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_encoder(device="cpu", offline=False):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL, revision=REVISION, device=device,
                                    local_files_only=offline, trust_remote_code=False)
    except Exception as error:
        raise SystemExit(f"Chargement E5 impossible : {error}\nSi l'erreur concerne _spropack, lancer repair_scipy.py --apply.")
    model.default_prompt_name = None  # Préfixes fournis explicitement.
    model.max_seq_length = 512
    dimension_getter = getattr(model, "get_embedding_dimension", None)
    if dimension_getter is None:
        dimension_getter = model.get_sentence_embedding_dimension
    if dimension_getter() != DIMENSION:
        raise SystemExit("Dimension inattendue pour E5.")
    return model


def encode(model, texts, batch_size=16, progress=False):
    import numpy as np
    if not texts:
        raise ValueError("Aucun texte à encoder.")
    for i, text in enumerate(texts):
        size = len(model.tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"])
        if size > 512:
            raise ValueError(f"Texte {i} : {size} tokens, supérieur à 512. Aucun tronquage silencieux autorisé.")
    vectors = np.asarray(model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                     normalize_embeddings=True, show_progress_bar=progress), dtype=np.float32)
    if vectors.shape != (len(texts), DIMENSION) or not np.isfinite(vectors).all():
        raise ValueError("Embeddings invalides.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("Embeddings non normalisés.")
    return vectors


def open_index():
    from local_qdrant import LocalQdrantClient as QdrantClient
    if not INDEX_MANIFEST.exists():
        raise SystemExit("Index absent : lancer build_indexes.py.")
    manifest = json.loads(INDEX_MANIFEST.read_text())
    if manifest.get("state") != "ready" or manifest.get("model") != MODEL or manifest.get("revision") != REVISION:
        raise SystemExit("Index incomplet ou modèle différent : reconstruire les index.")
    for strategy, entry in manifest["collections"].items():
        path = ROOT / "data/chunks" / f"chunks_{strategy}.jsonl"
        if not path.exists() or file_sha(path) != entry["chunks_sha256"]:
            raise SystemExit("Les chunks ont changé : reconstruire les index avec --rebuild.")
    client = QdrantClient(path=str(DB_PATH))
    try:
        for entry in manifest["collections"].values():
            if client.count(entry["name"], exact=True).count != entry["points"]:
                raise ValueError("Nombre de points incohérent : reconstruire les index.")
    except Exception:
        client.close()
        raise
    return client, manifest


def retrieve(client, strategy, vector, top_k=4, doc_id=None):
    from qdrant_client import models
    if top_k < 1:
        raise ValueError("top_k doit être positif.")
    query_filter = None
    if doc_id:
        query_filter = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
    started = time.perf_counter()
    hits = client.query_points(collection_name=COLLECTIONS[strategy], query=vector.tolist(),
                               query_filter=query_filter, limit=top_k, with_payload=True).points
    latency_ms = (time.perf_counter() - started) * 1000
    return [{"rank": i, "score": float(hit.score), "payload": hit.payload} for i, hit in enumerate(hits, 1)], latency_ms
