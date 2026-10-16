#!/usr/bin/env python3
"""Contournement local du défaut Mach-O __thread_bss de PROPACK sur macOS.

python scripts/repair_scipy.py             # inspecter et vérifier
python scripts/repair_scipy.py --apply     # sauvegarder, corriger, signer, vérifier
python scripts/repair_scipy.py --restore   # restaurer nos propres changements

Cette correction n'est pas une mise à jour officielle de SciPy.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUPS = ROOT / "data/maintenance/scipy_propack_backup"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bad_sections(blob):
    """Retourner seulement les offsets de __thread_bss S_THREAD_LOCAL_ZEROFILL."""
    if len(blob) < 32 or struct.unpack_from("<I", blob)[0] != 0xFEEDFACF:
        raise ValueError("Format autre que Mach-O 64 bits little-endian : correction refusée.")
    _, cpu, _, _, ncmds, sizeofcmds, _, _ = struct.unpack_from("<8I", blob)
    if cpu != 0x0100000C:
        raise ValueError("Ce correctif est limité aux fichiers Mach-O ARM64.")
    commands_end = 32 + sizeofcmds
    if commands_end > len(blob):
        raise ValueError("En-tête Mach-O incomplet.")
    result, cursor = [], 32
    for _ in range(ncmds):
        if cursor + 8 > commands_end:
            raise ValueError("Commande Mach-O incomplète.")
        cmd, size = struct.unpack_from("<II", blob, cursor)
        if size < 8 or cursor + size > commands_end:
            raise ValueError("Taille de commande invalide.")
        if cmd == 0x19:  # LC_SEGMENT_64
            if size < 72:
                raise ValueError("Segment incomplet.")
            nsects = struct.unpack_from("<I", blob, cursor + 64)[0]
            if 72 + nsects * 80 > size:
                raise ValueError("Table des sections incomplète.")
            for i in range(nsects):
                p = cursor + 72 + i * 80
                name = blob[p:p + 16].split(b"\0")[0]
                segment = blob[p + 16:p + 32].split(b"\0")[0]
                offset = struct.unpack_from("<I", blob, p + 48)[0]
                section_type = struct.unpack_from("<I", blob, p + 64)[0] & 0xFF
                if name == b"__thread_bss" and segment == b"__DATA" and section_type == 0x12 and offset:
                    result.append({"field_position": p + 48, "old_offset": offset})
        cursor += size
    if cursor != commands_end:
        raise ValueError("Comptage des commandes incohérent.")
    return result


def check_import():
    code = "import scipy.sparse.linalg; from sklearn.metrics import roc_curve; from sentence_transformers import SentenceTransformer; print('SciPy, sklearn et SentenceTransformer OK')"
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--apply", action="store_true")
    options.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("Correctif prévu pour macOS ARM64 uniquement.")
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Activer ai_env avant toute correction.")
    distribution = importlib.metadata.distribution("scipy")
    root = Path(distribution.locate_file("scipy")).resolve()
    if not root.is_relative_to(Path(sys.prefix).resolve()):
        raise SystemExit("SciPy se trouve hors de l'environnement actif : arrêt.")
    manifest = BACKUPS / "manifest.json"
    print(f"Python : {sys.executable}\nSciPy : {distribution.version}\nDossier : {root}")
    if args.restore:
        if not manifest.exists():
            raise SystemExit("Aucune sauvegarde créée par ce script.")
        history = json.loads(manifest.read_text())
        for item in history:
            target = Path(item["target"]).resolve()
            backup = Path(item["backup"])
            if not target.is_relative_to(root) or not backup.resolve().is_relative_to(BACKUPS.resolve()):
                raise SystemExit("Chemin de restauration incohérent.")
            if sha(target) != item["after_sha256"] or sha(backup) != item["before_sha256"]:
                raise SystemExit("Fichier modifié depuis la correction : restauration automatique refusée.")
        for item in history:
            shutil.copy2(item["backup"], item["target"])
        manifest.rename(BACKUPS / "manifest_restored.json")
        print("Fichiers originaux restaurés.")
        return
    files = list((root / "sparse/linalg/_propack").glob("*.so"))
    if not files:
        raise SystemExit("Fichiers PROPACK introuvables.")
    repairs = []
    for path in files:
        path = path.resolve()
        if not path.is_relative_to(root):
            raise SystemExit("Lien vers un fichier extérieur : correction refusée.")
        fields = bad_sections(path.read_bytes())
        if fields:
            repairs.append((path, fields))
            print(f"À corriger : {path.name}, {len(fields)} champ(s) d'offset.")
    if repairs and not args.apply:
        print("Inspection uniquement. Pour appliquer avec sauvegarde : python scripts/repair_scipy.py --apply")
        return
    if repairs:
        if not Path("/usr/bin/codesign").exists():
            raise SystemExit("codesign absent : aucun fichier modifié.")
        if manifest.exists():
            raise SystemExit("Une sauvegarde active existe déjà ; ne pas la remplacer.")
        BACKUPS.mkdir(parents=True, exist_ok=True)
        history = []
        try:
            for path, fields in repairs:
                backup = BACKUPS / path.name
                if backup.exists():
                    raise RuntimeError(f"Sauvegarde existante : {backup}")
                original_hash = sha(path)
                shutil.copy2(path, backup)
                history.append({"target": str(path), "backup": str(backup), "before_sha256": original_hash})
                blob = bytearray(path.read_bytes())
                for field in fields:
                    struct.pack_into("<I", blob, field["field_position"], 0)
                path.write_bytes(blob)
                subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(path)],
                               check=True, capture_output=True, text=True)
                if bad_sections(path.read_bytes()):
                    raise RuntimeError("Le défaut persiste après correction.")
                history[-1]["after_sha256"] = sha(path)
                print(f"Corrigé et signé localement : {path.name}")
        except Exception as error:
            for item in history:
                shutil.copy2(item["backup"], item["target"])
                Path(item["backup"]).unlink()
            if getattr(error, "stderr", None):
                print(error.stderr, file=sys.stderr)
            raise
        manifest.write_text(json.dumps(history, indent=2) + "\n")
    else:
        print("Aucun offset __thread_bss invalide détecté.")
    result = check_import()
    if result.returncode:
        print(result.stderr)
        raise SystemExit("Import encore en échec. Ne pas lancer l'indexation ; partager cette erreur.")
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
