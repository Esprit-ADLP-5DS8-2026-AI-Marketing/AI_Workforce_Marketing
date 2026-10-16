import argparse
import importlib.metadata
import json
import platform
import shutil
import struct
import sys
from pathlib import Path

from repair_scipy import bad_sections, sha, check_import

ROOT = Path(__file__).resolve().parents[1]
BACKUPS = ROOT / "data/maintenance/scipy_extra_backup"
MANIFEST = BACKUPS / "manifest.json"


def main():
    parser = argparse.ArgumentParser()
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--apply", action="store_true")
    options.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("Ce correctif nécessite macOS ARM64.")
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Active ai_env avant de continuer.")

    distribution = importlib.metadata.distribution("scipy")
    scipy_root = Path(distribution.locate_file("scipy")).resolve()
    if not scipy_root.is_relative_to(Path(sys.prefix).resolve()):
        raise SystemExit("SciPy est hors de l'environnement actif.")

    history = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    for item in history:
        target = Path(item["target"]).resolve()
        backup = Path(item["backup"]).resolve()
        if (
            not target.is_relative_to(scipy_root)
            or not backup.is_relative_to(BACKUPS.resolve())
        ):
            raise SystemExit("Chemin de sauvegarde incohérent.")
        if (
            sha(target) != item["after_sha256"]
            or sha(backup) != item["before_sha256"]
        ):
            raise SystemExit("Un fichier a changé depuis sa correction.")

    if args.restore:
        if not MANIFEST.exists():
            raise SystemExit("Aucune correction supplémentaire à restaurer.")
        for item in history:
            shutil.copy2(item["backup"], item["target"])
        MANIFEST.replace(BACKUPS / "manifest_restored.json")
        print("Corrections supplémentaires restaurées.")
        return

    files = sorted(scipy_root.rglob("*.so"))
    if not files:
        raise SystemExit("Aucune extension SciPy trouvée.")

    repairs = []
    for path in files:
        path = path.resolve()
        if not path.is_relative_to(scipy_root):
            raise SystemExit("Extension extérieure à SciPy : arrêt.")
        fields = bad_sections(path.read_bytes())
        if fields:
            repairs.append((path, fields))
            print("À corriger :", path.relative_to(scipy_root))

    print(f"{len(files)} extensions inspectées ; {len(repairs)} à corriger.")

    if repairs and not args.apply:
        print("Pour corriger : python scripts/repair_scipy_extra.py --apply")
        return

    if repairs:
        import subprocess

        if not Path("/usr/bin/codesign").exists():
            raise SystemExit("codesign absent : aucun changement.")

        BACKUPS.mkdir(parents=True, exist_ok=True)
        added = []
        pending = BACKUPS / "manifest.pending.json"

        try:
            for path, fields in repairs:
                backup = BACKUPS / path.relative_to(scipy_root)
                backup.parent.mkdir(parents=True, exist_ok=True)
                original_hash = sha(path)

                if backup.exists() and sha(backup) != original_hash:
                    raise RuntimeError("Une sauvegarde différente existe déjà.")

                shutil.copy2(path, backup)
                added.append({
                    "target": str(path),
                    "backup": str(backup),
                    "before_sha256": original_hash,
                })

                blob = bytearray(path.read_bytes())
                if bad_sections(blob) != fields:
                    raise RuntimeError("Le fichier a changé depuis l'inspection.")

                for field in fields:
                    struct.pack_into("<I", blob, field["field_position"], 0)

                path.write_bytes(blob)
                subprocess.run(
                    ["/usr/bin/codesign", "--force", "--sign", "-", str(path)],
                    check=True,
                )

                if bad_sections(path.read_bytes()):
                    raise RuntimeError("Le défaut persiste après correction.")

                added[-1]["after_sha256"] = sha(path)
                print("Corrigé :", path.relative_to(scipy_root))

            pending.write_text(json.dumps(history + added, indent=2) + "\n")
            pending.replace(MANIFEST)

        except BaseException:
            for item in added:
                shutil.copy2(item["backup"], item["target"])
            if pending.exists():
                pending.unlink()
            raise

    result = check_import()
    if result.returncode:
        print(result.stderr)
        raise SystemExit("Import encore en échec : partager cette erreur.")
    print(result.stdout.strip())


if __name__ == "__main__":
    main()