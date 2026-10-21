#!/usr/bin/env python3
"""Ajouter les réglages API manquants à .env sans afficher les secrets."""
from pathlib import Path
import os
import secrets
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def configure(path=ROOT / '.env'):
    path = Path(path)
    values = dotenv_values(path, interpolate=False) if path.exists() else {}
    defaults = {'RAG_SERVICE_TOKEN': secrets.token_urlsafe(32), 'RAG_OFFLINE': 'true', 'RAG_DEVICE': 'cpu'}
    additions = {key: value for key, value in defaults.items() if not values.get(key) and not os.environ.get(key)}
    if additions:
        # Ne pas réécrire les lignes Gemini existantes ni afficher leur contenu.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, 'a', encoding='utf-8') as stream:
            stream.write('\n# Internal RAG API\n')
            for key, value in additions.items():
                stream.write(f'{key}={value}\n')
        try:
            path.chmod(0o600)
        except OSError:
            pass
    return list(additions)


if __name__ == '__main__':
    added = configure()
    print('Réglages ajoutés : ' + ', '.join(added) if added else 'Configuration API déjà présente.')
    print('Clés conservées dans .env ; aucune clé affichée. Ne pas versionner .env.')
