"""Build-time preparation; no private products and no Gemini calls."""
import os
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from rag_common import load_encoder, open_index
from rerank_rag import load_reranker

def main():
    os.chdir(ROOT)
    load_encoder('cpu', offline=False)
    load_reranker('cpu', offline=False)
    for command in (
        ['scripts/prepare_corpus.py'],
        ['scripts/chunk_corpus.py', '--offline'],
        ['scripts/build_indexes.py', '--offline', '--device', 'cpu'],
    ):
        subprocess.run([sys.executable, *command], check=True)
    # Verify the complete cache and index before exporting an image.
    load_encoder('cpu', offline=True)
    load_reranker('cpu', offline=True)
    client, manifest = open_index()
    try:
        counts = {key: value['points'] for key, value in manifest['collections'].items()}
        if counts != {'fixed': 79, 'sections': 144}:
            raise RuntimeError('Unexpected corpus counts: ' + str(counts))
    finally:
        client.close()
    print('Image assets ready: models cached, 79 fixed / 144 sections; no Gemini call.')

if __name__ == '__main__':
    main()
