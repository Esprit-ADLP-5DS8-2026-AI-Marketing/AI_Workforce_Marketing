"""Read-only preflight; never display settings."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from rag_service import load_settings
required = ['scripts/local_qdrant.py', 'scripts/rag_api.py', 'scripts/run_rag_api.py',
            'scripts/prepare_corpus.py', 'scripts/build_indexes.py',
            'prompts/rag_answer_v1_3.txt', '.env']
missing = [name for name in required if not (ROOT / name).is_file()]
if missing:
    raise SystemExit('Missing files: ' + ', '.join(missing))
settings = load_settings(ROOT / '.env')
if not settings.gemini_api_key:
    raise SystemExit('GEMINI_API_KEY is missing.')
if len(list((ROOT / 'data/raw').glob('*.pdf'))) != 4:
    raise SystemExit('Keep exactly the four original PDF files in data/raw before building.')
print('Inputs OK. No secret displayed.')
