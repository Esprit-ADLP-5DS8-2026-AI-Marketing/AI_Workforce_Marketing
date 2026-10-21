"""Copy only the required settings into the explicitly requested private image."""
import json
import sys
from pathlib import Path
from dotenv import dotenv_values

def embed(source, destination):
    values = dotenv_values(source, interpolate=False)
    token = values.get('RAG_SERVICE_TOKEN') or ''
    key = values.get('GEMINI_API_KEY') or ''
    model = values.get('GEMINI_MODEL') or ''
    if not 32 <= len(token) <= 256 or not token.isascii():
        raise ValueError('Missing or invalid RAG_SERVICE_TOKEN; run configure_rag_api.py.')
    if not key.strip() or not model.strip():
        raise ValueError('GEMINI_API_KEY and GEMINI_MODEL are required in the local .env.')
    selected = {'GEMINI_API_KEY': key, 'GEMINI_MODEL': model,
                'RAG_SERVICE_TOKEN': token, 'RAG_OFFLINE': 'true', 'RAG_DEVICE': 'cpu'}
    # JSON double-quoted strings are valid dotenv values for these credentials.
    if any(any(c in value for c in '\r\n\x00') for value in selected.values()):
        raise ValueError('Multiline settings are not supported.')
    output = Path(destination)
    output.write_text(''.join(name + '=' + json.dumps(value) + '\n'
                              for name, value in selected.items()), encoding='utf-8')
    output.chmod(0o600)
    print('Private runtime configuration embedded; values are not displayed.')

if __name__ == '__main__':
    embed(sys.argv[1], sys.argv[2])
