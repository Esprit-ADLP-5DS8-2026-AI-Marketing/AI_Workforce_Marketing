import json
import urllib.request
with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4) as response:
    if json.load(response).get('status') != 'ready':
        raise SystemExit(1)
