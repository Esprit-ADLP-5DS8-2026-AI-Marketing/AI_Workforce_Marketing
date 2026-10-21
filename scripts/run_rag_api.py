#!/usr/bin/env python3
"""Démarrer l'API interne avec un seul worker et sans reload automatique."""
import argparse
import uvicorn
from rag_api import create_app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, workers=1,
                limit_concurrency=16, timeout_keep_alive=10)
