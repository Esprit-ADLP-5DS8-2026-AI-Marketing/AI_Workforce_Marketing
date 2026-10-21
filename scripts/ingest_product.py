#!/usr/bin/env python3
"""Ingérer un PDF produit. CLI interne de démonstration, sans authentification web."""
import argparse
import json
import sys
from pathlib import Path
from product_rag import ProductError, STORE_PATH, ingest_product_pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', type=Path)
    parser.add_argument('--user-id', required=True, help='Identité de test ; en production, fournie par le backend authentifié')
    parser.add_argument('--product-id', required=True)
    parser.add_argument('--title', help='Nom documentaire, 160 caractères maximum')
    parser.add_argument('--synthetic', action='store_true', help='Marquer explicitement un document fictif de démonstration')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--store', type=Path, default=STORE_PATH)
    args = parser.parse_args()
    result = ingest_product_pdf(args.pdf, args.user_id, args.product_id,
        title=args.title, synthetic=args.synthetic, offline=args.offline,
        device=args.device, store_path=args.store)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except ProductError as error:
        print(json.dumps({'status': 'error', 'code': error.code, 'message': str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
    except (RuntimeError, ValueError, OSError) as error:
        print(json.dumps({'status': 'error', 'code': 'ingestion_failed', 'message': str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
