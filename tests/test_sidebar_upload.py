"""Send one PDF to the authenticated n8n ingestion webhook."""
import argparse
import getpass
import json
from pathlib import Path
import httpx

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--file',type=Path,required=True)
    p.add_argument('--user-id',default='demo_mohamed')
    p.add_argument('--product-id',default='aqua_lumina')
    p.add_argument('--title',default='Aqua Lumina - demonstration')
    p.add_argument('--synthetic',action='store_true')
    p.add_argument('--production',action='store_true')
    a=p.parse_args()
    if not a.file.is_file():p.error('PDF introuvable.')
    key=getpass.getpass('Cle X-Workflow-Key du backend (saisie masquee) : ')
    if not key:raise SystemExit('Cle requise.')
    route='webhook' if a.production else 'webhook-test'
    url=f'http://localhost:5678/{route}/rag/v2/product-document'
    try:
        with a.file.open('rb') as f, httpx.Client(timeout=240,follow_redirects=False) as client:
            r=client.post(url,headers={'X-Workflow-Key':key,'X-User-ID':a.user_id},
                data={'product_id':a.product_id,'title':a.title,'synthetic':str(a.synthetic).lower()},
                files={'file':(a.file.name,f,'application/pdf')})
        print('HTTP',r.status_code)
        try:print(json.dumps(r.json(),ensure_ascii=False,indent=2))
        except ValueError:print('Reponse non JSON. Verifier le webhook en ecoute et son authentification.')
    except httpx.RequestError:
        raise SystemExit('n8n inaccessible ou delai expire. Verifier les conteneurs et le webhook.')

if __name__=='__main__':main()
