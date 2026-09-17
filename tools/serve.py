"""Loopback static development server; optional local-only test report sink."""
import json
from urllib.parse import urlsplit
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Handler(SimpleHTTPRequestHandler):
    extensions_map={**SimpleHTTPRequestHandler.extensions_map,'.js':'text/javascript','.mjs':'text/javascript','.gz':'application/octet-stream'}
    def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT),**kw)
    def end_headers(self):
        self.send_header('Cache-Control','no-cache')
        self.send_header('Cross-Origin-Opener-Policy','same-origin')
        self.send_header('Cross-Origin-Embedder-Policy','require-corp')
        super().end_headers()
    def do_GET(self):
        url=urlsplit(self.path)
        if url.path=='/' and (ROOT/'dist'/'index.html').exists():
            self.send_response(302)
            self.send_header('Location','/dist/'+(('?'+url.query) if url.query else ''))
            self.end_headers()
            return
        super().do_GET()
    def do_POST(self):
        if self.path!='/test-report':self.send_error(404);return
        size=int(self.headers.get('Content-Length','0'))
        if not 0<size<1000000:self.send_error(413);return
        obj=json.loads(self.rfile.read(size));out=ROOT/'local-results';out.mkdir(exist_ok=True)
        with (out/'browser.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(obj)+'\n')
        self.send_response(204);self.end_headers()
    def log_message(self,*args):pass
print('Static WebGPU preview http://127.0.0.1:8812',flush=True)
ThreadingHTTPServer(('127.0.0.1',8812),Handler).serve_forever()
