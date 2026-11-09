"""A very small HTTP server so that a game engine can ask for damage reports.

    python -m crashdamage serve --model models/gnn_demo.pt --port 8765

    POST /predict[?engine=physics|model]   body: scenario JSON   ->  damage report JSON
    GET  /health
    GET  /parts        part ids, materials and the connection graph
    GET  /obstacles    obstacle presets, materials, shapes

It only uses the standard library and listens on 127.0.0.1 by default.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.parse import parse_qs, urlparse

from .materials import MATERIAL_NAMES
from .obstacles import AFTER, FORMS, PRESET_NAMES, SHAPES
from .vehicle import EDGES, FACES, PART_NAMES, PART_TEMPLATE

MAX_BODY = 1_000_000


def make_handler(predictor):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):      # keep the console quiet
            pass

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/health":
                self._send(200, {"status": "ok", "model_loaded": predictor.has_model})
            elif path == "/parts":
                self._send(200, {
                    "faces": FACES,
                    "parts": [{"id": p.name, "kind": p.kind, "material": p.material, "thickness_mm": p.thickness_mm}
                              for p in PART_TEMPLATE],
                    "edges": [[PART_NAMES[i], PART_NAMES[j], w] for i, j, w in EDGES]})
            elif path == "/obstacles":
                self._send(200, {"presets": PRESET_NAMES, "materials": MATERIAL_NAMES, "shapes": SHAPES,
                                 "forms": FORMS, "after": AFTER})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            url = urlparse(self.path)
            if url.path != "/predict":
                self._send(404, {"error": "not found"})
                return
            engine = parse_qs(url.query).get("engine", ["auto"])[0]
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self._send(413, {"error": "request too large"})
                return
            try:
                scenario = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError as e:
                self._send(400, {"error": f"invalid JSON: {e}"})
                return
            try:
                self._send(200, predictor.predict(scenario, engine=engine))
            except (ValueError, KeyError, TypeError) as e:
                self._send(400, {"error": str(e)})

    return Handler


def create_server(predictor, host="127.0.0.1", port=8765):
    return ThreadingHTTPServer((host, port), make_handler(predictor))


def serve(predictor, host="127.0.0.1", port=8765):
    server = create_server(predictor, host, port)
    mode = "trained model" if predictor.has_model else "physics layer only"
    print(f"listening on http://{host}:{server.server_address[1]}  ({mode})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
