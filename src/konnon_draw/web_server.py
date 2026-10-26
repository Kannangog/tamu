"""Lightweight HTTP server providing the interactive Live Drawing Studio webview."""

import argparse
import json
import mimetypes
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import torch

from .renderer import actions_to_bezier_segments, render_svg
from .sampling import generate
from .training import load_model

WEB_DIR = Path(__file__).parent / "web"
PROJECT_ROOT = Path(__file__).parent.parent.parent


class DrawingStudioHandler(BaseHTTPRequestHandler):
    model = None
    ckpt = None
    classes = []
    checkpoint_path = ""

    def log_message(self, format, *args):
        # Keep server log concise
        sys.stderr.write(f"[{self.log_date_time_string()}] {args[0]} {args[1]} -> {args[2]}\n")

    def send_json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            index_path = WEB_DIR / "index.html"
            self.serve_file(index_path, "text/html; charset=utf-8")
            return

        if path == "/app.js":
            js_path = WEB_DIR / "app.js"
            self.serve_file(js_path, "application/javascript; charset=utf-8")
            return

        if path == "/api/status":
            self.send_json({
                "status": "ready",
                "checkpoint": str(self.checkpoint_path),
                "classes": self.classes,
                "hidden_dim": self.ckpt.get("config", {}).get("hidden_dim", 256),
                "parameters": self.model.num_parameters() if hasattr(self.model, "num_parameters") else 390000,
            })
            return

        if path == "/api/showcase":
            showcase_dir = PROJECT_ROOT / "outputs" / "samples" / "showcase"
            items = []
            if showcase_dir.exists():
                svg_files = sorted(list(showcase_dir.glob("*.svg")))
                for sf in svg_files:
                    stem = sf.stem  # e.g. bicycle_000
                    cat = stem.split("_")[0]
                    jf = showcase_dir / f"{stem}.json"
                    if jf.exists():
                        items.append({
                            "id": stem,
                            "category": cat,
                            "svg_url": f"/outputs/samples/showcase/{sf.name}",
                            "json_url": f"/outputs/samples/showcase/{jf.name}",
                        })
            self.send_json(items)
            return

        # Serve static assets from outputs/
        if path.startswith("/outputs/"):
            rel_path = path.lstrip("/")
            local_path = PROJECT_ROOT / rel_path
            if local_path.is_file() and local_path.exists():
                mime, _ = mimetypes.guess_type(str(local_path))
                if path.endswith(".svg"):
                    mime = "image/svg+xml"
                elif path.endswith(".json"):
                    mime = "application/json"
                self.serve_file(local_path, mime or "application/octet-stream")
                return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self):
        path = self.path.split("?")[0]

        if path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)

            try:
                payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
            except Exception:
                payload = {}

            category = payload.get("category", "bicycle")
            seed = int(payload.get("seed", 42))
            temperature = float(payload.get("temperature", 0.4))
            top_k = payload.get("top_k", None)
            smooth = bool(payload.get("smooth", True))
            continuous = bool(payload.get("continuous", True))

            if top_k is not None:
                top_k = int(top_k)

            if category not in self.classes:
                category = self.classes[0] if self.classes else "bicycle"

            class_id = self.classes.index(category)
            max_len = self.ckpt.get("config", {}).get("max_len", 160)

            # Autoregressive generation
            res = generate(
                model=self.model,
                class_id=class_id,
                seed=seed,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k,
                continuous_coords=continuous,
            )

            actions = res["actions"]
            svg_smooth = render_svg(actions, width=256, height=256, smooth=True)
            svg_raw = render_svg(actions, width=256, height=256, smooth=False)
            bezier_strokes = actions_to_bezier_segments(actions, width=512, height=512)

            response_data = {
                "category": category,
                "class_id": class_id,
                "seed": seed,
                "temperature": temperature,
                "top_k": top_k,
                "smooth": smooth,
                "continuous": continuous,
                "n_tokens": res.get("n_tokens", len(actions)),
                "natural_end": res.get("natural_end", True),
                "repaired": res.get("repaired", False),
                "actions": actions,
                "bezier_strokes": bezier_strokes,
                "svg_smooth": svg_smooth,
                "svg_raw": svg_raw,
                "svg": svg_smooth if smooth else svg_raw,
            }
            self.send_json(response_data)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint Not Found")

    def serve_file(self, file_path: Path, content_type: str):
        try:
            content = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Error reading file: {e}")


def run_server(host: str = "127.0.0.1", port: int = 8080, checkpoint_path: str = "outputs/checkpoints/best.pt"):
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading trained Claude model from {ckpt_path}...")
    model, ckpt = load_model(str(ckpt_path))
    classes = ckpt.get("classes", ["bed", "bicycle", "car", "chair", "house", "table"])

    DrawingStudioHandler.model = model
    DrawingStudioHandler.ckpt = ckpt
    DrawingStudioHandler.classes = classes
    DrawingStudioHandler.checkpoint_path = ckpt_path

    server = ThreadingHTTPServer((host, port), DrawingStudioHandler)
    url = f"http://{host}:{port}"
    print(f"\n=======================================================")
    print(f"  Konnon Drawing Model — Live Drawing Studio Webview")
    print(f"  Serving at: {url}")
    print(f"  Loaded Categories: {', '.join(classes)}")
    print(f"  Press Ctrl+C to stop.")
    print(f"=======================================================\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping webview server...")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Start the Live Drawing Studio webview server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt", help="Path to checkpoint")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
