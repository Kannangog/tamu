"""Lightweight HTTP server providing the interactive Live Drawing Studio webview."""

import argparse
import base64
import io
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
from .sampling import generate, generate_with_prefix
from .training import load_model
from .v2.schema import DocumentV2, PointV2, StrokeActionV2
from .v2.renderer import export_svg_v2, render_document_v2
from .v2.convert_v1 import v1_to_v2
from .v2.tokens import doc_to_token_sequence, token_sequence_to_doc, VOCAB_SIZE
from .model.stroke_transformer import KonnonStagedTransformer
from .model.text_encoder import FrozenTextEncoder
from .model.sampling import generate_staged_drawing, sample_stage_tokens

WEB_DIR = Path(__file__).parent / "web"
PROJECT_ROOT = Path(__file__).parent.parent.parent


def image_to_data_url(img) -> str:
    """Converts a PIL Image to a base64 PNG data URL."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class DrawingStudioHandler(BaseHTTPRequestHandler):
    model = None
    ckpt = None
    classes = []
    checkpoint_path = ""
    staged_model = None
    staged_encoder = None
    staged_ckpt_path = None

    @classmethod
    def get_gru_model(cls):
        if cls.model is None:
            cands = [
                PROJECT_ROOT / "outputs_qd" / "checkpoints" / "best.pt",
                PROJECT_ROOT / "outputs" / "checkpoints" / "best.pt",
                PROJECT_ROOT / "outputs_qd_h384" / "checkpoints" / "best.pt",
            ]
            for cand in cands:
                if cand.exists():
                    try:
                        raw = torch.load(str(cand), map_location="cpu")
                        if "model_config" in raw:
                            from .model.gru import DrawingGRU
                            m = DrawingGRU(**raw["model_config"])
                            m.load_state_dict(raw["model_state"])
                            m.eval()
                            cls.model = m
                            cls.ckpt = raw
                            cls.classes = raw.get("classes", ["bed", "bicycle", "car", "chair", "house", "table"])
                            if not cls.checkpoint_path:
                                cls.checkpoint_path = str(cand)
                            break
                    except Exception:
                        pass
        return cls.model, cls.ckpt, cls.classes

    @classmethod
    def get_staged_model(cls):
        if cls.staged_model is None:
            candidates = [
                PROJECT_ROOT / "outputs_v2" / "scaled_116m" / "checkpoint_best.pt",
                PROJECT_ROOT / "outputs_v2" / "scaled" / "full" / "checkpoint_best.pt",
                PROJECT_ROOT / "outputs_v2" / "laptop" / "checkpoints" / "best.pt",
                PROJECT_ROOT / "outputs_v2" / "overfit" / "model_overfit.pt",
                PROJECT_ROOT / "outputs_v2" / "tiny" / "checkpoints" / "best.pt",
            ]
            ckpt_path = None
            for cand in candidates:
                if cand.exists():
                    ckpt_path = cand
                    break

            if ckpt_path and ckpt_path.exists():
                c_data = torch.load(ckpt_path, map_location="cpu")
                cfg = c_data.get("config", {})
                m_cfg = cfg.get("model", cfg)  # handles both flat and nested configs
                d_m = m_cfg.get("d_model", 128)
                cls.staged_model = KonnonStagedTransformer(
                    vocab_size=VOCAB_SIZE,
                    d_model=d_m,
                    n_layers=m_cfg.get("n_layers", 3),
                    n_heads=m_cfg.get("n_heads", 4),
                    d_ff=m_cfg.get("d_ff", 384),
                    max_seq_len=m_cfg.get("max_seq_len", 160),
                    canvas_res=cfg.get("data", {}).get("canvas_res", m_cfg.get("canvas_res", 128)),
                )
                cls.staged_model.load_state_dict(c_data["model_state_dict"])
                cls.staged_model.eval()
                cls.staged_encoder = FrozenTextEncoder(
                    model_type=cfg.get("text_encoder", {}).get("model_type", "builtin") if isinstance(cfg.get("text_encoder"), dict) else "builtin",
                    d_model=d_m,
                )
                cls.staged_ckpt_path = ckpt_path
        return cls.staged_model, cls.staged_encoder

    def log_message(self, format, *args):
        # Keep server log concise and robust
        try:
            sys.stderr.write(f"[{self.log_date_time_string()}] {self.address_string()} - {format % args}\n")
        except Exception:
            pass

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
            gru_model, gru_ckpt, gru_classes = self.get_gru_model()
            stg_model, _ = self.get_staged_model()
            self.send_json({
                "status": "ready",
                "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else (str(self.staged_ckpt_path) if self.staged_ckpt_path else "ready"),
                "classes": gru_classes or self.classes or ["bed", "bicycle", "car", "chair", "house", "table"],
                "hidden_dim": gru_ckpt.get("config", {}).get("hidden_dim", 256) if gru_ckpt else 256,
                "parameters": gru_model.num_parameters() if (gru_model and hasattr(gru_model, "num_parameters")) else 390000,
                "staged_model_ready": stg_model is not None,
                "staged_checkpoint": str(self.staged_ckpt_path) if self.staged_ckpt_path else None,
                "staged_parameters": sum(p.numel() for p in stg_model.parameters()) if stg_model else 0,
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

        if path == "/api/hand_demos":
            demos_dir = PROJECT_ROOT / "outputs" / "hand_demos"
            items = []
            if demos_dir.exists():
                demos_meta = [
                    ("bicycle_ink_study", "Bicycle Ink Study", "Fine ink liner with light-directed line weights, spokes, chainring, and ground hatching shadow.", "Ink Liner"),
                    ("pencil_still_life", "Pencil Still Life", "3 classical geometric forms with core shading, graphite tooth grain, and projected cast shadows.", "Pencil"),
                    ("color_landscape", "Color Landscape", "Atmospheric perspective gradient sky, 3-tier depth mountain ridges, lake reflections, and pine trees.", "Color Wash"),
                    ("charcoal_portrait", "Charcoal Portrait Study", "Loomis proportion guides, head construction, high-friction charcoal grain, and chiaroscuro value masses.", "Charcoal"),
                ]
                for piece_id, title, desc, medium in demos_meta:
                    png_file = demos_dir / f"{piece_id}.png"
                    svg_file = demos_dir / f"{piece_id}.svg"
                    if png_file.exists():
                        items.append({
                            "id": piece_id,
                            "title": title,
                            "description": desc,
                            "medium": medium,
                            "png_url": f"/outputs/hand_demos/{piece_id}.png",
                            "svg_url": f"/outputs/hand_demos/{piece_id}.svg",
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
            prefix = payload.get("prefix", None)

            if top_k is not None:
                top_k = int(top_k)

            gru_model, gru_ckpt, gru_classes = self.get_gru_model()
            if gru_model is None:
                self.send_json({"error": "DrawingGRU model checkpoint not found on server"}, status=HTTPStatus.NOT_FOUND)
                return

            classes = gru_classes or self.classes or ["bed", "bicycle", "car", "chair", "house", "table"]
            if category not in classes:
                category = classes[0] if classes else "bicycle"

            class_id = classes.index(category)
            max_len = gru_ckpt.get("config", {}).get("max_len", 160) if gru_ckpt else 160

            try:
                # Autoregressive generation (with user prefix if provided)
                if prefix and isinstance(prefix, list) and len(prefix) > 0:
                    res = generate_with_prefix(
                        model=gru_model,
                        class_id=class_id,
                        prefix_actions=prefix,
                        seed=seed,
                        max_len=max_len,
                        temperature=temperature,
                        top_k=top_k,
                        continuous_coords=continuous,
                    )
                else:
                    res = generate(
                        model=gru_model,
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
                    "user_stroke_count": res.get("user_stroke_count", 0),
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
            except Exception as e:
                self.send_json({"error": f"Generation error: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return

        if path == "/api/generate_text":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)

            try:
                payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
            except Exception:
                payload = {}

            prompt = payload.get("prompt", "a red circle above a blue square")
            seed = int(payload.get("seed", 42))
            temperature = float(payload.get("temperature", 0.5))
            top_k = int(payload.get("top_k", 25))
            prefix = payload.get("prefix", None)

            # Lazy load general model
            if not hasattr(DrawingStudioHandler, "general_model") or DrawingStudioHandler.general_model is None:
                from .general.text_encoder import FrozenTextEncoder
                from .general.transformer_decoder import KonnonTransformerDecoder
                from .general.vocab import VOCAB_SIZE

                small_ckpt = PROJECT_ROOT / "outputs_general" / "small" / "checkpoints" / "best.pt"
                tiny_ckpt = PROJECT_ROOT / "outputs_general" / "tiny" / "checkpoints" / "best.pt"
                chosen_ckpt = small_ckpt if small_ckpt.exists() else tiny_ckpt

                if chosen_ckpt.exists():
                    c_data = torch.load(chosen_ckpt, map_location="cpu")
                    cfg = c_data.get("config", {})
                    m_cfg = cfg.get("model", {})
                    g_model = KonnonTransformerDecoder(
                        vocab_size=VOCAB_SIZE,
                        d_model=m_cfg.get("d_model", 256),
                        n_layers=m_cfg.get("n_layers", 4),
                        n_heads=m_cfg.get("n_heads", 8),
                        d_ff=m_cfg.get("d_ff", 768),
                        max_seq_len=m_cfg.get("max_seq_len", 192),
                    )
                    g_model.load_state_dict(c_data["model_state_dict"])
                    g_model.eval()
                    DrawingStudioHandler.general_model = g_model
                    DrawingStudioHandler.general_encoder = FrozenTextEncoder(
                        model_type=cfg.get("text_encoder", {}).get("model_type", "builtin"),
                        d_model=m_cfg.get("d_model", 256),
                    )

            if hasattr(DrawingStudioHandler, "general_model") and DrawingStudioHandler.general_model is not None:
                from .general.sampling_general import generate_from_prompt, generate_with_prefix_prompt

                if prefix and isinstance(prefix, list) and len(prefix) > 0:
                    pref_doc = v1_to_v2(prefix)
                    gen_res = generate_with_prefix_prompt(
                        model=DrawingStudioHandler.general_model,
                        text_encoder=DrawingStudioHandler.general_encoder,
                        prompt=prompt,
                        prefix_doc=pref_doc,
                        max_tokens=160,
                        temperature=temperature,
                        top_k=top_k,
                        seed=seed,
                    )
                else:
                    gen_res = generate_from_prompt(
                        model=DrawingStudioHandler.general_model,
                        text_encoder=DrawingStudioHandler.general_encoder,
                        prompt=prompt,
                        max_tokens=160,
                        temperature=temperature,
                        top_k=top_k,
                        seed=seed,
                    )

                doc = gen_res["doc"]
                svg = export_svg_v2(doc, width=512, height=512)

                v1_actions = []
                for cmd in doc.commands:
                    if getattr(cmd, "type", "") == "stroke":
                        pts = getattr(cmd, "points", [])
                        if len(pts) >= 2:
                            v1_actions.append({"type": "MOVE", "x": pts[0].x, "y": pts[0].y})
                            for p in pts[1:]:
                                v1_actions.append({"type": "LINE", "x": p.x, "y": p.y})
                            v1_actions.append({"type": "PEN_UP"})
                if v1_actions:
                    v1_actions.append({"type": "END"})

                response_data = {
                    "prompt": prompt,
                    "seed": seed,
                    "temperature": temperature,
                    "n_tokens": gen_res["n_tokens"],
                    "actions": v1_actions,
                    "svg": svg,
                    "bezier_strokes": actions_to_bezier_segments(v1_actions, width=512, height=512) if v1_actions else [],
                }
                self.send_json(response_data)
                return

        if path == "/api/generate_staged":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            try:
                payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
            except Exception:
                payload = {}

            prompt = payload.get("prompt", "a figure standing in relaxed pose")
            stages = payload.get("stages", ["layout", "guide", "ink", "shade"])
            seed = int(payload.get("seed", 42))
            temperature = float(payload.get("temperature", 0.5))
            top_k = int(payload.get("top_k", 25))

            stg_model, stg_encoder = self.get_staged_model()
            if stg_model is None or stg_encoder is None:
                self.send_json({"error": "Staged transformer model checkpoint not found"}, status=HTTPStatus.NOT_FOUND)
                return

            gen_res = generate_staged_drawing(
                model=stg_model,
                text_encoder=stg_encoder,
                prompt=prompt,
                stages=stages,
                canvas_res=128,
                out_res=512,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
            )

            stage_history_res = []
            cum_doc = DocumentV2(layers=list(gen_res["doc"].layers))
            for h in gen_res["stage_history"]:
                stg_doc = token_sequence_to_doc(h["tokens"])
                for cmd in stg_doc.commands:
                    setattr(cmd, "stage", h["stage_name"])
                cum_doc.commands.extend(stg_doc.commands)
                stg_svg = export_svg_v2(cum_doc, width=512, height=512)
                stage_history_res.append({
                    "stage_name": h["stage_name"],
                    "token_count": len(h["tokens"]),
                    "svg": stg_svg,
                    "canvas_data_url": image_to_data_url(h["canvas_image"]),
                })

            final_svg = export_svg_v2(gen_res["doc"], width=512, height=512)

            response_data = {
                "prompt": prompt,
                "seed": seed,
                "temperature": temperature,
                "top_k": top_k,
                "total_tokens": sum(len(h["tokens"]) for h in gen_res["stage_history"]),
                "stages": stage_history_res,
                "final_svg": final_svg,
                "final_canvas_data_url": image_to_data_url(gen_res["final_image"]),
                "doc": gen_res["doc"].to_dict(),
            }
            self.send_json(response_data)
            return

        if path == "/api/compare_ab":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            try:
                payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
            except Exception:
                payload = {}

            prompt = payload.get("prompt", "a standing figure")
            seed = int(payload.get("seed", 42))
            temperature = float(payload.get("temperature", 0.5))
            top_k = int(payload.get("top_k", 25))

            stg_model, stg_encoder = self.get_staged_model()
            if stg_model is None or stg_encoder is None:
                self.send_json({"error": "Staged transformer model checkpoint not found"}, status=HTTPStatus.NOT_FOUND)
                return

            # A: Staged multi-stage generation (Layout -> Guide -> Ink -> Shade)
            staged_res = generate_staged_drawing(
                model=stg_model,
                text_encoder=stg_encoder,
                prompt=prompt,
                stages=["layout", "guide", "ink", "shade"],
                canvas_res=128,
                out_res=512,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
            )
            staged_svg = export_svg_v2(staged_res["doc"], width=512, height=512)
            staged_tokens = sum(len(h["tokens"]) for h in staged_res["stage_history"])

            # B: Unconditioned / Single-Stage baseline (Single-pass ink)
            single_res = generate_staged_drawing(
                model=stg_model,
                text_encoder=stg_encoder,
                prompt=prompt,
                stages=["ink"],
                canvas_res=128,
                out_res=512,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
            )
            single_svg = export_svg_v2(single_res["doc"], width=512, height=512)
            single_tokens = sum(len(h["tokens"]) for h in single_res["stage_history"])

            response_data = {
                "prompt": prompt,
                "seed": seed,
                "staged": {
                    "num_stages": len(staged_res["stage_history"]),
                    "stage_names": ["layout", "guide", "ink", "shade"],
                    "total_tokens": staged_tokens,
                    "svg": staged_svg,
                    "canvas_data_url": image_to_data_url(staged_res["final_image"]),
                },
                "unconditioned": {
                    "num_stages": 1,
                    "stage_names": ["ink (single-pass)"],
                    "total_tokens": single_tokens,
                    "svg": single_svg,
                    "canvas_data_url": image_to_data_url(single_res["final_image"]),
                },
                "summary": {
                    "staged_token_count": staged_tokens,
                    "unconditioned_token_count": single_tokens,
                    "multi_stage_advantage": "Multi-stage conditioning grounds ink passes on prior guide marks.",
                }
            }
            self.send_json(response_data)
            return

        if path == "/api/record_stylus_stroke":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            try:
                payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
            except Exception:
                payload = {}

            raw_pts = payload.get("points", [])
            brush = payload.get("brush", "ink_liner")
            color = payload.get("color", "#000000")
            base_width = float(payload.get("base_width", 2.0))
            stage = payload.get("stage", "sketch")

            pts = [PointV2(x=float(p["x"]), y=float(p["y"]), p=float(p.get("pressure", 1.0)), tilt=float(p.get("tilt", 0.0))) for p in raw_pts]
            doc = DocumentV2()
            if pts:
                stroke = StrokeActionV2(points=pts, brush_id=brush, color=color, base_width=base_width, stage=stage)
                doc.commands.append(stroke)

            svg = export_svg_v2(doc, width=512, height=512)
            self.send_json({
                "recorded_points": len(pts),
                "brush": brush,
                "base_width": base_width,
                "stage": stage,
                "svg": svg,
                "doc": doc.to_dict(),
            })
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
        # Fallback to outputs_qd or outputs
        for cand in [
            PROJECT_ROOT / "outputs_qd" / "checkpoints" / "best.pt",
            PROJECT_ROOT / "outputs" / "checkpoints" / "best.pt",
            PROJECT_ROOT / "outputs_qd_h384" / "checkpoints" / "best.pt",
            PROJECT_ROOT / "outputs_v2" / "scaled" / "full" / "checkpoint_best.pt",
            PROJECT_ROOT / "outputs_v2" / "laptop" / "checkpoints" / "best.pt",
        ]:
            if cand.exists():
                ckpt_path = cand
                break

    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt_raw = torch.load(str(ckpt_path), map_location="cpu")
    if "model_config" in ckpt_raw:
        from .model.gru import DrawingGRU
        model = DrawingGRU(**ckpt_raw["model_config"])
        model.load_state_dict(ckpt_raw["model_state"])
        model.eval()
        ckpt = ckpt_raw
        classes = ckpt.get("classes", ["bed", "bicycle", "car", "chair", "house", "table"])
    else:
        # Checkpoint is v2 or general. Find and load v1 GRU model for /api/generate fallback.
        model = None
        ckpt = None
        classes = ["bed", "bicycle", "car", "chair", "house", "table"]
        for cand in [
            PROJECT_ROOT / "outputs_qd" / "checkpoints" / "best.pt",
            PROJECT_ROOT / "outputs" / "checkpoints" / "best.pt",
            PROJECT_ROOT / "outputs_qd_h384" / "checkpoints" / "best.pt",
        ]:
            if cand.exists():
                try:
                    v1_raw = torch.load(str(cand), map_location="cpu")
                    if "model_config" in v1_raw:
                        from .model.gru import DrawingGRU
                        model = DrawingGRU(**v1_raw["model_config"])
                        model.load_state_dict(v1_raw["model_state"])
                        model.eval()
                        ckpt = v1_raw
                        classes = v1_raw.get("classes", ["bed", "bicycle", "car", "chair", "house", "table"])
                        print(f"Loaded v1 DrawingGRU from {cand}")
                        break
                except Exception as e:
                    print(f"Could not load v1 fallback from {cand}: {e}")

    DrawingStudioHandler.model = model
    DrawingStudioHandler.ckpt = ckpt
    DrawingStudioHandler.classes = classes
    DrawingStudioHandler.checkpoint_path = ckpt_path

    # Pre-warm staged model
    DrawingStudioHandler.get_staged_model()

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
    default_ckpt = "outputs_qd/checkpoints/best.pt" if Path("outputs_qd/checkpoints/best.pt").exists() else "outputs/checkpoints/best.pt"
    parser = argparse.ArgumentParser(description="Start the Live Drawing Studio webview server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument("--checkpoint", default=default_ckpt, help="Path to checkpoint")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
