#!/usr/bin/env python3
"""
Local HTTP API Backend for the Tauri Holographic 3D Tarot / SAR Card Studio
===========================================================================
Uses only Python's standard library (`http.server.ThreadingHTTPServer`) plus
our existing OpenCV / NumPy / Pillow / PyTorch rendering modules.

Started silently by Tauri (`src-tauri/src/main.rs`) with CREATE_NO_WINDOW on a
dynamically allocated localhost port:
    python backend_server.py --port <PORT>
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import sys
import tempfile
import threading
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw

# Ensure sibling modules and app resources resolve cleanly in both dev & frozen builds
if os.environ.get("TAROT_APP_ROOT", "").strip():
    BASE_DIR = os.path.abspath(os.environ["TAROT_APP_ROOT"].strip())
elif getattr(sys, "frozen", False):
    _exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    _parent = os.path.dirname(_exe_dir)
    if os.path.isdir(os.path.join(_parent, "templates")) or os.path.isdir(os.path.join(_parent, "models")):
        BASE_DIR = _parent
    else:
        BASE_DIR = _exe_dir
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if not getattr(sys, "frozen", False) and BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from card_templates import (
    BG_TEMPLATE_META,
    FG_TEMPLATE_META,
    STUDIO_BG_META,
    ensure_default_templates,
    load_depth_map_f32,
)
from depth_estimator import generate_depth_from_image, is_ai_depth_available
from render_sar_card import render_3layer_sar_gif, render_live_preview_frame
from render_tarot_holographic import (
    GIF_QUALITY_PRESETS,
    METAL_META,
    METAL_PALETTES,
    PATTERN_META,
    STYLE_META,
    RenderCancelledError,
    build_micro_etching_texture,
    render_tarot_gif,
)


# ---------------------------------------------------------------------------
# Shared Server State (Thread-Safe Render Job Tracking & Swatch Cache)
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.template_paths = ensure_default_templates()
        self.upload_dir = os.path.join(tempfile.gettempdir(), "tarot_sar_tauri_uploads")
        os.makedirs(self.upload_dir, exist_ok=True)

        # Render job state
        self.running = False
        self.step = 0
        self.total = 1
        self.percent = 0.0
        self.message = "就绪"
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.cancel_event = threading.Event()
        self.preview_lock = threading.Lock()

        # Precompute small data-URL swatches for styles, patterns, and metals
        self.catalog_cache = self._build_catalog()

    def _pil_to_data_url(self, im: Image.Image) -> str:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _make_grad_swatch(self, w: int, h: int, c1, c2) -> str:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        t = np.clip((xx / max(1, w - 1)) * 0.55 + (yy / max(1, h - 1)) * 0.45, 0.0, 1.0)[..., None]
        arr = np.array(c1, dtype=np.float32) * (1.0 - t) + np.array(c2, dtype=np.float32) * t
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(im)
        draw.rectangle([2, 2, w - 3, h - 3], outline=(255, 244, 214), width=1)
        return self._pil_to_data_url(im)

    def _make_metal_swatch(self, key: str, w: int = 96, h: int = 48) -> str:
        if key == "auto":
            return self._make_grad_swatch(w, h, (92, 84, 128), (225, 198, 132))
        m_base, m_bright, m_shadow = METAL_PALETTES[key]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        diag = (xx / max(1, w - 1)) * 0.72 + (yy / max(1, h - 1)) * 0.28
        spec = np.exp(-((diag - 0.48) ** 2) / 0.045)[..., None]
        bt = np.clip(diag * 0.9, 0.0, 1.0)[..., None]
        col = np.array(m_shadow) * (1.0 - bt) + np.array(m_base) * bt + np.array(m_bright) * spec * 0.65
        rng = np.random.default_rng(abs(hash(key)) % 9999)
        col = np.clip((col + rng.normal(0.0, 0.035, size=(h, 1, 1))) * 255.0, 0, 255).astype(np.uint8)
        return self._pil_to_data_url(Image.fromarray(col, mode="RGB"))

    def _make_pattern_swatch(self, key: str, w: int = 120, h: int = 72) -> str:
        if key == "auto":
            return self._make_grad_swatch(w, h, (52, 48, 84), (124, 102, 160))
        etch, grat = build_micro_etching_texture(h * 2, w * 2, key)
        yy, xx = np.mgrid[0:h * 2, 0:w * 2].astype(np.float32)
        phase = (xx / (w * 2)) * 4.8 + (yy / (h * 2)) * 2.6 + grat * 0.9
        irid = np.stack([
            0.55 + 0.42 * np.cos(phase),
            0.52 + 0.40 * np.cos(phase + 2.0),
            0.60 + 0.40 * np.cos(phase + 4.0),
        ], axis=-1).astype(np.float32)
        rgb = np.array([0.07, 0.07, 0.12], dtype=np.float32) + irid * (etch[..., None] * 0.82 + grat[..., None] * 0.16)
        im = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")
        return self._pil_to_data_url(im.resize((w, h), Image.Resampling.LANCZOS))

    def _build_catalog(self) -> Dict[str, Any]:
        ai_ok, ai_desc = is_ai_depth_available()

        studios = [{"key": "auto", "label": "自动匹配", "thumb": self._make_grad_swatch(80, 110, (55, 62, 112), (168, 140, 96))}]
        for key, label in STUDIO_BG_META:
            studios.append({"key": key, "label": label, "thumb": f"/api/template/studio/{key}"})
        studios.append({"key": "obsidian_dark", "label": "经典曜石暗黑", "thumb": self._make_grad_swatch(80, 110, (14, 14, 22), (34, 32, 46))})

        bgs = [{"key": "auto", "label": "自动匹配", "thumb": self._make_grad_swatch(80, 120, (48, 44, 78), (144, 120, 88))}]
        for key, label in BG_TEMPLATE_META:
            short = label.split("(")[0].strip()
            bgs.append({"key": key, "label": short, "thumb": f"/api/template/bg/{key}"})

        fgs = [{"key": "auto", "label": "自动匹配", "thumb": self._make_grad_swatch(80, 120, (38, 36, 56), (178, 150, 92))}]
        for key, label in FG_TEMPLATE_META:
            short = label.split("·")[-1].split("(")[0].strip()
            fgs.append({"key": key, "label": short, "thumb": f"/api/template/fg/{key}"})
        fgs.append({"key": "none", "label": "无前景框", "thumb": self._make_grad_swatch(80, 120, (26, 24, 36), (42, 40, 56))})

        styles = [
            {"key": k, "label": lbl, "thumb": self._make_grad_swatch(110, 62, c1, c2)}
            for k, lbl, c1, c2 in STYLE_META
        ]

        patterns = [
            {"key": k, "label": lbl, "thumb": self._make_pattern_swatch(k)}
            for k, lbl in PATTERN_META
        ]

        metals = [
            {"key": k, "label": lbl, "thumb": self._make_metal_swatch(k, w=110, h=62)}
            for k, lbl in METAL_META
        ]

        return {
            "ai_depth_available": ai_ok,
            "ai_depth_desc": ai_desc,
            "templates_dir": os.path.join(BASE_DIR, "templates"),
            "default_output_dir": BASE_DIR,
            "studios": studios,
            "bgs": bgs,
            "fgs": fgs,
            "styles": styles,
            "patterns": patterns,
            "metals": metals,
        }


STATE = AppState()


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------

class APIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        # Quiet logging in production desktop mode
        pass

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, content_type: str, cache_control: str = "no-cache") -> None:
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", cache_control)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path == "/api/health":
            ai_ok, ai_desc = is_ai_depth_available()
            self._send_json({"ok": True, "ai_depth": ai_ok, "ai_desc": ai_desc})
            return

        if path == "/api/catalog":
            self._send_json(STATE.catalog_cache)
            return

        if path == "/api/render/status":
            with STATE.lock:
                self._send_json({
                    "running": STATE.running,
                    "step": STATE.step,
                    "total": STATE.total,
                    "percent": round(STATE.percent, 1),
                    "message": STATE.message,
                    "result": STATE.result,
                    "error": STATE.error,
                })
            return

        if path.startswith("/api/template/"):
            parts = path.split("/")
            if len(parts) == 5:
                kind, key = parts[3], parts[4]
                tpl_map = STATE.template_paths.get(kind, {})
                fpath = tpl_map.get(key)
                if fpath and os.path.isfile(fpath):
                    im = Image.open(fpath)
                    if kind == "fg":
                        fg = im.convert("RGBA").resize((120, 192), Image.Resampling.LANCZOS)
                        dark = Image.new("RGBA", fg.size, (22, 20, 34, 255))
                        out = Image.alpha_composite(dark, fg).convert("RGB")
                    else:
                        out = im.convert("RGB").resize((120, 180), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    out.save(buf, format="PNG")
                    self._send_bytes(buf.getvalue(), "image/png", cache_control="public, max-age=3600")
                    return
            self._send_json({"error": "Template not found"}, HTTPStatus.NOT_FOUND)
            return

        if path == "/api/file":
            fpath = qs.get("path", [""])[0]
            as_depth = qs.get("as_depth", ["0"])[0] == "1"
            if fpath and os.path.isfile(fpath):
                if as_depth:
                    try:
                        d_f32, _ = load_depth_map_f32(fpath)
                        d_max = float(d_f32.max())
                        if d_max > 1e-5:
                            d_vis = np.clip(d_f32 / d_max * 255.0 + 0.5, 0, 255).astype(np.uint8)
                        else:
                            d_vis = np.zeros(d_f32.shape, dtype=np.uint8)
                        buf = io.BytesIO()
                        Image.fromarray(d_vis, mode="L").save(buf, format="PNG")
                        self._send_bytes(buf.getvalue(), "image/png")
                        return
                    except Exception:
                        pass
                ctype, _ = mimetypes.guess_type(fpath)
                with open(fpath, "rb") as f:
                    data = f.read()
                self._send_bytes(data, ctype or "application/octet-stream")
                return
            self._send_json({"error": "File not found"}, HTTPStatus.NOT_FOUND)
            return

        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        try:
            body = self._read_json_body()
        except Exception as exc:
            self._send_json({"error": f"Invalid JSON: {exc}"}, HTTPStatus.BAD_REQUEST)
            return

        if path == "/api/upload":
            # Save drag-and-drop base64 image cleanly to temp uploads folder
            filename = os.path.basename(str(body.get("filename") or "dropped.png"))
            data_b64 = str(body.get("data_base64") or "")
            if "," in data_b64:
                data_b64 = data_b64.split(",", 1)[1]
            if not data_b64:
                self._send_json({"error": "Empty upload"}, HTTPStatus.BAD_REQUEST)
                return
            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
            dest = os.path.join(STATE.upload_dir, safe_name)
            with open(dest, "wb") as f:
                f.write(base64.b64decode(data_b64))
            self._send_json({"path": dest, "filename": safe_name})
            return

        if path == "/api/preview":
            try:
                with STATE.preview_lock:
                    subj = body.get("subject_path") or None
                    if subj and not os.path.isfile(subj):
                        subj = None
                    res = render_live_preview_frame(
                        subject_path=subj,
                        angle_preset=str(body.get("angle_preset", "right")),
                        preview_width=int(body.get("preview_width", 340)),
                        bg_path=body.get("bg_path") or None,
                        fg_path=body.get("fg_path") or None,
                        depth_path=body.get("depth_path") or None,
                        studio_bg_path=body.get("studio_bg_path") or None,
                        bg_template=str(body.get("bg_template", "auto")),
                        fg_template=str(body.get("fg_template", "auto")),
                        studio_bg=str(body.get("studio_bg", "auto")),
                        style=str(body.get("style", "auto")),
                        pattern=str(body.get("pattern", "auto")),
                        metal=str(body.get("metal", "auto")),
                        intensity=str(body.get("intensity", "medium")),
                        bg_mode=str(body.get("bg_mode", "auto")),
                        depth_engine=str(body.get("depth_engine", "auto")),
                        line_etch=str(body.get("line_etch", "medium")),
                        quicksand_strength=float(body.get("quicksand_strength", 1.0)),
                        relief_strength=float(body.get("relief_strength", 1.0)),
                        enable_dof=bool(body.get("enable_dof", True)),
                        subject_scale=float(body.get("subject_scale", 1.0)),
                        subject_offset_x=float(body.get("subject_offset_x", 0.0)),
                        subject_offset_y=float(body.get("subject_offset_y", 0.0)),
                        card_aspect=str(body.get("card_aspect", "auto")),
                    )
                im: Image.Image = res["preview_image"]
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                self._send_json({
                    "image_data_url": f"data:image/png;base64,{b64}",
                    "style": res["style"],
                    "pattern": res["pattern"],
                    "metal": res["metal"],
                    "bg_source": res["bg_source"],
                    "fg_source": res["fg_source"],
                    "studio_source": res["studio_source"],
                    "depth_source": res["depth_source"],
                    "subject_size": list(res.get("subject_size", (0, 0))),
                    "output_size": list(res.get("output_size", (0, 0))),
                })
            except Exception as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if path == "/api/depth":
            subj = str(body.get("subject_path") or "")
            if not subj or not os.path.isfile(subj):
                self._send_json({"error": "请先选择有效的中景主体图片"}, HTTPStatus.BAD_REQUEST)
                return
            engine = str(body.get("depth_engine", "auto"))
            bg_mode = str(body.get("bg_mode", "auto"))
            out_depth = f"{os.path.splitext(subj)[0]}_depth.png"
            try:
                _, used = generate_depth_from_image(subj, output_path=out_depth, engine=engine, bg_mode=bg_mode)
                self._send_json({"depth_path": os.path.abspath(out_depth), "engine": used})
            except Exception as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if path == "/api/render/start":
            with STATE.lock:
                if STATE.running:
                    self._send_json({"error": "已有渲染任务正在运行"}, HTTPStatus.CONFLICT)
                    return
                subj = str(body.get("subject_path") or "")
                if not subj or not os.path.isfile(subj):
                    self._send_json({"error": "请先选择有效的中景主体图片"}, HTTPStatus.BAD_REQUEST)
                    return
                output_path = str(body.get("output_path") or "").strip()
                if not output_path:
                    output_path = f"{os.path.splitext(subj)[0]}_holographic_card.gif"
                out_dir = os.path.dirname(os.path.abspath(output_path))
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)

                STATE.running = True
                STATE.step = 0
                STATE.total = int(body.get("n_frames", 48)) + 3
                STATE.percent = 0.0
                STATE.message = "正在初始化渲染管线..."
                STATE.result = None
                STATE.error = None
                STATE.cancel_event.clear()

            threading.Thread(target=self._run_render_job, args=(body, output_path), daemon=True).start()
            self._send_json({"started": True, "output_path": os.path.abspath(output_path)})
            return

        if path == "/api/render/cancel":
            STATE.cancel_event.set()
            with STATE.lock:
                if STATE.running:
                    STATE.message = "正在取消渲染..."
            self._send_json({"cancelling": True})
            return

        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def _run_render_job(self, body: Dict[str, Any], output_path: str) -> None:
        def _progress(step: int, total: int, msg: str) -> None:
            with STATE.lock:
                STATE.step = step
                STATE.total = max(1, total)
                STATE.percent = (step / max(1, total)) * 100.0
                STATE.message = msg

        try:
            render_mode = str(body.get("render_mode", "3layer_sar"))
            common = {
                "output_path": output_path,
                "studio_bg": str(body.get("studio_bg", "auto")),
                "studio_bg_path": body.get("studio_bg_path") or None,
                "style": str(body.get("style", "auto")),
                "pattern": str(body.get("pattern", "auto")),
                "metal": str(body.get("metal", "auto")),
                "intensity": str(body.get("intensity", "medium")),
                "bg_mode": str(body.get("bg_mode", "auto")),
                "line_etch": str(body.get("line_etch", "medium")),
                "subject_scale": float(body.get("subject_scale", 1.0)),
                "subject_offset_x": float(body.get("subject_offset_x", 0.0)),
                "subject_offset_y": float(body.get("subject_offset_y", 0.0)),
                "card_aspect": str(body.get("card_aspect", "auto")),
                "quality_tier": str(body.get("quality_tier", "hd")),
                "final_width": int(body.get("final_width", 480)),
                "n_frames": int(body.get("n_frames", 48)),
                "supersample": int(body.get("supersample", 2)),
                "progress_callback": _progress,
                "cancel_event": STATE.cancel_event,
            }
            if render_mode == "3layer_sar":
                info = render_3layer_sar_gif(
                    subject_path=str(body["subject_path"]),
                    bg_path=body.get("bg_path") or None,
                    fg_path=body.get("fg_path") or None,
                    depth_path=body.get("depth_path") or None,
                    bg_template=str(body.get("bg_template", "auto")),
                    fg_template=str(body.get("fg_template", "auto")),
                    depth_engine=str(body.get("depth_engine", "auto")),
                    quicksand_strength=float(body.get("quicksand_strength", 1.0)),
                    relief_strength=float(body.get("relief_strength", 1.0)),
                    enable_dof=bool(body.get("enable_dof", True)),
                    **common,
                )
            else:
                info = render_tarot_gif(input_path=str(body["subject_path"]), **common)

            with STATE.lock:
                STATE.running = False
                STATE.percent = 100.0
                STATE.message = "渲染完成！"
                STATE.result = info
        except RenderCancelledError:
            with STATE.lock:
                STATE.running = False
                STATE.percent = 0.0
                STATE.message = "渲染已取消"
                STATE.error = None
        except Exception as exc:
            with STATE.lock:
                STATE.running = False
                STATE.percent = 0.0
                STATE.message = "渲染失败"
                STATE.error = str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local HTTP API server for Tauri Holographic Card Studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), APIHandler)
    print(f"TAROT_BACKEND_READY http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
