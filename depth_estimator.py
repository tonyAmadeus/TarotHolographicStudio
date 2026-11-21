#!/usr/bin/env python3
"""
2D -> Monocular Depth Map Generator (Fast Procedural + Optional PyTorch CUDA AI)
================================================================================
Generates a clean, smooth grayscale monocular depth map (pure black background,
white = near, black = far) directly from a 2D image for 3D Relief Mapping.

Engines:
  - "fast": Pure OpenCV + NumPy multi-cue 2D->3D depth synthesis (offline, <0.1s).
            Combines silhouette distance-transform volume, anatomical/face
            foreground hierarchy, and smoothed occlusion/shading layering.
  - "ai":   Real deep-learning monocular depth estimation on PyTorch (uses CUDA
            GPU automatically when available) via:
              1) Depth-Anything-V2-Small (if `transformers` is installed)
              2) Intel MiDaS DPT_Hybrid  (if `timm` is installed)
  - "auto": Uses "ai" if available, otherwise seamlessly uses "fast".

CLI Usage:
    python depth_estimator.py --input character.webp --output character_depth.png --engine fast
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


def find_large_model_path() -> Optional[str]:
    """Locate locally downloaded `Depth-Anything-V2-Large-hf` directory."""
    candidates = []
    env_dir = os.environ.get("TAROT_LARGE_DEPTH_DIR", "").strip()
    if env_dir:
        candidates.append(env_dir)
    candidates.extend([
        r"D:\AI\AllModel\Depth-Anything-V2-Large-hf",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Depth-Anything-V2-Large-hf"),
    ])
    for d in candidates:
        if d and os.path.isdir(d) and os.path.isfile(os.path.join(d, "config.json")):
            if os.path.isfile(os.path.join(d, "model.safetensors")) or os.path.isfile(os.path.join(d, "pytorch_model.bin")):
                return os.path.abspath(d)
    return None


def find_onnx_model_path() -> Optional[str]:
    """Locate bundled or local `models/depth_anything_v2_small.onnx`."""
    candidates = []
    env_model = os.environ.get("TAROT_ONNX_MODEL", "").strip()
    if env_model:
        candidates.append(env_model)
    env_root = os.environ.get("TAROT_APP_ROOT", "").strip()
    if env_root:
        candidates.append(os.path.join(env_root, "models", "depth_anything_v2_small.onnx"))
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        candidates.append(os.path.join(exe_dir, "models", "depth_anything_v2_small.onnx"))
        candidates.append(os.path.join(os.path.dirname(exe_dir), "models", "depth_anything_v2_small.onnx"))
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(os.path.join(meipass, "models", "depth_anything_v2_small.onnx"))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "models", "depth_anything_v2_small.onnx"))
    for p in candidates:
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    return None


def _is_onnx_depth_ready() -> bool:
    if find_onnx_model_path() is None:
        return False
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def is_ai_depth_available() -> Tuple[bool, str]:
    """Check whether ONNX Runtime (lightweight) or PyTorch is available for AI depth."""
    onnx_ready = _is_onnx_depth_ready()
    if getattr(sys, "frozen", False) and onnx_ready:
        return True, "Depth-Anything-V2 (ONNX 轻量 AI)"

    # In dev environments with NVIDIA CUDA GPU, report CUDA GPU first
    try:
        import torch
        if torch.cuda.is_available():
            try:
                import transformers  # noqa: F401
                if find_large_model_path():
                    return True, "Depth-Anything-V2-Large + Small (CUDA GPU)"
                return True, "Depth-Anything-V2 (CUDA GPU)"
            except ImportError:
                pass
            try:
                import timm  # noqa: F401
                return True, "MiDaS (CUDA GPU)"
            except ImportError:
                pass
    except ImportError:
        pass

    if onnx_ready:
        return True, "Depth-Anything-V2 (ONNX 轻量 AI)"

    try:
        import torch  # noqa: F401
        try:
            import transformers  # noqa: F401
            return True, "Depth-Anything-V2 (CPU)"
        except ImportError:
            pass
        try:
            import timm  # noqa: F401
            return True, "MiDaS (CPU)"
        except ImportError:
            pass
    except ImportError:
        pass

    return False, "未检测到 ONNX 模型或 PyTorch（将使用离线导向滤波算法）"


def _guided_filter_gray(guide: np.ndarray, src: np.ndarray, radius: int, eps: float = 0.008) -> np.ndarray:
    """Fast O(N) edge-preserving Guided Image Filter (He et al.) using OpenCV boxFilter."""
    k = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(guide, cv2.CV_32F, k)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, k)
    mean_Ip = cv2.boxFilter(guide * src, cv2.CV_32F, k)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, k)
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, k)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, k)
    return (mean_a * guide + mean_b).astype(np.float32)


def _guided_filter_color(
    guide_rgb: np.ndarray,
    src: np.ndarray,
    radius: int,
    eps: float = 0.0022,
) -> np.ndarray:
    """
    Fast O(N) 3-channel CIE Lab Color Guided Image Filter (He & Sun):
    locks depth edges onto BOTH luminance and isoluminant hue/chroma boundaries
    (e.g. pink sleeve over light-blue wing, peach skin next to white hair).
    """
    lab = cv2.cvtColor(np.clip(guide_rgb, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2LAB)
    I0 = (lab[..., 0] * np.float32(0.01)).astype(np.float32)
    I1 = ((lab[..., 1] + np.float32(128.0)) * np.float32(0.00333)).astype(np.float32)
    I2 = ((lab[..., 2] + np.float32(128.0)) * np.float32(0.00333)).astype(np.float32)
    p = src.astype(np.float32)

    k = (2 * radius + 1, 2 * radius + 1)
    m0 = cv2.boxFilter(I0, cv2.CV_32F, k)
    m1 = cv2.boxFilter(I1, cv2.CV_32F, k)
    m2 = cv2.boxFilter(I2, cv2.CV_32F, k)
    mp = cv2.boxFilter(p, cv2.CV_32F, k)

    e = np.float32(eps)
    v00 = cv2.boxFilter(I0 * I0, cv2.CV_32F, k) - m0 * m0 + e
    v01 = cv2.boxFilter(I0 * I1, cv2.CV_32F, k) - m0 * m1
    v02 = cv2.boxFilter(I0 * I2, cv2.CV_32F, k) - m0 * m2
    v11 = cv2.boxFilter(I1 * I1, cv2.CV_32F, k) - m1 * m1 + e
    v12 = cv2.boxFilter(I1 * I2, cv2.CV_32F, k) - m1 * m2
    v22 = cv2.boxFilter(I2 * I2, cv2.CV_32F, k) - m2 * m2 + e

    c0 = cv2.boxFilter(I0 * p, cv2.CV_32F, k) - m0 * mp
    c1 = cv2.boxFilter(I1 * p, cv2.CV_32F, k) - m1 * mp
    c2 = cv2.boxFilter(I2 * p, cv2.CV_32F, k) - m2 * mp

    cof00 = v11 * v22 - v12 * v12
    cof01 = v02 * v12 - v01 * v22
    cof02 = v01 * v12 - v02 * v11
    cof11 = v00 * v22 - v02 * v02
    cof12 = v01 * v02 - v00 * v12
    cof22 = v00 * v11 - v01 * v01

    det = np.maximum(v00 * cof00 + v01 * cof01 + v02 * cof02, np.float32(1e-10))
    inv_det = np.float32(1.0) / det

    a0 = (cof00 * c0 + cof01 * c1 + cof02 * c2) * inv_det
    a1 = (cof01 * c0 + cof11 * c1 + cof12 * c2) * inv_det
    a2 = (cof02 * c0 + cof12 * c1 + cof22 * c2) * inv_det
    b = mp - a0 * m0 - a1 * m1 - a2 * m2

    ma0 = cv2.boxFilter(a0, cv2.CV_32F, k)
    ma1 = cv2.boxFilter(a1, cv2.CV_32F, k)
    ma2 = cv2.boxFilter(a2, cv2.CV_32F, k)
    mb = cv2.boxFilter(b, cv2.CV_32F, k)
    return (ma0 * I0 + ma1 * I1 + ma2 * I2 + mb).astype(np.float32)


def estimate_depth_fast(
    subject_rgb: np.ndarray,
    subject_alpha: np.ndarray,
) -> np.ndarray:
    """
    Upgraded multi-cue offline 2D -> smooth monocular depth map (pure OpenCV/NumPy):
      1. Two-scale Euclidean distance-transform 3D volume (plump torso + distinct limbs)
      2. Skin/face-anchored anatomical & vertical perspective hierarchy
      3. Ink-removed CIE Lab lightness + warm-tone shading recession cue (separates
         bright lit foreground garments/hands/face from shaded rear wings/hair/tail)
      4. Color-Guided edge-aware filtering (`_guided_filter_gray`) so depth snaps
         cleanly along real clothing, sleeve, limb & wing boundaries rather than
         looking like a blurry blob
    Returns float32 depth map in [0, 1] (white=near, black=far, pure black outside).
    """
    h, w = subject_alpha.shape[:2]
    bin_mask = (subject_alpha > 0.30).astype(np.uint8)
    if not np.any(bin_mask):
        return np.zeros((h, w), dtype=np.float32)

    # 1. Two-scale 3D core & limb volume
    dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5).astype(np.float32)
    max_d = max(1.0, float(np.percentile(dist[bin_mask > 0], 98)))
    core_vol = np.clip(dist / max_d, 0.0, 1.0) ** 0.58
    limb_vol = np.clip(dist / max(1.0, max_d * 0.32), 0.0, 1.0) ** 0.45

    # 2. Subject bounding box & upper-skin face/head anchor
    ys, xs = np.where(bin_mask > 0)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    bw = max(1.0, float(x_max - x_min))
    bh = max(1.0, float(y_max - y_min))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u_rel = (xx - x_min) / bw
    v_rel = (yy - y_min) / bh

    u8 = np.clip(subject_rgb * 255.0, 0, 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    skin = (
        np.exp(-0.5 * ((ycrcb[..., 1] - 148.0) / 15.0) ** 2)
        * np.exp(-0.5 * ((ycrcb[..., 2] - 110.0) / 15.0) ** 2)
        * (bin_mask > 0).astype(np.float32)
    )
    upper_skin = skin * (v_rel < 0.38).astype(np.float32)
    if float(np.sum(upper_skin)) > 25.0:
        cx_u = float(np.sum(u_rel * upper_skin) / np.sum(upper_skin))
        cy_head = float(np.sum(v_rel * upper_skin) / np.sum(upper_skin))
    else:
        cx_u = float(np.mean(u_rel[bin_mask > 0]))
        cy_head = 0.20

    head_dome = np.exp(-(((u_rel - cx_u) / 0.22) ** 2 + ((v_rel - cy_head) / 0.14) ** 2))
    chest_dome = np.exp(-(((u_rel - cx_u) / 0.26) ** 2 + ((v_rel - min(0.75, cy_head + 0.22)) / 0.22) ** 2))
    # Gentle ground-plane perspective: front lower skirt/feet slightly nearer than rear wings
    ground_tilt = np.clip((v_rel - 0.35) / 0.65, 0.0, 1.0) ** 1.15 * np.exp(-(((u_rel - cx_u) / 0.42) ** 2))
    # Horizontal recession: outer wings / trailing hair / tail recede behind central body
    horiz_depth = np.clip(1.0 - (np.abs(u_rel - cx_u) ** 1.20) * 0.98, 0.22, 1.0)

    # 3. Ink-cleaned CIE Lab Lightness + Warmth shading depth cue
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L_norm = lab[..., 0] / 255.0
    a_warm = (lab[..., 1] - 128.0) / 128.0  # warm pink/skin/gold > 0, cool shadow/blue-grey < 0
    k_ink = max(5, int(round(w * 0.013)) | 1)
    k_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_ink, k_ink))
    L_clean = cv2.morphologyEx(L_norm, cv2.MORPH_CLOSE, k_el)
    # Multi-scale shading: fine-medium preserves dress folds & sleeves, macro separates shaded back parts
    L_med = cv2.GaussianBlur(L_clean, (0, 0), sigmaX=max(3.5, w * 0.009))
    L_mac = cv2.GaussianBlur(L_clean, (0, 0), sigmaX=max(9.0, w * 0.026))
    shading_raw = L_med * 0.52 + L_mac * 0.48 + np.clip(a_warm * 1.4, -0.25, 0.35) * 0.28 + skin * 0.14

    s_fg = shading_raw[bin_mask > 0]
    s_lo, s_hi = float(np.percentile(s_fg, 4)), float(np.percentile(s_fg, 96))
    shading_cue = np.clip((shading_raw - s_lo) / max(1e-4, s_hi - s_lo), 0.0, 1.0)

    # Combine anatomical + shading cues with stronger depth contrast
    raw_depth = (
        0.10
        + 0.24 * core_vol
        + 0.15 * limb_vol
        + 0.18 * head_dome
        + 0.14 * chest_dome
        + 0.16 * ground_tilt
        + 0.36 * shading_cue
    ) * (0.36 + 0.64 * horiz_depth)

    # 4. Edge-preserving Guided Filter using ink-cleaned luminance as guide
    #    Snaps depth cleanly to sleeve/dress/hand/wing/tail contours!
    guide = np.where(bin_mask > 0, cv2.GaussianBlur(L_clean, (0, 0), sigmaX=1.5), 0.0).astype(np.float32)
    raw_fg_only = np.where(bin_mask > 0, raw_depth, float(np.percentile(raw_depth[bin_mask > 0], 10))).astype(np.float32)
    guided_d = _guided_filter_gray(guide, raw_fg_only, radius=max(8, int(round(w * 0.024))), eps=0.004)

    # Dynamic range stretch + subtle S-curve for rich, unmistakable 3D separation
    fg_vals = guided_d[bin_mask > 0]
    d_lo, d_hi = float(np.percentile(fg_vals, 2)), float(np.percentile(fg_vals, 98.5))
    norm_d = np.clip((guided_d - d_lo) / max(1e-4, d_hi - d_lo), 0.0, 1.0)
    norm_d = norm_d * norm_d * (3.0 - 2.0 * norm_d) * 0.45 + norm_d * 0.55
    norm_d = 0.10 + 0.88 * norm_d

    depth = np.where(subject_alpha > 0.20, norm_d, 0.0).astype(np.float32)
    depth = cv2.bilateralFilter(depth, d=9, sigmaColor=0.18, sigmaSpace=max(5.0, w * 0.012))
    depth = np.where(subject_alpha > 0.12, depth, 0.0).astype(np.float32)
    return np.clip(depth, 0.0, 1.0).astype(np.float32)


_AI_DEPTH_CACHE: dict = {}


def _postprocess_ai_depth(raw_depth: np.ndarray, subject_alpha: np.ndarray) -> np.ndarray:
    """
    Losslessly normalize raw float32 AI depth inside subject silhouette without
    cross-zero bilateral edge bleeding (keeps outer silhouette rim crisp!).
    """
    h, w = subject_alpha.shape[:2]
    valid_fg = subject_alpha > 0.04
    fg_stats = subject_alpha > 0.15
    if np.any(fg_stats):
        fg_vals = raw_depth[fg_stats]
        lo, hi = float(np.percentile(fg_vals, 1.5)), float(np.percentile(fg_vals, 98.5))
        norm = np.clip((raw_depth - lo) / max(1e-5, hi - lo), 0.0, 1.0)
        norm = (0.06 + 0.94 * norm).astype(np.float32)
    else:
        lo, hi = float(raw_depth.min()), float(raw_depth.max())
        norm = np.clip((raw_depth - lo) / max(1e-5, hi - lo), 0.0, 1.0).astype(np.float32)

    # Extrapolate valid interior depth outward BEFORE light bilateral smoothing so
    # transparent background 0.0 NEVER bleeds inward across silhouette contours!
    if np.any(fg_stats) and not np.all(valid_fg):
        inv = (~fg_stats).astype(np.uint8)
        _, labels = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        coords = np.argwhere(fg_stats)
        idx = np.clip(labels - 1, 0, len(coords) - 1)
        d_ext = np.where(fg_stats, norm, norm[coords[idx, 0], coords[idx, 1]]).astype(np.float32)
    else:
        d_ext = norm

    d_clean = cv2.bilateralFilter(d_ext, d=5, sigmaColor=0.06, sigmaSpace=max(3.0, w * 0.006))
    depth = np.where(valid_fg, np.clip(d_clean, 0.06, 1.0), 0.0).astype(np.float32)
    return depth


def estimate_depth_onnx(
    subject_rgb: np.ndarray,
    subject_alpha: np.ndarray,
) -> np.ndarray:
    """
    Lightweight PyTorch-free AI monocular depth estimation via ONNX Runtime
    using the bundled Depth-Anything-V2-Small ONNX model (518x518, ~0.36s CPU).
    """
    import onnxruntime as ort

    model_path = find_onnx_model_path()
    if not model_path:
        raise FileNotFoundError("未找到 depth_anything_v2_small.onnx 模型文件")

    h, w = subject_rgb.shape[:2]
    neutral_bg = np.array([0.08, 0.08, 0.10], dtype=np.float32)
    sa = subject_alpha[..., None]
    comp_rgb = np.clip(subject_rgb * sa + neutral_bg[None, None, :] * (1.0 - sa), 0.0, 1.0).astype(np.float32)

    # Exact Depth-Anything-V2 ImageNet preprocessing (518x518 bicubic)
    resized = cv2.resize(comp_rgb, (518, 518), interpolation=cv2.INTER_CUBIC)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_in = (resized - mean[None, None, :]) / std[None, None, :]
    tensor_in = np.transpose(norm_in, (2, 0, 1))[None, ...].astype(np.float32)

    if "onnx" not in _AI_DEPTH_CACHE:
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _AI_DEPTH_CACHE["onnx"] = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
    sess = _AI_DEPTH_CACHE["onnx"]
    in_name = sess.get_inputs()[0].name
    pred = sess.run(None, {in_name: tensor_in})[0]
    pred_2d = np.squeeze(pred).astype(np.float32)
    raw_depth = cv2.resize(pred_2d, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    return _postprocess_ai_depth(raw_depth, subject_alpha)


def estimate_depth_ai(
    subject_rgb: np.ndarray,
    subject_alpha: np.ndarray,
    model_variant: str = "auto",
) -> np.ndarray:
    """
    Real deep-learning monocular depth estimation:
      - model_variant="large": uses local Depth-Anything-V2-Large-hf (`D:\\AI\\AllModel\\Depth-Anything-V2-Large-hf`)
        on CUDA GPU (or CPU fallback).
      - model_variant="small": uses Depth-Anything-V2-Small (CUDA GPU or ONNX Runtime).
      - model_variant="auto": uses Large on CUDA GPU if locally installed, otherwise Small / ONNX.
    """
    onnx_ready = _is_onnx_depth_ready()
    if getattr(sys, "frozen", False) and onnx_ready and model_variant != "large":
        return estimate_depth_onnx(subject_rgb, subject_alpha)

    use_pytorch_cuda = False
    try:
        import torch
        use_pytorch_cuda = bool(torch.cuda.is_available())
    except ImportError:
        torch = None  # type: ignore[assignment]

    if model_variant != "large" and not use_pytorch_cuda and onnx_ready:
        return estimate_depth_onnx(subject_rgb, subject_alpha)

    if torch is None:
        if onnx_ready:
            return estimate_depth_onnx(subject_rgb, subject_alpha)
        raise RuntimeError("AI 单目深度估计需要 ONNX Runtime 模型或 PyTorch + transformers。")

    h, w = subject_rgb.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    neutral_bg = np.array([0.08, 0.08, 0.10], dtype=np.float32)
    sa = subject_alpha[..., None]
    comp_rgb = subject_rgb * sa + neutral_bg[None, None, :] * (1.0 - sa)
    pil_in = Image.fromarray(np.clip(comp_rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")

    raw_depth: Optional[np.ndarray] = None

    # Resolve target HuggingFace / local directory
    large_local = find_large_model_path()
    want_large = (model_variant == "large") or (model_variant == "auto" and use_pytorch_cuda and large_local is not None)

    # Priority 1: Depth-Anything-V2 (Large or Small) via HuggingFace transformers
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        if want_large:
            cache_key = "dav2_large"
            source = large_local if large_local else "depth-anything/Depth-Anything-V2-Large-hf"
        else:
            cache_key = "dav2_small"
            source = "depth-anything/Depth-Anything-V2-Small-hf"

        if cache_key not in _AI_DEPTH_CACHE:
            try:
                processor = AutoImageProcessor.from_pretrained(source, local_files_only=True)
                model = AutoModelForDepthEstimation.from_pretrained(source, local_files_only=True).to(device)
            except Exception:
                processor = AutoImageProcessor.from_pretrained(source)
                model = AutoModelForDepthEstimation.from_pretrained(source).to(device)
            model.eval()
            _AI_DEPTH_CACHE[cache_key] = (processor, model)
        processor, model = _AI_DEPTH_CACHE[cache_key]

        inputs = processor(images=pil_in, return_tensors="pt").to(device)
        with torch.no_grad():
            if device == "cuda" and want_large:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            pred = outputs.predicted_depth.float()
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
            ).squeeze()
        raw_depth = pred.detach().cpu().numpy().astype(np.float32)
    except ImportError:
        pass

    # Priority 2: ONNX Runtime fallback
    if raw_depth is None and onnx_ready:
        return estimate_depth_onnx(subject_rgb, subject_alpha)

    # Priority 3: Intel MiDaS via torch.hub (requires timm)
    if raw_depth is None:
        try:
            import timm  # noqa: F401
            midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True).to(device)
            midas.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            transform = midas_transforms.dpt_transform
            img_np = np.clip(subject_rgb * 255.0, 0, 255).astype(np.uint8)
            inp = transform(img_np).to(device)
            with torch.no_grad():
                pred = midas(inp)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
                ).squeeze()
            raw_depth = pred.detach().cpu().numpy().astype(np.float32)
        except ImportError as exc:
            raise RuntimeError(
                "AI 单目深度估计需要内置 ONNX 模型或安装 transformers。"
            ) from exc

    return _postprocess_ai_depth(raw_depth, subject_alpha)


def generate_depth_from_image(
    input_path: str,
    output_path: Optional[str] = None,
    engine: str = "fast",
    bg_mode: str = "auto",
) -> Tuple[np.ndarray, str]:
    """
    Generate a smooth grayscale monocular depth map from any 2D image path.
    engine: "fast" | "ai" | "auto"
    Returns (depth_f32, used_engine_label).
    """
    from render_tarot_holographic import extract_subject_alpha

    pil_img = ImageOps.exif_transpose(Image.open(input_path))
    rgb, alpha, _ = extract_subject_alpha(pil_img, bg_mode=bg_mode)

    used_engine = "fast"
    if engine == "ai_large":
        depth = estimate_depth_ai(rgb, alpha, model_variant="large")
        used_engine = "ai_large"
    elif engine == "ai":
        depth = estimate_depth_ai(rgb, alpha, model_variant="small")
        used_engine = "ai_small"
    elif engine == "auto":
        ok, _ = is_ai_depth_available()
        if ok:
            try:
                depth = estimate_depth_ai(rgb, alpha, model_variant="auto")
                used_engine = "ai_large" if (not getattr(sys, "frozen", False) and find_large_model_path()) else "ai"
            except Exception:
                depth = estimate_depth_fast(rgb, alpha)
                used_engine = "fast"
        else:
            depth = estimate_depth_fast(rgb, alpha)
            used_engine = "fast"
    else:
        depth = estimate_depth_fast(rgb, alpha)
        used_engine = "fast"

    if output_path:
        # Save lossless 16-bit grayscale PNG (0..65535) to preserve full AI float32 precision
        out_u16 = np.clip(depth * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
        ok, enc = cv2.imencode(".png", out_u16)
        if ok:
            enc.tofile(output_path)
        else:
            Image.fromarray(out_u16, mode="I;16").save(output_path)

    return depth, used_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a smooth grayscale monocular depth map from a 2D image.")
    parser.add_argument("--input", required=True, help="Path to source 2D image")
    parser.add_argument("--output", default=None, help="Output grayscale depth PNG path")
    parser.add_argument("--engine", default="fast", choices=["fast", "ai", "ai_large", "auto"],
                        help="Depth engine: fast (OpenCV/NumPy), ai (Small/ONNX), ai_large (Large GPU), auto")
    parser.add_argument("--bg-mode", default="auto", choices=["auto", "remove", "keep"])
    args = parser.parse_args()

    out_path = args.output or f"{os.path.splitext(args.input)[0]}_depth.png"
    _, used = generate_depth_from_image(args.input, output_path=out_path, engine=args.engine, bg_mode=args.bg_mode)
    print(f"Depth map saved to: {os.path.abspath(out_path)} (engine={used})")


if __name__ == "__main__":
    main()
