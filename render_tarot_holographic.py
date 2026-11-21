#!/usr/bin/env python3
"""
Tarot Holographic 3D Floating GIF Renderer
==========================================
Creates a collector-grade gilded / astral-etched / iridescent-foil 3D floating
looping GIF directly from an input image using OpenCV, NumPy, and Pillow.

Dependencies:
    pip install opencv-python-headless numpy pillow
    (or opencv-python instead of opencv-python-headless)

Usage:
    python render_tarot_holographic.py --input path/to/image.jpg --output tarot_holographic_float.gif

Optional flags:
    --style     auto | classical | moon | sun | gothic | dream   (default: auto)
    --pattern   auto | geometric | cosmos | sunburst | silk_wave (default: auto)
    --metal     auto | gold | silver | champagne | violet_gold   (default: auto)
    --intensity subtle | medium | lavish                         (default: medium)
    --width     Final GIF canvas width in pixels                 (default: 460)
    --frames    Number of loop frames                            (default: 48)
    --supersample Supersampling factor for anti-aliasing         (default: 2)
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import threading

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps


# ---------------------------------------------------------------------------
# Configuration & Auto-Theme Analysis
# ---------------------------------------------------------------------------

@dataclass
class ThemeConfig:
    style_name: str
    pattern_name: str
    metal_name: str
    intensity_name: str
    intensity_scale: float
    # RGB in [0, 1]
    metal_base: np.ndarray
    metal_bright: np.ndarray
    metal_shadow: np.ndarray
    bg_center: np.ndarray
    bg_edge: np.ndarray
    halo_primary: np.ndarray
    halo_secondary: np.ndarray
    irid_phase_shift: float


METAL_PALETTES: Dict[str, Tuple[List[float], List[float], List[float]]] = {
    "gold": (
        [0.86, 0.71, 0.38],
        [1.00, 0.94, 0.74],
        [0.42, 0.30, 0.13],
    ),
    "champagne": (
        [0.88, 0.78, 0.56],
        [1.00, 0.96, 0.85],
        [0.44, 0.36, 0.24],
    ),
    "silver": (
        [0.78, 0.83, 0.90],
        [0.97, 0.99, 1.00],
        [0.34, 0.38, 0.45],
    ),
    "violet_gold": (
        [0.84, 0.70, 0.52],
        [0.98, 0.91, 0.84],
        [0.38, 0.26, 0.36],
    ),
    "rose_gold": (
        [0.90, 0.67, 0.66],
        [1.00, 0.91, 0.90],
        [0.46, 0.24, 0.28],
    ),
    "obsidian_gold": (
        [0.46, 0.42, 0.36],
        [0.98, 0.86, 0.56],
        [0.12, 0.11, 0.15],
    ),
    "crimson_gold": (
        [0.85, 0.36, 0.34],
        [1.00, 0.80, 0.64],
        [0.38, 0.10, 0.14],
    ),
    "jade_gold": (
        [0.50, 0.82, 0.72],
        [0.92, 0.99, 0.92],
        [0.18, 0.38, 0.34],
    ),
    "cyber_chrome": (
        [0.52, 0.86, 0.96],
        [0.94, 0.99, 1.00],
        [0.16, 0.34, 0.52],
    ),
    "aurora_prism": (
        [0.78, 0.74, 0.96],
        [0.98, 0.96, 1.00],
        [0.32, 0.36, 0.56],
    ),
    "antique_bronze": (
        [0.76, 0.56, 0.36],
        [0.97, 0.84, 0.64],
        [0.34, 0.21, 0.12],
    ),
}

STYLE_META: List[Tuple[str, str, Tuple[int, int, int], Tuple[int, int, int]]] = [
    ("auto", "自动匹配", (82, 76, 118), (216, 185, 120)),
    ("classical", "古典神秘", (48, 42, 66), (222, 196, 140)),
    ("moon", "月夜星象", (34, 48, 98), (192, 212, 240)),
    ("sun", "太阳圣辉", (92, 46, 36), (242, 192, 92)),
    ("gothic", "暗黑哥特", (78, 28, 62), (218, 162, 110)),
    ("dream", "梦境水光", (30, 82, 108), (168, 218, 236)),
    ("guofeng", "东方国风", (22, 74, 68), (218, 194, 128)),
    ("sakura", "绯夜樱吹", (88, 38, 72), (246, 178, 206)),
    ("cyber", "赛博霓虹", (28, 30, 88), (74, 224, 248)),
    ("prism", "幻彩棱镜", (64, 42, 112), (236, 194, 248)),
    ("nouveau", "穆夏鎏金", (64, 52, 38), (236, 188, 118)),
    ("abyss", "深渊冰霜", (18, 36, 74), (154, 216, 250)),
]

PATTERN_META: List[Tuple[str, str]] = [
    ("auto", "自动匹配"),
    ("geometric", "几何秘仪"),
    ("cosmos", "星轨 Cosmos"),
    ("sunburst", "放射日冕"),
    ("silk_wave", "水波丝绸"),
    ("prism_shard", "碎钻棱镜 SAR"),
    ("fingerprint", "指纹等高 UR"),
    ("hex_cyber", "赛博蜂巢"),
    ("guofeng_cloud", "国风祥云"),
    ("stained_glass", "玫瑰花窗"),
    ("sakura_petal", "樱花花雨"),
    ("star_lattice", "十字星闪格"),
]

METAL_META: List[Tuple[str, str]] = [
    ("auto", "自动匹配"),
    ("champagne", "香槟浅金"),
    ("gold", "古典鎏金"),
    ("silver", "月光冰银"),
    ("violet_gold", "幻彩紫金"),
    ("rose_gold", "玫瑰粉金"),
    ("obsidian_gold", "黑曜暗金"),
    ("crimson_gold", "绯红赤金"),
    ("jade_gold", "翡翠青金"),
    ("cyber_chrome", "电光冰铬"),
    ("aurora_prism", "极光铂彩"),
    ("antique_bronze", "古铜雕金"),
]

INTENSITY_SCALES = {
    "subtle": 0.72,
    "medium": 1.00,
    "lavish": 1.28,
    "rich": 1.28,
}

LINE_ETCH_SCALES = {
    "off": 0.0,
    "subtle": 0.55,
    "medium": 1.00,
    "strong": 1.45,
}

CARD_ASPECT_RATIOS: Dict[str, Optional[float]] = {
    "auto": None,
    "tarot": 1.62,
    "pokemon": 1.40,
    "portrait": 4.0 / 3.0,
    "square": 1.0,
}

CARD_ASPECT_META: List[Tuple[str, str]] = [
    ("auto", "跟随原图比例"),
    ("tarot", "标准塔罗牌 (5:8)"),
    ("pokemon", "宝可梦闪卡 (5:7)"),
    ("portrait", "经典竖卡 (3:4)"),
    ("square", "方形流麻砖 (1:1)"),
]

GIF_QUALITY_PRESETS: Dict[str, Dict[str, object]] = {
    "standard": {
        "label": "轻量流畅",
        "desc": "360px · 36帧 · 小体积快传",
        "width": 360,
        "frames": 36,
        "supersample": 2,
        "palette_sample": 110,
        "sharpen_pct": 0,
    },
    "hd": {
        "label": "高清标准",
        "desc": "480px · 48帧 · 均衡推荐",
        "width": 480,
        "frames": 48,
        "supersample": 2,
        "palette_sample": 144,
        "sharpen_pct": 26,
    },
    "ultra": {
        "label": "超清典藏",
        "desc": "680px · 48帧 · 高精锐化",
        "width": 680,
        "frames": 48,
        "supersample": 2,
        "palette_sample": 200,
        "sharpen_pct": 42,
    },
}


def resolve_inner_aspect(src_h: int, src_w: int, card_aspect: str = "auto") -> float:
    """Resolve card inner height/width ratio from preset or source image dimensions."""
    ratio = CARD_ASPECT_RATIOS.get(card_aspect)
    if ratio is not None:
        return float(ratio)
    return float(src_h) / max(1.0, float(src_w))


def place_subject_on_canvas(
    src_rgb: np.ndarray,
    src_alpha: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    fit_w: Optional[int] = None,
    fit_h: Optional[int] = None,
    subject_scale: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    depth_raw: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Place subject RGB, alpha, and optional 1:1 aligned depth map onto a target
    (canvas_h, canvas_w) surface with premultiplied-alpha anti-aliasing (zero
    dark edge fringe), proportional contain-fit inside (fit_w, fit_h), user zoom
    (`subject_scale`), and center offset (`offset_x`, `offset_y` in fractions of
    canvas dimensions).
    """
    src_h, src_w = src_rgb.shape[:2]
    fw = float(fit_w if fit_w is not None else canvas_w)
    fh = float(fit_h if fit_h is not None else canvas_h)
    base_fit = min(fw / max(1.0, float(src_w)), fh / max(1.0, float(src_h)))
    scale = max(0.15, min(3.0, float(subject_scale))) * base_fit

    cx_s = (src_w - 1.0) * 0.5
    cy_s = (src_h - 1.0) * 0.5
    cx_t = (canvas_w - 1.0) * 0.5 + float(offset_x) * float(canvas_w)
    cy_t = (canvas_h - 1.0) * 0.5 + float(offset_y) * float(canvas_h)

    M = np.array(
        [
            [scale, 0.0, cx_t - scale * cx_s],
            [0.0, scale, cy_t - scale * cy_s],
        ],
        dtype=np.float32,
    )

    interp = cv2.INTER_AREA if scale < 0.99 else cv2.INTER_LINEAR
    premul = (src_rgb * src_alpha[..., None]).astype(np.float32)
    out_premul = cv2.warpAffine(
        premul,
        M,
        (canvas_w, canvas_h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0),
    )
    out_a = cv2.warpAffine(
        src_alpha.astype(np.float32),
        M,
        (canvas_w, canvas_h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    out_a = np.clip(out_a, 0.0, 1.0).astype(np.float32)
    out_rgb = np.where(
        out_a[..., None] > 1e-4,
        np.clip(out_premul / np.maximum(out_a[..., None], 1e-4), 0.0, 1.0),
        0.0,
    ).astype(np.float32)

    out_depth: Optional[np.ndarray] = None
    if depth_raw is not None:
        d_src = depth_raw.astype(np.float32)
        if d_src.shape[:2] != (src_h, src_w):
            d_src = cv2.resize(d_src, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
        d_premul = (d_src * src_alpha.astype(np.float32)).astype(np.float32)
        out_d_pre = cv2.warpAffine(
            d_premul,
            M,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        out_depth = np.where(
            out_a > 1e-4,
            np.clip(out_d_pre / np.maximum(out_a, 1e-4), 0.0, 1.0),
            0.0,
        ).astype(np.float32)

    return out_rgb, out_a, out_depth


def analyze_image_theme(
    img_rgb: np.ndarray,
    style: str = "auto",
    pattern: str = "auto",
    metal: str = "auto",
    intensity: str = "medium",
    fg_mask: Optional[np.ndarray] = None,
) -> ThemeConfig:
    """Analyze image palette & luminance to auto-select harmonious tarot styling."""
    hsv = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    h_deg = hsv[..., 0].astype(np.float32) * 2.0
    s_val = hsv[..., 1].astype(np.float32) / 255.0
    v_val = hsv[..., 2].astype(np.float32) / 255.0

    valid = (fg_mask > 0.5) if (fg_mask is not None and np.any(fg_mask > 0.5)) else np.ones(v_val.shape, dtype=bool)
    mean_v = float(np.mean(v_val[valid]))
    mean_s = float(np.mean(s_val[valid]))
    # Weighted circular hue for saturated foreground pixels
    sat_mask = valid & (s_val > 0.14)
    if np.any(sat_mask):
        angles = np.deg2rad(h_deg[sat_mask])
        weights = s_val[sat_mask]
        dom_hue = float(np.rad2deg(np.arctan2(np.sum(np.sin(angles) * weights),
                                              np.sum(np.cos(angles) * weights))) % 360.0)
        sat_mean_s = float(np.mean(s_val[sat_mask]))
    else:
        dom_hue = 38.0  # Warm neutral default
        sat_mean_s = 0.0

    # Auto style
    if style == "auto":
        if mean_v < 0.28:
            style = "gothic" if mean_s < 0.30 else "moon"
        elif 195 <= dom_hue <= 295 and sat_mean_s > 0.16:
            style = "dream" if (dom_hue < 235 and mean_s > 0.22) else "moon"
        elif 165 <= dom_hue < 195 and mean_s > 0.20:
            style = "dream"
        elif (dom_hue <= 55 or dom_hue >= 330) and mean_v > 0.58 and mean_s > 0.35:
            style = "sun"
        else:
            style = "classical"

    # Auto pattern
    if pattern == "auto":
        pattern_map = {
            "moon": "cosmos",
            "sun": "sunburst",
            "dream": "silk_wave",
            "gothic": "stained_glass",
            "classical": "geometric",
            "guofeng": "guofeng_cloud",
            "sakura": "sakura_petal",
            "cyber": "hex_cyber",
            "prism": "prism_shard",
            "nouveau": "stained_glass",
            "abyss": "fingerprint",
        }
        pattern = pattern_map.get(style, "geometric")

    # Auto metal
    if metal == "auto":
        metal_map = {
            "moon": "silver",
            "sun": "gold",
            "dream": "violet_gold",
            "gothic": "crimson_gold",
            "guofeng": "jade_gold",
            "sakura": "rose_gold",
            "cyber": "cyber_chrome",
            "prism": "aurora_prism",
            "nouveau": "antique_bronze",
            "abyss": "silver",
        }
        if style in metal_map:
            metal = metal_map[style]
        else:
            # Classical warm/neutral tones pair elegantly with antique champagne gold
            metal = "champagne" if mean_s < 0.28 else "gold"

    m_base, m_bright, m_shadow = METAL_PALETTES.get(metal, METAL_PALETTES["champagne"])

    # Studio background & halo harmonized with style
    style_lighting = {
        "moon": ([0.075, 0.085, 0.145], [0.020, 0.024, 0.048], [0.36, 0.48, 0.78], [0.62, 0.56, 0.85], 1.15),
        "sun": ([0.115, 0.082, 0.062], [0.032, 0.022, 0.018], [0.85, 0.64, 0.30], [0.76, 0.42, 0.28], 0.25),
        "dream": ([0.068, 0.098, 0.125], [0.018, 0.030, 0.044], [0.38, 0.68, 0.78], [0.66, 0.48, 0.80], 2.05),
        "gothic": ([0.095, 0.055, 0.088], [0.024, 0.014, 0.026], [0.66, 0.42, 0.58], [0.78, 0.60, 0.34], 3.20),
        "guofeng": ([0.058, 0.105, 0.096], [0.016, 0.032, 0.030], [0.36, 0.74, 0.62], [0.82, 0.72, 0.44], 1.65),
        "sakura": ([0.118, 0.068, 0.096], [0.034, 0.018, 0.032], [0.90, 0.56, 0.72], [0.78, 0.66, 0.88], 2.65),
        "cyber": ([0.056, 0.064, 0.132], [0.014, 0.016, 0.042], [0.24, 0.84, 0.94], [0.88, 0.34, 0.76], 3.85),
        "prism": ([0.086, 0.070, 0.136], [0.024, 0.018, 0.044], [0.74, 0.58, 0.96], [0.46, 0.86, 0.94], 4.40),
        "nouveau": ([0.108, 0.086, 0.068], [0.030, 0.024, 0.020], [0.84, 0.66, 0.40], [0.44, 0.68, 0.64], 0.92),
        "abyss": ([0.048, 0.078, 0.134], [0.012, 0.022, 0.046], [0.34, 0.68, 0.94], [0.68, 0.86, 1.00], 1.42),
        "classical": ([0.088, 0.080, 0.112], [0.022, 0.020, 0.034], [0.78, 0.64, 0.42], [0.42, 0.48, 0.72], 0.65),
    }
    bg_center, bg_edge, halo_p, halo_s, irid_shift = style_lighting.get(style, style_lighting["classical"])

    return ThemeConfig(
        style_name=style,
        pattern_name=pattern,
        metal_name=metal,
        intensity_name=intensity,
        intensity_scale=INTENSITY_SCALES.get(intensity, 1.0),
        metal_base=np.array(m_base, dtype=np.float32),
        metal_bright=np.array(m_bright, dtype=np.float32),
        metal_shadow=np.array(m_shadow, dtype=np.float32),
        bg_center=np.array(bg_center, dtype=np.float32),
        bg_edge=np.array(bg_edge, dtype=np.float32),
        halo_primary=np.array(halo_p, dtype=np.float32),
        halo_secondary=np.array(halo_s, dtype=np.float32),
        irid_phase_shift=irid_shift,
    )


# ---------------------------------------------------------------------------
# Card Geometry, Safe Outer Border & Anti-Aliased Rounded Mask
# ---------------------------------------------------------------------------

def detect_existing_border(img_rgb: np.ndarray) -> bool:
    """Detect whether the input image already has a clean uniform outer border."""
    h, w = img_rgb.shape[:2]
    b = max(6, int(min(h, w) * 0.03))
    top = img_rgb[:b, :, :]
    bot = img_rgb[-b:, :, :]
    left = img_rgb[:, :b, :]
    right = img_rgb[:, -b:, :]
    border_pixels = np.concatenate(
        [top.reshape(-1, 3), bot.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)],
        axis=0,
    )
    border_std = float(np.mean(np.std(border_pixels, axis=0)))
    return border_std < 0.055


def build_rounded_rect_sdf(h: int, w: int, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Signed distance field (negative inside, positive outside) and smooth outward
    perimeter unit normals (nx, ny) for a rounded rectangle.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    hx = (w - 1) * 0.5 - radius
    hy = (h - 1) * 0.5 - radius

    px = xx - cx
    py = yy - cy
    qx = np.abs(px) - hx
    qy = np.abs(py) - hy

    outside_x = np.maximum(qx, 0.0)
    outside_y = np.maximum(qy, 0.0)
    outside_len = np.sqrt(outside_x * outside_x + outside_y * outside_y)
    inside_dist = np.minimum(np.maximum(qx, qy), 0.0)
    sdf = outside_len + inside_dist - radius

    # Smooth gradient of SDF -> outward unit normal
    gx = cv2.Sobel(sdf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sdf, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx = gx / norm
    ny = gy / norm
    return sdf, nx, ny


# ---------------------------------------------------------------------------
# Smart Solid / Transparent Background Cutout & Color De-Fringing
# ---------------------------------------------------------------------------

def extract_subject_alpha(
    pil_img: Image.Image,
    bg_mode: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Detect and cleanly remove solid-color or pre-existing transparent backgrounds
    (e.g., white-background anime/character illustrations or RGBA PNGs) while
    preserving white hair/clothing via border-connected flood-fill + enclosed
    ink-bounded hole detection + 1px anti-aliased edge color decontamination.

    bg_mode:
        - "keep":   never remove background
        - "auto":   remove if image has alpha transparency OR uniform solid backdrop
        - "remove": force border-connected solid backdrop cutout
    Returns:
        (rgb_f32, subject_alpha_f32, bg_was_removed)
    """
    rgba = np.asarray(pil_img.convert("RGBA"), dtype=np.float32) / 255.0
    rgb = rgba[..., :3]
    src_a = rgba[..., 3]
    h, w = rgb.shape[:2]

    if bg_mode in ("keep", "preserve"):
        return rgb, np.ones((h, w), dtype=np.float32), False

    # Case 1: Image already carries meaningful alpha transparency
    if float(np.mean(src_a < 0.90)) > 0.05:
        return rgb, src_a, True

    # Case 2: Solid outer background (white, light grey, black, studio flat color)
    u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    border_strip = np.concatenate([u8[0, :], u8[-1, :], u8[:, 0], u8[:, -1]], axis=0)
    bg_col = np.median(border_strip, axis=0).astype(np.float32)
    border_dist = np.linalg.norm(border_strip.astype(np.float32) - bg_col[None, :], axis=-1)
    uniform_border_ratio = float(np.mean(border_dist < 18.0))

    if bg_mode == "auto" and uniform_border_ratio < 0.65:
        return rgb, np.ones((h, w), dtype=np.float32), False

    dist = np.linalg.norm(u8.astype(np.float32) - bg_col[None, None, :], axis=-1)
    cand = (dist < 22.0).astype(np.uint8)
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=4)
    border_lbls = set(np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1],
    ]))) - {0}

    if not border_lbls:
        return rgb, np.ones((h, w), dtype=np.float32), False

    bg_mask = np.isin(labels, list(border_lbls)).astype(np.uint8)
    bg_area_ratio = float(np.mean(bg_mask))
    min_ratio = 0.12 if bg_mode == "auto" else 0.03
    if bg_area_ratio < min_ratio or bg_area_ratio > 0.96:
        return rgb, np.ones((h, w), dtype=np.float32), False

    # Detect enclosed background holes (e.g., gaps between hair, arms, tail, wings)
    # Only removes interior regions that are uniformly bg_col AND surrounded by
    # high-contrast/dark boundaries — protects white dress & white hair highlights!
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    bg_lum = float(0.299 * bg_col[0] + 0.587 * bg_col[1] + 0.114 * bg_col[2]) / 255.0
    strict_cand = (dist < 18.0).astype(np.uint8)
    n_in, l_in, s_in, _ = cv2.connectedComponentsWithStats(strict_cand, connectivity=4)
    border_in = set(np.unique(np.concatenate([
        l_in[0, :], l_in[-1, :], l_in[:, 0], l_in[:, -1],
    ]))) - {0}
    k5 = np.ones((5, 5), np.uint8)
    max_hole_area = int(h * w * 0.04)

    for lbl in range(1, n_in):
        if lbl in border_in:
            continue
        area = int(s_in[lbl, cv2.CC_STAT_AREA])
        if area < 40 or area > max_hole_area:
            continue
        hole = (l_in == lbl).astype(np.uint8)
        perim = (cv2.dilate(hole, k5) - hole) > 0
        if not np.any(perim):
            continue
        contrast_ratio = float(np.mean(np.abs(lum[perim] - bg_lum) > 0.30))
        pure_ratio = float(np.mean(dist[hole > 0] < 3.5))
        if contrast_ratio > 0.65 and pure_ratio > 0.55:
            bg_mask = np.maximum(bg_mask, hole)

    # Anti-aliased 1px transition ring & color decontamination (eliminates white halo)
    bg_f = bg_mask.astype(np.float32)
    ring = (cv2.dilate(bg_mask, np.ones((3, 3), np.uint8)) - bg_mask) > 0
    alpha = 1.0 - bg_f
    ring_alpha = np.clip(dist / 72.0, 0.0, 1.0).astype(np.float32)
    alpha = np.where(ring, ring_alpha, alpha)

    bg_rgb_f = bg_col / 255.0
    a_safe = np.maximum(alpha, 0.08)[..., None]
    rgb_clean = np.where(
        ring[..., None],
        np.clip((rgb - bg_rgb_f[None, None, :] * (1.0 - alpha[..., None])) / a_safe, 0.0, 1.0),
        rgb,
    )
    return rgb_clean.astype(np.float32), alpha.astype(np.float32), True


def build_tarot_etched_backdrop(
    card_h: int,
    card_w: int,
    theme: ThemeConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesize a rich, deep velvet tarot backdrop + subtle sacred-geometry
    celestial astrolabe ornament mask for cutout illustrations.
    """
    yy, xx = np.mgrid[0:card_h, 0:card_w].astype(np.float32)
    u = (xx / max(1, card_w - 1)) * 2.0 - 1.0
    v = (yy / max(1, card_h - 1)) * 2.0 - 1.0
    aspect = card_h / float(card_w)
    v_iso = v * aspect

    # Deep studio card velvet radial gradient
    r2 = u * u * 0.85 + v * v * 0.72
    grad = np.exp(-r2 * 1.15)[..., None]
    base_c = np.clip(theme.bg_center * 1.25 + 0.015, 0.04, 0.22)
    edge_c = np.clip(theme.bg_edge * 1.15 + 0.008, 0.02, 0.10)
    backdrop = edge_c[None, None, :] * (1.0 - grad) + base_c[None, None, :] * grad

    # Soft backlit moonlit/gilded aura centered around upper-middle behind subject
    cy_aura = -0.18 * aspect
    r_aura = np.sqrt(u * u + (v_iso - cy_aura) ** 2)
    aura_glow = np.exp(-(r_aura ** 2) / 0.58) * 0.24
    backdrop = backdrop + theme.halo_primary[None, None, :] * aura_glow[..., None]

    # Delicate sacred-geometry celestial astrolabe rings behind subject (no text/runes)
    r1 = 0.66
    r2_ring = 0.74
    ring_a = np.exp(-((r_aura - r1) ** 2) / 0.00014)
    ring_b = np.exp(-((r_aura - r2_ring) ** 2) / 0.00009)
    # Subtle 12-spoke celestial rays inside the ring band
    ang = np.arctan2(v_iso - cy_aura, u)
    spokes = ((0.5 + 0.5 * np.cos(ang * 12.0)) ** 18.0) * np.exp(-((r_aura - 0.70) ** 2) / 0.0022)
    # Subtle vertical axis hairline
    axis_v = np.exp(-(u * u) / 0.00004) * np.clip(1.0 - np.abs(v) * 0.8, 0.0, 1.0) * 0.45

    celestial_orn = np.clip(ring_a * 0.75 + ring_b * 0.55 + spokes * 0.65 + axis_v, 0.0, 1.0).astype(np.float32)
    orn_tint = theme.metal_base * 0.55 + theme.halo_secondary * 0.45
    backdrop = backdrop + orn_tint[None, None, :] * (celestial_orn * 0.28)[..., None]

    return np.clip(backdrop, 0.0, 1.0).astype(np.float32), celestial_orn


def build_card_base_and_frame(
    src_rgb: np.ndarray,
    target_card_w: int,
    theme: ThemeConfig,
    subject_alpha: Optional[np.ndarray] = None,
    bg_removed: bool = False,
    subject_scale: float = 1.0,
    subject_offset_x: float = 0.0,
    subject_offset_y: float = 0.0,
    card_aspect: str = "auto",
) -> Dict[str, np.ndarray]:
    """
    Prepare the card face texture at working resolution while preserving the
    subject's proportions and applying optional user scaling/offset & card aspect.
    """
    src_h, src_w = src_rgb.shape[:2]
    has_user_xform = (
        abs(subject_scale - 1.0) > 1e-3
        or abs(subject_offset_x) > 1e-3
        or abs(subject_offset_y) > 1e-3
        or card_aspect != "auto"
    )
    has_border = False if (bg_removed or has_user_xform) else detect_existing_border(src_rgb)
    aspect_hw = resolve_inner_aspect(src_h, src_w, card_aspect=card_aspect)

    if has_border:
        pad = 0
        inner_w = target_card_w
        inner_h = int(round(inner_w * aspect_hw))
        card_w, card_h = inner_w, inner_h
    else:
        pad = max(20, int(round(target_card_w * 0.052)))
        inner_w = target_card_w - 2 * pad
        inner_h = int(round(inner_w * aspect_hw))
        card_w = inner_w + 2 * pad
        card_h = inner_h + 2 * pad

    src_a_in = subject_alpha if (bg_removed and subject_alpha is not None) else np.ones((src_h, src_w), dtype=np.float32)
    inner_rgb, inner_a, _ = place_subject_on_canvas(
        src_rgb,
        src_a_in,
        canvas_w=inner_w,
        canvas_h=inner_h,
        fit_w=inner_w,
        fit_h=inner_h,
        subject_scale=subject_scale,
        offset_x=subject_offset_x,
        offset_y=subject_offset_y,
    )

    subject_alpha_full = np.zeros((card_h, card_w), dtype=np.float32)
    subject_alpha_full[pad:pad + inner_h, pad:pad + inner_w] = inner_a
    effective_bg_removed = bg_removed or bool(np.any(inner_a < 0.95))

    # Anti-aliased rounded corner mask (only cuts outer border when pad > 0)
    radius = float(max(14, int(round(pad * 1.15)))) if pad > 0 else float(max(12, int(round(min(card_w, card_h) * 0.028))))
    sdf, edge_nx, edge_ny = build_rounded_rect_sdf(card_h, card_w, radius)
    alpha_mask = np.clip(0.5 - sdf / 1.5, 0.0, 1.0).astype(np.float32)

    # Metallic bevel mask along outer perimeter (inside the card edge)
    bevel_width = max(3.5, card_w * 0.0085)
    bevel_mask = np.clip(1.0 - np.abs(sdf + bevel_width * 0.55) / (bevel_width * 0.65), 0.0, 1.0)
    bevel_mask *= alpha_mask

    photo_mask = np.zeros((card_h, card_w), dtype=np.float32)
    photo_mask[pad:pad + inner_h, pad:pad + inner_w] = 1.0
    outer_frame_mask = np.clip(1.0 - photo_mask, 0.0, 1.0)
    ornament_mask = np.zeros((card_h, card_w), dtype=np.float32)
    celestial_orn = np.zeros((card_h, card_w), dtype=np.float32)

    if effective_bg_removed:
        # Unified dark etched tarot canvas behind the cutout/resized subject
        card_rgb, celestial_orn = build_tarot_etched_backdrop(card_h, card_w, theme)
    elif pad > 0:
        border_stock = np.clip(theme.bg_center * 0.85 + 0.025, 0.04, 0.14)
        card_rgb = np.ones((card_h, card_w, 3), dtype=np.float32) * border_stock[None, None, :]
        card_rgb[pad:pad + inner_h, pad:pad + inner_w] = inner_rgb
    else:
        card_rgb = inner_rgb.copy()

    if pad > 0:
        orn_u8 = np.zeros((card_h, card_w), dtype=np.uint8)
        x0, y0 = pad - 1, pad - 1
        x1, y1 = pad + inner_w, pad + inner_h
        cv2.rectangle(orn_u8, (x0, y0), (x1, y1), 235, thickness=max(1, pad // 14), lineType=cv2.LINE_AA)

        m = max(4, int(round(pad * 0.42)))
        mid_sdf, _, _ = build_rounded_rect_sdf(card_h, card_w, max(6.0, radius - m))
        mid_line = np.clip(1.0 - np.abs(mid_sdf + m) / max(1.1, pad * 0.055), 0.0, 1.0)

        d_sz = max(4, int(round(pad * 0.24)))
        corner_centers = [
            (int(pad * 0.50), int(pad * 0.50)),
            (card_w - 1 - int(pad * 0.50), int(pad * 0.50)),
            (int(pad * 0.50), card_h - 1 - int(pad * 0.50)),
            (card_w - 1 - int(pad * 0.50), card_h - 1 - int(pad * 0.50)),
        ]
        for cx, cy in corner_centers:
            pts = np.array([
                [cx, cy - d_sz], [cx + d_sz, cy], [cx, cy + d_sz], [cx - d_sz, cy],
            ], dtype=np.int32)
            cv2.polylines(orn_u8, [pts], True, 240, thickness=max(1, pad // 15), lineType=cv2.LINE_AA)
            cv2.circle(orn_u8, (cx, cy), max(1, d_sz // 3), 255, thickness=-1, lineType=cv2.LINE_AA)

        mid_ticks = [
            (card_w // 2, int(pad * 0.48)),
            (card_w // 2, card_h - 1 - int(pad * 0.48)),
            (int(pad * 0.48), card_h // 2),
            (card_w - 1 - int(pad * 0.48), card_h // 2),
        ]
        t_sz = max(3, int(round(pad * 0.18)))
        for cx, cy in mid_ticks:
            pts = np.array([
                [cx, cy - t_sz], [cx + t_sz, cy], [cx, cy + t_sz], [cx - t_sz, cy],
            ], dtype=np.int32)
            cv2.fillConvexPoly(orn_u8, pts, 230, lineType=cv2.LINE_AA)

        ornament_mask = np.maximum(orn_u8.astype(np.float32) / 255.0, mid_line * 0.85)
        ornament_mask *= outer_frame_mask
        ornament_mask *= alpha_mask

        gold_pigment = theme.metal_base * 0.72 + theme.metal_shadow * 0.28
        card_rgb = card_rgb * (1.0 - ornament_mask[..., None] * 0.88) + gold_pigment[None, None, :] * (ornament_mask[..., None] * 0.88)
    else:
        border_band = np.clip((sdf + min(card_w, card_h) * 0.085) / (min(card_w, card_h) * 0.05), 0.0, 1.0)
        outer_frame_mask = border_band * alpha_mask
        gray = cv2.cvtColor((card_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.clip(np.sqrt(gx * gx + gy * gy) * 1.4, 0.0, 1.0)
        ornament_mask = np.clip(outer_frame_mask * (edge_mag * 0.8 + np.clip((gray - 0.55) * 2.0, 0.0, 1.0) * 0.5), 0.0, 1.0)

    # Extract crisp character/subject linework mask directly from inner_rgb (before dark backdrop composite)
    inner_lum = (0.299 * inner_rgb[..., 0] + 0.587 * inner_rgb[..., 1] + 0.114 * inner_rgb[..., 2]).astype(np.float32)
    bh_k = max(5, int(round(inner_w * 0.011)) | 1)
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_k, bh_k))
    local_hi = cv2.morphologyEx(inner_lum, cv2.MORPH_CLOSE, bh_kernel)
    blackhat = np.clip((local_hi - inner_lum) * 3.2, 0.0, 1.0)
    inner_line = np.clip((blackhat - 0.14) / 0.55, 0.0, 1.0) ** 0.85

    # Soften line glare on upper face/skin so eyes & expression remain crisp and gentle
    inner_u8 = np.clip(inner_rgb * 255.0, 0, 255).astype(np.uint8)
    inner_ycrcb = cv2.cvtColor(inner_u8, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    inner_skin = (
        np.exp(-0.5 * ((inner_ycrcb[..., 1] - 146.0) / 14.0) ** 2)
        * np.exp(-0.5 * ((inner_ycrcb[..., 2] - 112.0) / 14.0) ** 2)
    )
    iy, _ = np.mgrid[0:inner_h, 0:inner_w].astype(np.float32)
    skin_blur = cv2.GaussianBlur(inner_skin, (0, 0), sigmaX=max(6.0, inner_w * 0.025))
    face_atten = np.clip(skin_blur * 2.2 * np.clip(1.2 - iy / max(1.0, inner_h * 0.45), 0.0, 1.0), 0.0, 0.82)
    inner_line = (inner_line * (1.0 - face_atten * 0.78)).astype(np.float32)

    line_etch_mask = np.zeros((card_h, card_w), dtype=np.float32)
    line_etch_mask[pad:pad + inner_h, pad:pad + inner_w] = inner_line
    line_etch_mask *= subject_alpha_full
    line_gx = cv2.Sobel(line_etch_mask, cv2.CV_32F, 1, 0, ksize=3)
    line_gy = cv2.Sobel(line_etch_mask, cv2.CV_32F, 0, 1, ksize=3)

    # Composite cutout/resized subject cleanly ON TOP of the etched backdrop + subtle grounding shadow
    bg_cutout_mask = np.zeros((card_h, card_w), dtype=np.float32)
    if effective_bg_removed:
        # Soft ambient grounding shadow cast by subject onto the etched tarot background
        shadow_blur = cv2.GaussianBlur(subject_alpha_full, (0, 0), sigmaX=max(5.0, card_w * 0.012))
        shadow_shift = np.roll(np.roll(shadow_blur, max(2, card_h // 180), axis=0), max(2, card_w // 180), axis=1)
        card_rgb *= (1.0 - shadow_shift[..., None] * 0.45 * (1.0 - subject_alpha_full[..., None]))

        # Composite de-fringed subject
        subj_canvas = np.zeros((card_h, card_w, 3), dtype=np.float32)
        subj_canvas[pad:pad + inner_h, pad:pad + inner_w] = inner_rgb
        sa = subject_alpha_full[..., None]
        card_rgb = card_rgb * (1.0 - sa) + subj_canvas * sa

        bg_cutout_mask = np.clip(1.0 - subject_alpha_full, 0.0, 1.0) * alpha_mask
        # Include celestial backdrop lines in ornament_mask strictly behind the subject
        ornament_mask = np.clip(ornament_mask + celestial_orn * bg_cutout_mask * 0.65, 0.0, 1.0)

    return {
        "card_rgb": card_rgb,
        "alpha_mask": alpha_mask,
        "sdf": sdf,
        "edge_nx": edge_nx,
        "edge_ny": edge_ny,
        "bevel_mask": bevel_mask,
        "photo_mask": photo_mask,
        "outer_frame_mask": outer_frame_mask,
        "ornament_mask": ornament_mask,
        "bg_cutout_mask": bg_cutout_mask,
        "subject_alpha_full": subject_alpha_full,
        "line_etch_mask": line_etch_mask,
        "line_gx": line_gx,
        "line_gy": line_gy,
        "bg_removed": np.array([1 if bg_removed else 0], dtype=np.int32),
        "pad": np.array([pad], dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Conservative Zoning, Micro-Etching & Surface Normals (Card-Local Space)
# ---------------------------------------------------------------------------

def build_conservative_masks(
    card_rgb: np.ndarray,
    outer_frame_mask: np.ndarray,
    ornament_mask: np.ndarray,
    photo_mask: np.ndarray,
    bg_cutout_mask: Optional[np.ndarray] = None,
    subject_alpha_full: Optional[np.ndarray] = None,
    bg_removed: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Build smoothly feathered, conservative zoning masks:
      - protect_mask: protects human faces/skin, cutout character subject & crisp text
      - dark_bg_mask: knocked-out tarot backdrop or dark zones for astral etching & foil
      - accent_mask: frame & bright decorative elements for gilded reflections
    """
    h, w = card_rgb.shape[:2]
    u8 = np.clip(card_rgb * 255.0, 0, 255).astype(np.uint8)
    lum = (0.299 * card_rgb[..., 0] + 0.587 * card_rgb[..., 1] + 0.114 * card_rgb[..., 2]).astype(np.float32)

    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)

    if bg_removed and bg_cutout_mask is not None and subject_alpha_full is not None:
        # Pixel-accurate subject vs etched-backdrop zoning!
        # Protect the entire cutout character cleanly so hair, face, eyes & attire stay pristine,
        # while the knocked-out dark tarot backdrop carries full astral/geometric etching & foil.
        subj_soft = cv2.GaussianBlur(subject_alpha_full, (0, 0), sigmaX=max(2.0, w * 0.004))
        protect_mask = np.clip(subj_soft * 0.90, 0.0, 1.0)
        dark_bg_mask = np.clip(bg_cutout_mask, 0.0, 1.0)
        frame_soft = cv2.GaussianBlur(outer_frame_mask, (0, 0), sigmaX=max(1.5, w * 0.004))
        accent_mask = np.clip(ornament_mask * 1.0 + frame_soft * 0.55, 0.0, 1.0)
    else:
        # 1. Conservative skin-tone detection in YCrCb
        ycrcb = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        cr = ycrcb[..., 1]
        cb = ycrcb[..., 2]
        skin_cr = np.exp(-0.5 * ((cr - 151.0) / 14.0) ** 2)
        skin_cb = np.exp(-0.5 * ((cb - 107.0) / 14.0) ** 2)
        skin_prob = skin_cr * skin_cb * photo_mask

        # Spatial prior for upper-central portrait area (soft oval)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx / max(1, w - 1) - 0.48) / 0.30
        ny = (yy / max(1, h - 1) - 0.44) / 0.34
        center_prior = np.exp(-0.5 * (nx * nx + ny * ny))

        face_protect = cv2.GaussianBlur(skin_prob * (0.45 + 0.55 * center_prior), (0, 0), sigmaX=max(8.0, w * 0.03))
        face_protect = np.clip(face_protect * 2.4, 0.0, 1.0)

        # 2. High-frequency text / fine detail protection
        local_detail = cv2.GaussianBlur(grad_mag, (0, 0), sigmaX=max(5.0, w * 0.015))
        bright_detail = np.clip((lum - 0.42) * 2.2, 0.0, 1.0) * np.clip(local_detail * 3.2, 0.0, 1.0)
        text_protect = cv2.GaussianBlur(bright_detail, (0, 0), sigmaX=max(6.0, w * 0.02))
        text_protect = np.clip(text_protect * 1.6, 0.0, 1.0)

        protect_mask = np.clip(np.maximum(face_protect, text_protect * 0.85) + center_prior * 0.32, 0.0, 1.0)
        protect_mask = cv2.GaussianBlur(protect_mask, (0, 0), sigmaX=max(10.0, w * 0.032))
        protect_mask = np.clip(protect_mask * photo_mask, 0.0, 1.0)

        # 3. Dark / shadow foil carrier zone (outside protected areas)
        dark_raw = np.clip((0.46 - lum) / 0.34, 0.0, 1.0)
        dark_bg_mask = cv2.GaussianBlur(dark_raw, (0, 0), sigmaX=max(6.0, w * 0.018))
        dark_bg_mask = np.clip(dark_bg_mask * (1.0 - protect_mask * 0.78) * photo_mask, 0.0, 1.0)

        # 4. Gilded frame & bright decorative accent mask
        frame_soft = cv2.GaussianBlur(outer_frame_mask, (0, 0), sigmaX=max(1.5, w * 0.004))
        bright_edges = np.clip(grad_mag * 1.8, 0.0, 1.0) * np.clip((lum - 0.50) * 2.0, 0.0, 1.0)
        bright_edges = cv2.GaussianBlur(bright_edges, (0, 0), sigmaX=max(2.0, w * 0.005))
        accent_mask = np.clip(
            ornament_mask * 1.0 + frame_soft * 0.55 + bright_edges * (1.0 - protect_mask * 0.85) * 0.35,
            0.0,
            1.0,
        )

    # 5. Smooth micro-emboss normal field from luminance Sobel gradients
    lum_smooth = cv2.GaussianBlur(lum, (0, 0), sigmaX=1.2)
    relief_height = lum_smooth * (1.0 - protect_mask * 0.72) * 0.45 + ornament_mask * 0.90
    emb_gx = cv2.Sobel(relief_height, cv2.CV_32F, 1, 0, ksize=3)
    emb_gy = cv2.Sobel(relief_height, cv2.CV_32F, 0, 1, ksize=3)

    return {
        "lum": lum,
        "protect_mask": protect_mask,
        "dark_bg_mask": dark_bg_mask,
        "accent_mask": accent_mask,
        "emb_gx": emb_gx,
        "emb_gy": emb_gy,
    }


def build_micro_etching_texture(h: int, w: int, pattern: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate card-local micro-etching texture and diagonal micro-diffraction grating.
    Frequencies are tuned to avoid moiré and GIF quantization flicker at 460px output.
    Returns (etch_pattern, micro_grating), both in [0, 1].
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx / max(1, w - 1)) * 2.0 - 1.0
    v = (yy / max(1, h - 1)) * 2.0 - 1.0
    aspect = h / float(w)
    v_iso = v * aspect
    r = np.sqrt(u * u + v_iso * v_iso)
    ang = np.arctan2(v_iso, u)

    # Diagonal micro-diffraction grating (period ~10px at 2x SS -> ~5px final, crisp & moiré-free)
    period_px = max(8.5, w / 64.0)
    grating = 0.5 + 0.5 * np.sin((xx * 0.78 + yy * 0.62) * (2.0 * math.pi / period_px))

    if pattern == "cosmos":
        # Astral orbits + cosmos star speckles + ecliptic arcs
        orbits = np.exp(-((np.sin(r * 9.5) ** 2) / 0.042)) * 0.72
        arc2 = np.exp(-((np.sin((r + 0.25 * np.cos(ang * 2.0)) * 7.0) ** 2) / 0.038)) * 0.45
        rng = np.random.default_rng(42)
        stars = np.zeros((h, w), dtype=np.float32)
        n_stars = 140
        sx = rng.integers(0, w, size=n_stars)
        sy = rng.integers(0, h, size=n_stars)
        for x0, y0 in zip(sx, sy):
            cv2.circle(stars, (int(x0), int(y0)), max(2, w // 240), float(rng.uniform(0.6, 1.0)), -1, cv2.LINE_AA)
        stars = cv2.GaussianBlur(stars, (0, 0), sigmaX=1.2)
        etch = np.clip(np.maximum(orbits, arc2) + stars * 0.90, 0.0, 1.0)
    elif pattern == "sunburst":
        # Crisp radial rays + concentric solar rings
        rays = (0.5 + 0.5 * np.cos(ang * 20.0)) ** 2.4
        rings = np.exp(-((np.sin(r * 8.5) ** 2) / 0.048))
        etch = np.clip(rays * 0.62 + rings * 0.65, 0.0, 1.0)
    elif pattern == "silk_wave":
        # Flowing silk/water foil ripples
        wave1 = (0.5 + 0.5 * np.sin(u * 12.0 + np.sin(v_iso * 4.5) * 2.0)) ** 1.6
        wave2 = (0.5 + 0.5 * np.sin(v_iso * 10.0 - np.cos(u * 5.5) * 1.6)) ** 1.6
        etch = np.clip(wave1 * 0.62 + wave2 * 0.52, 0.0, 1.0)
    elif pattern == "prism_shard":
        # Cracked-Ice / Pokémon SAR prismatic Voronoi crystal facets with per-shard grating angles
        rng = np.random.default_rng(77)
        n_cells = 26
        seeds_u = rng.uniform(-1.05, 1.05, size=n_cells).astype(np.float32)
        seeds_v = rng.uniform(-aspect * 1.05, aspect * 1.05, size=n_cells).astype(np.float32)
        d1 = np.full((h, w), 1e6, dtype=np.float32)
        d2 = np.full((h, w), 1e6, dtype=np.float32)
        cell_id = np.zeros((h, w), dtype=np.int32)
        for idx in range(n_cells):
            dist_i = np.sqrt((u - seeds_u[idx]) ** 2 + (v_iso - seeds_v[idx]) ** 2)
            closer = dist_i < d1
            second = (~closer) & (dist_i < d2)
            d2 = np.where(closer, d1, np.where(second, dist_i, d2))
            d1 = np.where(closer, dist_i, d1)
            cell_id = np.where(closer, idx, cell_id)
        seam = np.exp(-((d2 - d1) ** 2) / 0.0014)
        cell_angles = rng.uniform(0.0, math.pi, size=n_cells).astype(np.float32)
        ang_map = cell_angles[cell_id]
        facet_dir = xx * np.cos(ang_map) + yy * np.sin(ang_map)
        grating = 0.5 + 0.5 * np.sin(facet_dir * (2.0 * math.pi / (period_px * 0.92)) + cell_id * 1.1)
        facet_tone = (cell_id.astype(np.float32) % 7.0) / 7.0
        etch = np.clip(seam * 0.95 + (0.20 + 0.34 * facet_tone) * grating, 0.0, 1.0)
    elif pattern == "fingerprint":
        # Iconic Japanese UR / SAR topographic fingerprint contour micro-etching
        phi = (
            r * 1.18
            + 0.28 * np.sin(u * 2.4 + v_iso * 1.35)
            + 0.22 * np.cos(u * 1.65 - v_iso * 2.55)
            + 0.14 * np.sin((u * u - v_iso * v_iso) * 2.1)
        )
        ridges = np.exp(-((np.sin(phi * 13.5) ** 2) / 0.048))
        fine_echo = np.exp(-((np.sin(phi * 27.0) ** 2) / 0.065)) * 0.28
        etch = np.clip(ridges * 0.92 + fine_echo, 0.0, 1.0)
        grating = 0.5 + 0.5 * np.sin(phi * 34.0 + (xx * 0.35 + yy * 0.25) * (2.0 * math.pi / (period_px * 1.4)))
    elif pattern == "hex_cyber":
        # Analytic honeycomb hexagonal cyber-circuit matrix + glowing intersection nodes
        scale_hex = 5.6
        px = u * scale_hex
        py = v_iso * scale_hex
        sx = 1.0
        sy = math.sqrt(3.0)
        ax = np.mod(px, sx) - 0.5 * sx
        ay = np.mod(py, sy) - 0.5 * sy
        bx = np.mod(px - 0.5 * sx, sx) - 0.5 * sx
        by = np.mod(py - 0.5 * sy, sy) - 0.5 * sy
        use_a = (ax * ax + ay * ay) < (bx * bx + by * by)
        hx = np.where(use_a, ax, bx)
        hy = np.where(use_a, ay, by)
        hex_d = np.maximum(np.abs(hx), np.abs(hx) * 0.5 + np.abs(hy) * 0.8660254)
        hex_edge = np.exp(-((0.50 - hex_d) ** 2) / 0.0024)
        hex_core = np.exp(-((hx * hx + hy * hy) / 0.010)) * 0.45
        scanline = 0.5 + 0.5 * np.sin(yy * (2.0 * math.pi / max(6.0, period_px * 0.85)))
        etch = np.clip(hex_edge * 0.92 + hex_core, 0.0, 1.0)
        grating = np.clip(grating * 0.55 + scanline * 0.45, 0.0, 1.0)
    elif pattern == "guofeng_cloud":
        # Auspicious Seigaiha / East Asian overlapping cloud-wave brocade fans
        freq_x = 3.2
        freq_y = 3.2
        ux = u * freq_x
        vy = v_iso * freq_y
        # Layer 1 & staggered Layer 2 concentric arcs
        fx1 = np.mod(ux + 0.5, 1.0) - 0.5
        fy1 = np.mod(vy, 1.0)
        r1 = np.sqrt(fx1 * fx1 + fy1 * fy1)
        arc1 = np.exp(-((np.sin(r1 * 15.0) ** 2) / 0.080)) * np.clip(1.0 - (r1 - 0.48) * 6.0, 0.0, 1.0)

        fx2 = np.mod(ux, 1.0) - 0.5
        fy2 = np.mod(vy + 0.5, 1.0)
        r2 = np.sqrt(fx2 * fx2 + fy2 * fy2)
        arc2 = np.exp(-((np.sin(r2 * 15.0) ** 2) / 0.080)) * np.clip(1.0 - (r2 - 0.48) * 6.0, 0.0, 1.0)

        mist = np.exp(-((np.sin((v_iso * 2.8 + 0.28 * np.sin(u * 4.2)) * math.pi) ** 2) / 0.055)) * 0.45
        etch = np.clip(np.maximum(arc1, arc2) * 0.95 + mist, 0.0, 1.0)
    elif pattern == "stained_glass":
        # Gothic cathedral rose-window tracery + pointed arch vaulting ribs
        vc = v_iso + 0.08 * aspect
        rc = np.sqrt(u * u + vc * vc)
        ac = np.arctan2(vc, u)
        petal_r = rc - 0.12 * np.abs(np.cos(ac * 6.0))
        tracery = np.exp(-((np.sin(petal_r * 11.5) ** 2) / 0.044)) * 0.88
        mullions = ((0.5 + 0.5 * np.cos(ac * 12.0)) ** 12.0) * np.clip(rc * 1.4, 0.0, 1.0) * 0.72
        arch = np.exp(-((np.sin((np.abs(u) * 1.15 + np.sqrt(np.maximum(0.02, 1.35 - v_iso * 0.55)) * 0.85) * 9.0) ** 2) / 0.048)) * 0.42
        etch = np.clip(np.maximum(tracery, mullions) + arch * 0.55, 0.0, 1.0)
    elif pattern == "sakura_petal":
        # Swirling twilight breeze ribbons + delicate 5-petal sakura blossom stamps
        swirl = np.exp(-((np.sin((u * 1.9 + v_iso * 1.15 + 0.36 * np.sin(v_iso * 3.6)) * 7.5) ** 2) / 0.052)) * 0.48
        blossoms = np.zeros((h, w), dtype=np.float32)
        rng = np.random.default_rng(131)
        n_flowers = 20
        for _ in range(n_flowers):
            cx = float(rng.uniform(0.08, 0.92)) * w
            cy = float(rng.uniform(0.08, 0.92)) * h
            rad = float(rng.uniform(0.055, 0.098)) * min(w, h)
            rot0 = float(rng.uniform(0.0, 2.0 * math.pi))
            for k in range(5):
                a_p = rot0 + k * (2.0 * math.pi / 5.0)
                px_c = int(round(cx + math.cos(a_p) * rad * 0.54))
                py_c = int(round(cy + math.sin(a_p) * rad * 0.54))
                ax1 = max(3, int(round(rad * 0.48)))
                ax2 = max(2, int(round(rad * 0.27)))
                cv2.ellipse(blossoms, (px_c, py_c), (ax1, ax2), math.degrees(a_p), 0, 360, 0.92, max(2, w // 140), cv2.LINE_AA)
            cv2.circle(blossoms, (int(round(cx)), int(round(cy))), max(2, int(round(rad * 0.16))), 1.0, -1, cv2.LINE_AA)
        blossoms = cv2.GaussianBlur(blossoms, (0, 0), sigmaX=0.9)
        etch = np.clip(swirl + blossoms * 0.95, 0.0, 1.0)
    elif pattern == "star_lattice":
        # Vintage collector holo star-sparkle crosshatch grid
        gx = (0.5 + 0.5 * np.cos(u * 14.5)) ** 6.0
        gy = (0.5 + 0.5 * np.cos(v_iso * 14.5)) ** 6.0
        stars = np.sqrt(np.clip(gx * gy, 0.0, 1.0)) * 0.95
        beams = np.clip((gx * np.exp(-((np.sin(v_iso * 7.25) ** 2) / 0.06)) + gy * np.exp(-((np.sin(u * 7.25) ** 2) / 0.06))) * 0.52, 0.0, 1.0)
        diag_glint = np.exp(-((np.sin((u + v_iso) * 10.2) ** 2) / 0.045)) * 0.34
        etch = np.clip(np.maximum(stars, beams) + diag_glint, 0.0, 1.0)
    else:  # geometric (classical / mystical)
        # Sacred-geometry concentric arcs + crisp diamond filigree lattice
        arcs = np.exp(-((np.sin(r * 9.5) ** 2) / 0.045)) * 0.82
        diag_a = np.exp(-((np.sin((u + v_iso) * 13.5) ** 2) / 0.055)) * 0.65
        diag_b = np.exp(-((np.sin((u - v_iso) * 13.5) ** 2) / 0.055)) * 0.65
        etch = np.clip(np.maximum(arcs, np.maximum(diag_a, diag_b) * 0.88), 0.0, 1.0)

    return etch.astype(np.float32), grating.astype(np.float32)


def build_brushed_metal_edge_texture(h: int, w: int) -> np.ndarray:
    """Fixed low-contrast brushed metal texture in card-local coordinates."""
    rng = np.random.default_rng(108)
    noise = rng.normal(0.0, 0.07, size=(h, w)).astype(np.float32)
    # Anisotropic blur for fine metallic brushing
    brushed_h = cv2.GaussianBlur(noise, (21, 3), sigmaX=7.0, sigmaY=0.7)
    brushed_v = cv2.GaussianBlur(noise, (3, 21), sigmaX=0.7, sigmaY=7.0)
    # Blend horizontal brushing on top/bottom and vertical brushing on left/right
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = np.abs(xx / max(1, w - 1) - 0.5)
    dy = np.abs(yy / max(1, h - 1) - 0.5)
    w_v = np.clip((dx - dy) * 6.0 + 0.5, 0.0, 1.0)
    brushed = brushed_v * w_v + brushed_h * (1.0 - w_v)
    return np.clip(0.88 + brushed, 0.70, 1.05).astype(np.float32)


def select_anchored_glint_points(
    card_w: int,
    card_h: int,
    pad: int,
    ornament_mask: np.ndarray,
    protect_mask: np.ndarray,
    lum: np.ndarray,
) -> List[Tuple[float, float, float]]:
    """
    Select 6 tasteful card-local (x, y, size_scale) anchor points for 4-point
    cross star glints. Prioritizes gilded border corners & cardinal ornaments,
    strictly avoiding face, eyes, and fine text.
    """
    anchors: List[Tuple[float, float, float]] = []
    if pad > 0:
        c = pad * 0.50
        anchors.extend([
            (c, c, 1.05),
            (card_w - 1 - c, c, 1.00),
            (c, card_h - 1 - c, 0.95),
            (card_w - 1 - c, card_h - 1 - c, 1.05),
            (card_w * 0.5, pad * 0.48, 0.85),
            (card_w * 0.5, card_h - 1 - pad * 0.48, 0.85),
        ])
    else:
        # Pick top local highlight spots on the border / ornament outside protected areas
        score = (ornament_mask * 0.7 + lum * 0.3) * (1.0 - protect_mask)
        score = cv2.GaussianBlur(score, (0, 0), sigmaX=4.0)
        min_dist = min(card_w, card_h) * 0.18
        for _ in range(6):
            idx = int(np.argmax(score))
            py, px = divmod(idx, card_w)
            if score[py, px] < 0.15:
                break
            anchors.append((float(px), float(py), 0.92))
            cv2.circle(score, (px, py), int(min_dist), 0.0, -1)
    return anchors


# ---------------------------------------------------------------------------
# 3D Orbital Transform & Perspective Homography
# ---------------------------------------------------------------------------

def rotation_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """3x3 rotation matrix R = Rz @ Ry @ Rx (angles in radians)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def compute_frame_pose(theta: float, card_h: float) -> Tuple[float, float, float, float]:
    """
    Unified periodic 3D orbital pose for phase theta in [0, 2*pi):
      - Y-axis yaw:   +/- 17.5 deg
      - X-axis pitch: +/- 10.5 deg (with natural orbital phase offset)
      - Z-axis roll:  +/- 2.0 deg
      - Float Y:      gentle vertical bob
    """
    ry = math.radians(17.5) * math.sin(theta)
    rx = math.radians(10.5) * math.cos(theta + 0.22)
    rz = math.radians(2.0) * math.sin(theta + 0.90)
    float_y = card_h * 0.022 * math.sin(theta + 0.45)
    return rx, ry, rz, float_y


def plane_homography_for_z(
    card_w: int,
    card_h: int,
    R: np.ndarray,
    z_local: float,
    float_y: float,
    focal: float,
    canvas_cx: float,
    canvas_cy: float,
) -> np.ndarray:
    """
    Exact 3x3 perspective homography mapping card-local pixel (u, v) on plane
    z = z_local to 2D canvas pixel (X, Y).
    """
    src_pts = np.array([
        [0.0, 0.0],
        [card_w - 1.0, 0.0],
        [card_w - 1.0, card_h - 1.0],
        [0.0, card_h - 1.0],
    ], dtype=np.float32)

    cx0 = (card_w - 1.0) * 0.5
    cy0 = (card_h - 1.0) * 0.5
    dst_pts = np.zeros((4, 2), dtype=np.float32)

    for k, (u, v) in enumerate(src_pts):
        p = np.array([u - cx0, v - cy0, z_local], dtype=np.float64)
        pr = R @ p
        # Camera at (0, 0, -focal), card centered at z = 0
        z_cam = focal + pr[2]
        scale = focal / z_cam
        dst_pts[k, 0] = canvas_cx + pr[0] * scale
        dst_pts[k, 1] = canvas_cy + (pr[1] + float_y) * scale

    return cv2.getPerspectiveTransform(src_pts, dst_pts)


def precompute_safe_canvas_size(
    card_w: int,
    card_h: int,
    n_frames: int,
    thickness_total: float,
    focal: float,
    target_canvas_w: int,
) -> Tuple[int, int, float]:
    """
    Precompute exact 3D projected bounding box across all frames (front & back
    thickness planes) so the card, thickness, and outer glints never clip.
    Returns (canvas_w, canvas_h, card_scale_adjustment).
    """
    min_x, max_x = 1e9, -1e9
    min_y, max_y = 1e9, -1e9
    cx0 = (card_w - 1.0) * 0.5
    cy0 = (card_h - 1.0) * 0.5
    corners = [(-cx0, -cy0), (cx0, -cy0), (cx0, cy0), (-cx0, cy0)]

    for i in range(n_frames):
        theta = 2.0 * math.pi * i / n_frames
        rx, ry, rz, float_y = compute_frame_pose(theta, card_h)
        R = rotation_matrix_xyz(rx, ry, rz)
        for z_loc in (0.0, thickness_total):
            for px, py in corners:
                pr = R @ np.array([px, py, z_loc], dtype=np.float64)
                s = focal / (focal + pr[2])
                x2 = pr[0] * s
                y2 = (pr[1] + float_y) * s
                min_x = min(min_x, x2)
                max_x = max(max_x, x2)
                min_y = min(min_y, y2)
                max_y = max(max_y, y2)

    proj_w = max_x - min_x
    proj_h = max_y - min_y
    # Leave generous studio margin (~15% each side horizontally, ~14% vertically)
    margin_ratio_x = 0.74
    scale_adj = (target_canvas_w * margin_ratio_x) / proj_w
    canvas_h = int(round((proj_h * scale_adj) / 0.76))
    # Keep even dimensions
    canvas_h += canvas_h % 2
    return target_canvas_w, canvas_h, scale_adj


# ---------------------------------------------------------------------------
# Shader: Dynamic Holographic Foil, Sweep, Gold Emboss & Studio Elements
# ---------------------------------------------------------------------------

def spectral_iridescence(phase: np.ndarray) -> np.ndarray:
    """
    Smooth, refined collector-foil spectral iridescence (gold -> violet -> teal -> sapphire)
    from a phase field in radians. Returns (H, W, 3) float32 in [0, 1].
    """
    r = 0.52 + 0.44 * np.cos(phase + 0.0)
    g = 0.50 + 0.42 * np.cos(phase + 2.05)
    b = 0.56 + 0.42 * np.cos(phase + 4.05)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def apply_highlight_rolloff(
    img: np.ndarray,
    knee: float = 0.92,
    spread: float = 0.08,
) -> np.ndarray:
    """
    C^1-continuous soft-knee highlight compression: preserves pure cel midtones
    below `knee` untouched while smoothly rolling bright specular peaks into [knee, 1].
    """
    over = np.maximum(img - knee, 0.0)
    rolled = knee + (1.0 - knee) * np.tanh(over / max(1e-4, spread))
    return np.clip(np.where(img > knee, rolled, img), 0.0, 1.0).astype(np.float32)


def premul_to_pil_rgba(premul_f32: np.ndarray, alpha_f32: np.ndarray) -> Image.Image:
    """Un-premultiply Premultiplied RGB + Alpha cleanly into a Pillow RGBA Image (zero dark fringe)."""
    a = np.clip(alpha_f32, 0.0, 1.0)
    a3 = a[..., None]
    rgb = np.where(a3 > 1e-4, np.clip(premul_f32 / np.maximum(a3, 1e-4), 0.0, 1.0), 0.0)
    h, w = a.shape[:2]
    rgba_u8 = np.empty((h, w, 4), dtype=np.uint8)
    rgba_u8[..., :3] = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgba_u8[..., 3] = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba_u8, mode="RGBA")


def warp_premultiplied_to_pil_rgba(
    rgb_f32: np.ndarray,
    alpha_f32: np.ndarray,
    H: np.ndarray,
    out_w: int,
    out_h: int,
) -> Image.Image:
    """Premultiplied-Alpha OpenCV perspective warp returned as a clean Pillow RGBA Image."""
    premul = (rgb_f32 * alpha_f32[..., None]).astype(np.float32)
    w_premul = cv2.warpPerspective(
        premul,
        H,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0),
    )
    w_alpha = cv2.warpPerspective(
        alpha_f32.astype(np.float32),
        H,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return premul_to_pil_rgba(w_premul, w_alpha)


def compose_studio_frame_pillow(
    studio_rgb: np.ndarray,
    shadow_alpha: np.ndarray,
    edge_premul: np.ndarray,
    edge_alpha: np.ndarray,
    face_pil_rgba: Image.Image,
    fx_add_rgb: np.ndarray,
    final_width: int,
    final_h: int,
) -> Image.Image:
    """
    Pillow-based RGBA layer compositing (studio -> drop shadow -> brushed metallic
    edge stack -> front 3D card face -> additive star glints/particles) + Lanczos downsample.
    """
    h, w = studio_rgb.shape[:2]
    studio_u8 = np.clip(studio_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    pil_comp = Image.fromarray(studio_u8, mode="RGB").convert("RGBA")

    sh_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    sh_rgba[..., 3] = np.clip(shadow_alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    pil_comp = Image.alpha_composite(pil_comp, Image.fromarray(sh_rgba, mode="RGBA"))

    if float(edge_alpha.max()) > 1e-4:
        pil_edge = premul_to_pil_rgba(edge_premul, edge_alpha)
        pil_comp = Image.alpha_composite(pil_comp, pil_edge)

    pil_comp = Image.alpha_composite(pil_comp, face_pil_rgba)

    pil_rgb = pil_comp.convert("RGB")
    if float(fx_add_rgb.max()) > 1e-4:
        fx_u8 = np.clip(fx_add_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
        pil_fx = Image.fromarray(fx_u8, mode="RGB")
        pil_rgb = ImageChops.add(pil_rgb, pil_fx)

    return pil_rgb.resize((final_width, final_h), Image.Resampling.LANCZOS)


def shade_card_face(
    theta: float,
    R: np.ndarray,
    card_data: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    etch_tex: np.ndarray,
    grating_tex: np.ndarray,
    uv_grid: Tuple[np.ndarray, np.ndarray],
    theme: ThemeConfig,
    line_etch_scale: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Evaluate the tarot holographic shader in card-local coordinates for phase theta.
    All patterns stay locked to the card; lighting, iridescence, sweep & emboss
    respond smoothly to the card's 3D orientation.
    Returns (shaded_rgb, sweep_center_pos).
    """
    u_norm, v_norm = uv_grid
    card_rgb = card_data["card_rgb"]
    bevel_mask = card_data["bevel_mask"]
    edge_nx = card_data["edge_nx"]
    edge_ny = card_data["edge_ny"]
    outer_frame = card_data["outer_frame_mask"]
    ornament = card_data["ornament_mask"]

    protect = masks["protect_mask"]
    dark_bg = masks["dark_bg_mask"]
    accent = masks["accent_mask"]
    emb_gx = masks["emb_gx"]
    emb_gy = masks["emb_gy"]

    inten = theme.intensity_scale

    # Card face world normal (local +Z points toward camera as -Z in our camera setup,
    # so use [0, 0, -1] for front-facing normal)
    face_normal = R @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    # Rotating studio key light (gentle orbital complement)
    light_dir = np.array([
        0.42 + 0.28 * math.sin(theta + 0.35),
        -0.52 + 0.18 * math.cos(theta),
        -0.78,
    ], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)
    view_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    # Half-vector in card-local space for physically responsive highlights
    half_vec = light_dir + view_dir
    half_vec /= np.linalg.norm(half_vec)
    local_light = R.T @ light_dir
    local_half = R.T @ half_vec

    # 1. Subtle 3D diffuse studio shading (keeps card grounded in 3D space)
    ndotl = float(np.clip(np.dot(face_normal, light_dir), 0.72, 1.0))
    # Subtle linear gradient across card due to perspective tilt
    tilt_grad = 1.0 + 0.055 * (u_norm - 0.5) * float(local_light[0]) + 0.045 * (v_norm - 0.5) * float(local_light[1])
    out = card_rgb * (0.91 + 0.11 * ndotl) * tilt_grad[..., None]

    # 2. Two-layer dynamic iridescent foil
    # Layer 1: broad diagonal foil wave shifted by yaw & pitch
    diag1 = u_norm * 0.76 + v_norm * 0.44
    phase1 = (
        theme.irid_phase_shift
        + diag1 * 5.2
        + float(local_half[0]) * 6.8
        + float(local_half[1]) * 4.2
        + grating_tex * 0.65
    )
    irid1 = spectral_iridescence(phase1)

    # Layer 2: counter-angled secondary diffraction wave
    diag2 = u_norm * 0.38 - v_norm * 0.82
    phase2 = (
        theme.irid_phase_shift + 1.85
        + diag2 * 6.4
        - float(local_half[0]) * 5.4
        + float(local_half[1]) * 5.8
        + etch_tex * 1.15
    )
    irid2 = spectral_iridescence(phase2)

    # Blend iridescence toward metal/gold harmony so it feels like collector foil, not neon rainbow
    irid_mix = (irid1 * 0.58 + irid2 * 0.42) * 0.72 + theme.metal_base[None, None, :] * 0.28

    # Angle-responsive foil visibility envelope
    angle_boost = 0.68 + 0.32 * abs(math.sin(theta + 0.3))
    non_protect = np.clip(1.0 - protect, 0.0, 1.0)
    foil_weight = (
        outer_frame * (0.36 + 0.45 * etch_tex + 0.24 * grating_tex)
        + ornament * 0.44
        + dark_bg * (0.16 + 0.34 * etch_tex + 0.14 * grating_tex)
        + non_protect * (1.0 - outer_frame) * (0.04 + 0.16 * etch_tex + 0.06 * grating_tex)
    ) * angle_boost * inten

    # Apply iridescent foil with gentle screen-like blend (prevents muddy darkening or blown whites)
    foil_add = irid_mix * foil_weight[..., None] * (1.0 - out * 0.38)
    out = out + foil_add

    # 3. Seamless periodic glass sweep band & moving local spotlight
    diag_coord = u_norm * 0.70 + v_norm * 0.30
    sweep_center = 0.50 + 0.64 * math.sin(theta - 0.35)
    sweep_band = np.exp(-0.5 * ((diag_coord - sweep_center) / 0.078) ** 2).astype(np.float32)

    # Broad soft moving spotlight linked to tilt
    spot_x = 0.50 + 0.30 * math.sin(theta + 0.15)
    spot_y = 0.46 - 0.22 * math.cos(theta + 0.25)
    spot = np.exp(-0.5 * (((u_norm - spot_x) / 0.36) ** 2 + ((v_norm - spot_y) / 0.38) ** 2)).astype(np.float32)

    # Etched metallic grooves ignite when sweep or spotlight passes over them
    etch_zone = np.clip(outer_frame * 0.95 + dark_bg * 0.90 + non_protect * 0.48, 0.0, 1.0) * (1.0 - protect * 0.88)
    etch_glint = etch_tex * etch_zone * (0.20 + 0.80 * np.clip(sweep_band * 1.25 + spot * 0.45, 0.0, 1.0)) * inten
    etch_color = irid_mix * 0.55 + theme.metal_bright[None, None, :] * 0.45
    out = out + etch_color * (etch_glint * 0.34)[..., None] * (1.0 - out * 0.28)

    # Sweep boosts strongly on gilded frame/ornaments, stays crisp & clean on protected subject/face
    sweep_mask = (
        0.045 * (1.0 - protect * 0.82)
        + accent * 0.34
        + ornament * 0.50
        + dark_bg * 0.12
    ) * inten
    sweep_color = theme.metal_bright * 0.65 + np.array([1.0, 1.0, 1.0], dtype=np.float32) * 0.35
    out = out + sweep_color[None, None, :] * (sweep_band * sweep_mask * 0.46)[..., None]

    # Multiplicative studio spotlight (keeps deep blacks rich, never adds milky haze over subject!)
    out = out * (0.96 + 0.06 * spot[..., None])
    spot_accent = (accent * 0.08 + dark_bg * 0.025) * inten
    out = out + (theme.metal_bright * 0.50 + 0.50)[None, None, :] * (spot * spot_accent)[..., None]

    # 4. Localized hot-stamping gold reflection & Sobel micro-emboss
    lx = float(local_light[0])
    ly = float(local_light[1])
    emboss = (emb_gx * lx + emb_gy * ly) * 0.38 * inten
    # Soft clamp emboss so there are no harsh double-edges or plastic artifacts
    emboss = np.tanh(emboss * 1.4) * 0.11
    emboss_tint = np.where(emboss[..., None] >= 0, theme.metal_bright[None, None, :], theme.metal_shadow[None, None, :])
    out = out + emboss_tint * np.abs(emboss)[..., None] * (0.45 + 0.55 * accent[..., None])

    # Localized gilded shimmer along ornament lines (varies spatially so whole border doesn't flash white)
    orn_shimmer = (0.35 + 0.65 * np.clip(sweep_band * 1.15 + spot * 0.55, 0.0, 1.0)) * ornament
    out = out + theme.metal_bright[None, None, :] * (orn_shimmer * 0.42 * inten)[..., None]

    # 4b. Character / Subject Linework Metallic Foil Etching & 3D Directional Line Relief
    if line_etch_scale > 0.01:
        line_mask = card_data["line_etch_mask"]
        line_gx = card_data["line_gx"]
        line_gy = card_data["line_gy"]

        # 1. Crisp 3D engraved groove relief: darken shadow lip & highlight lit lip
        #    (keeps ink sharp on white/light surfaces instead of washing lines out!)
        line_dot = np.tanh((line_gx * lx + line_gy * ly) * 1.9)
        groove_shadow = np.clip(-line_dot, 0.0, 1.0) * 0.28 * line_etch_scale * inten
        groove_lit = np.clip(line_dot, 0.0, 1.0) * 0.22 * line_etch_scale * inten
        out = out * (1.0 - groove_shadow[..., None] * np.clip(line_mask * 1.2, 0.0, 1.0)[..., None])

        # 2. Saturated metallic + prismatic foil color along the ink lines
        #    Boost chroma so foil glints pop unmistakably even on white dresses/hair
        sat_irid = np.clip((irid_mix - 0.48) * 1.85 + 0.50, 0.0, 1.0)
        foil_line_col = np.clip(theme.metal_base[None, None, :] * 0.42 + sat_irid * 0.58, 0.0, 1.0)
        foil_line_hot = np.clip(foil_line_col * 0.65 + theme.metal_bright[None, None, :] * 0.45, 0.0, 1.0)

        # Narrow, dynamic specular wave traveling along the linework as the card tilts
        angle_ripple = 0.58 + 0.42 * np.sin(phase1 * 1.15 - phase2 * 0.65)
        sweep_hit = np.clip((sweep_band * 1.35 + spot * 0.35 - 0.22) / 0.78, 0.0, 1.0) ** 1.25
        # Blend ink lines toward gleaming chromatic metallic foil where light catches them
        line_foil_alpha = np.clip(
            line_mask * (0.14 + 0.78 * sweep_hit * angle_ripple) * line_etch_scale * inten,
            0.0,
            0.88,
        )
        out = out * (1.0 - line_foil_alpha[..., None]) + foil_line_col * line_foil_alpha[..., None]

        # Crisp specular metallic lip right along the lit edge of the engraved stroke
        spec_lip = (groove_lit * (0.35 + 0.65 * sweep_hit) + (sweep_band ** 2.6) * line_mask * 0.42 * line_etch_scale * inten)
        out = out + foil_line_hot * spec_lip[..., None] * (0.45 + 0.55 * line_mask[..., None])

    # 5. Dynamic directional metallic bevel highlight on outer card rim
    # Dot perimeter outward normal with rotating light vector in card-local XY
    rim_dot = edge_nx * lx + edge_ny * ly
    rim_spec = np.clip(rim_dot * 0.5 + 0.5, 0.0, 1.0) ** 2.6
    rim_color = (
        theme.metal_shadow[None, None, :] * (1.0 - rim_spec[..., None])
        + theme.metal_bright[None, None, :] * rim_spec[..., None]
    )
    bevel_alpha = bevel_mask[..., None] * 0.88
    out = out * (1.0 - bevel_alpha) + rim_color * bevel_alpha

    # C^1 soft-knee highlight roll-off (protects against blown-out whites / harsh blocks)
    out = apply_highlight_rolloff(out, knee=0.90, spread=0.10)
    return out, sweep_center


def shade_metallic_edge_slice(
    R: np.ndarray,
    theta: float,
    card_data: Dict[str, np.ndarray],
    brushed_tex: np.ndarray,
    theme: ThemeConfig,
    slice_t: float,
) -> np.ndarray:
    """
    Shade one slice of the 8-layer solid brushed metallic edge.
    slice_t in [0, 1] from backmost (0) to frontmost (1).
    """
    edge_nx = card_data["edge_nx"]
    edge_ny = card_data["edge_ny"]

    light_dir = np.array([
        0.42 + 0.28 * math.sin(theta + 0.35),
        -0.52 + 0.18 * math.cos(theta),
        -0.78,
    ], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)
    local_light = R.T @ light_dir

    # Outward edge normal dot light direction -> realistic light/shadow along visible sidewall
    ndotl = edge_nx * float(local_light[0]) + edge_ny * float(local_light[1])
    diff = np.clip(ndotl * 0.55 + 0.45, 0.0, 1.0)
    spec = np.clip(ndotl * 0.5 + 0.5, 0.0, 1.0) ** 4.0

    # Slight front-to-back depth gradient so the edge reads as crisp beveled metal
    depth_shade = 0.72 + 0.28 * slice_t
    metal_col = (
        theme.metal_shadow[None, None, :] * (1.0 - diff[..., None])
        + theme.metal_base[None, None, :] * diff[..., None]
        + theme.metal_bright[None, None, :] * (spec[..., None] * 0.55)
    )
    metal_col *= brushed_tex[..., None] * depth_shade
    return np.clip(metal_col, 0.0, 1.0).astype(np.float32)


def render_studio_background(
    canvas_h: int,
    canvas_w: int,
    theta: float,
    float_y: float,
    rx: float,
    ry: float,
    theme: ThemeConfig,
    base_studio_bg: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render atmospheric studio backdrop + breathing rear halo + subtle celestial ring."""
    yy, xx = np.mgrid[0:canvas_h, 0:canvas_w].astype(np.float32)
    nx = (xx / max(1, canvas_w - 1) - 0.5) * 2.0
    ny = (yy / max(1, canvas_h - 1) - 0.5) * 2.0

    if base_studio_bg is not None:
        bg = base_studio_bg.copy()
    else:
        r2 = nx * nx * 0.95 + ny * ny * 0.82
        grad = np.exp(-r2 * 1.15)[..., None]
        bg = theme.bg_edge[None, None, :] * (1.0 - grad) + theme.bg_center[None, None, :] * grad

    # Subtle breathing halo behind card (moves gently with card)
    hx = 0.04 * math.sin(theta)
    hy = (float_y / canvas_h) * 1.6 - 0.03
    breath = 0.84 + 0.16 * math.sin(theta - 0.5)
    hr = np.sqrt((nx - hx) ** 2 + ((ny - hy) * (canvas_h / float(canvas_w))) ** 2)

    halo_scale = 0.14 if base_studio_bg is not None else 0.19
    halo1 = np.exp(-(hr ** 2) / 0.58) * halo_scale * breath
    halo2 = np.exp(-((hr - 0.52) ** 2) / 0.10) * 0.060 * breath
    bg = bg + theme.halo_primary[None, None, :] * halo1[..., None] + theme.halo_secondary[None, None, :] * halo2[..., None]

    # Very faint celestial ring silhouette in deep background (strictly background feel)
    ring_r = 0.86
    ring_line = np.exp(-((hr - ring_r) ** 2) / 0.00018) * 0.085 * breath
    bg = bg + theme.metal_base[None, None, :] * ring_line[..., None]

    if base_studio_bg is None:
        vig = np.clip(1.0 - 0.28 * (nx * nx + ny * ny), 0.52, 1.0)[..., None]
        bg = bg * vig
    return np.clip(bg, 0.0, 1.0).astype(np.float32)


def draw_star_glint(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    intensity: float,
    color: np.ndarray,
) -> None:
    """Additive 4-point cross star glint with soft core."""
    if intensity < 0.03 or radius < 2.0:
        return
    h, w = canvas.shape[:2]
    r_int = int(math.ceil(radius * 1.6))
    x0 = max(0, int(math.floor(cx)) - r_int)
    x1 = min(w, int(math.ceil(cx)) + r_int + 1)
    y0 = max(0, int(math.floor(cy)) - r_int)
    y1 = min(h, int(math.ceil(cy)) + r_int + 1)
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    dist = np.sqrt(dx * dx + dy * dy)

    # Soft glowing core
    core = np.exp(-(dist * dist) / (2.0 * (radius * 0.18) ** 2))
    # Crisp 4-point cross spikes
    spike_h = np.exp(-np.abs(dy) / max(0.85, radius * 0.065)) * np.exp(-np.abs(dx) / (radius * 0.58))
    spike_v = np.exp(-np.abs(dx) / max(0.85, radius * 0.065)) * np.exp(-np.abs(dy) / (radius * 0.58))
    glint = np.clip(core * 0.95 + (spike_h + spike_v) * 0.72, 0.0, 1.2) * intensity

    col = color * 0.65 + np.array([1.0, 1.0, 1.0], dtype=np.float32) * 0.35
    canvas[y0:y1, x0:x1] = np.clip(canvas[y0:y1, x0:x1] + glint[..., None] * col[None, None, :], 0.0, 1.0)


@dataclass
class ParticleField:
    x0: np.ndarray
    y0: np.ndarray
    amp_x: np.ndarray
    amp_y: np.ndarray
    phase: np.ndarray
    twinkle_k: np.ndarray
    twinkle_phase: np.ndarray
    radius: np.ndarray
    alpha_max: np.ndarray
    color: np.ndarray


def init_particles(
    n: int,
    canvas_w: int,
    canvas_h: int,
    theme: ThemeConfig,
    foreground: bool,
    seed: int,
) -> ParticleField:
    """Initialize fixed-seed periodic particle trajectories."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    while len(xs) < n:
        x = rng.uniform(0.05, 0.95) * canvas_w
        y = rng.uniform(0.05, 0.95) * canvas_h
        if foreground:
            # Keep foreground bokeh near the outer perimeter so it never obscures the subject
            nx = abs(x / canvas_w - 0.5)
            ny = abs(y / canvas_h - 0.5)
            if nx < 0.27 and ny < 0.32:
                continue
        xs.append(x)
        ys.append(y)

    x0 = np.array(xs, dtype=np.float32)
    y0 = np.array(ys, dtype=np.float32)
    scale = canvas_w / 920.0
    if foreground:
        amp_x = rng.uniform(10.0, 24.0, size=n).astype(np.float32) * scale
        amp_y = rng.uniform(12.0, 26.0, size=n).astype(np.float32) * scale
        radius = rng.uniform(4.2, 7.5, size=n).astype(np.float32) * scale
        alpha_max = rng.uniform(0.18, 0.32, size=n).astype(np.float32)
    else:
        amp_x = rng.uniform(5.0, 14.0, size=n).astype(np.float32) * scale
        amp_y = rng.uniform(6.0, 16.0, size=n).astype(np.float32) * scale
        radius = rng.uniform(1.6, 3.2, size=n).astype(np.float32) * scale
        alpha_max = rng.uniform(0.22, 0.45, size=n).astype(np.float32)

    phase = rng.uniform(0.0, 2.0 * math.pi, size=n).astype(np.float32)
    twinkle_k = rng.choice([1, 2], size=n).astype(np.float32)
    twinkle_phase = rng.uniform(0.0, 2.0 * math.pi, size=n).astype(np.float32)

    # Harmonize particle colors between metallic gold and secondary halo
    mix = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    col = theme.metal_bright[None, :] * mix + theme.halo_secondary[None, :] * (1.0 - mix)
    col = np.clip(col * 0.8 + 0.2, 0.0, 1.0)

    return ParticleField(x0, y0, amp_x, amp_y, phase, twinkle_k, twinkle_phase, radius, alpha_max, col)


def render_particles(canvas: np.ndarray, pf: ParticleField, theta: float) -> None:
    """Render periodic floating particles additively onto canvas."""
    h, w = canvas.shape[:2]
    for i in range(len(pf.x0)):
        cx = float(pf.x0[i] + pf.amp_x[i] * math.sin(theta + pf.phase[i]))
        cy = float(pf.y0[i] + pf.amp_y[i] * math.cos(theta + pf.phase[i] * 1.3))
        twinkle = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(pf.twinkle_k[i] * theta + pf.twinkle_phase[i]))
        alpha = float(pf.alpha_max[i] * twinkle)
        rad = float(pf.radius[i])

        r_int = int(math.ceil(rad * 2.6))
        x0 = max(0, int(math.floor(cx)) - r_int)
        x1 = min(w, int(math.ceil(cx)) + r_int + 1)
        y0 = max(0, int(math.floor(cy)) - r_int)
        y1 = min(h, int(math.ceil(cy)) + r_int + 1)
        if x1 <= x0 or y1 <= y0:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        glow = np.exp(-d2 / (2.0 * (rad * 0.65) ** 2)) * alpha
        canvas[y0:y1, x0:x1] = np.clip(
            canvas[y0:y1, x0:x1] + glow[..., None] * pf.color[i][None, None, :],
            0.0,
            1.0,
        )


# ---------------------------------------------------------------------------
# GIF Timing & Shared 256-Color Median-Cut Palette Quantization
# ---------------------------------------------------------------------------

def build_even_durations(n_frames: int = 48, n_50ms: int = 26) -> List[int]:
    """
    Distribute 50ms and 40ms frame durations evenly across n_frames using
    Bresenham-style error diffusion (default: 26x50ms + 22x40ms = 2.18s, ~22fps).
    """
    if n_frames != 48:
        n_50ms = int(round(n_frames * (26.0 / 48.0)))
    durations: List[int] = []
    acc = 0
    for _ in range(n_frames):
        acc += n_50ms
        if acc >= n_frames:
            acc -= n_frames
            durations.append(50)
        else:
            durations.append(40)
    return durations


def export_shared_palette_gif(
    frames_rgb_u8: List[np.ndarray],
    output_path: str,
    durations: List[int],
    quality_tier: str = "hd",
) -> None:
    """
    Tri-Subpalette Shared 256-Color Quantizer + Floyd-Steinberg Dithering:
      1. Allocates up to 54 Median-Cut palette entries strictly to dark pixels
         (lum < 72) so dark hair, shadows, and studio backdrops stay velvety smooth
         with zero dither speckle or frame-to-frame flicker.
      2. Allocates up to 36 Median-Cut entries to neutral cel whites, soft shading,
         and skin tones inside the card (`lum >= 72 & chroma < 58`) so anime cel
         surfaces stay clean without pinkish/grey Floyd-Steinberg dot noise.
      3. Allocates the remaining >=166 Fast-Octree palette entries from a 2x
         subject-prioritized frame mosaic so vivid prismatic holographic lasers
         retain smooth rainbow fidelity.
    """
    preset = GIF_QUALITY_PRESETS.get(quality_tier, GIF_QUALITY_PRESETS["hd"])
    sample_w = int(preset["palette_sample"])
    sharpen_pct = int(preset["sharpen_pct"])
    fh, fw = frames_rgb_u8[0].shape[:2]
    sample_h = max(80, int(round(sample_w * (fh / max(1.0, float(fw))))))

    n = len(frames_rgb_u8)
    stride = 1 if n <= 24 else 2
    sampled_idx = set(range(0, n, stride))
    tiles: List[Image.Image] = []
    pil_frames: List[Image.Image] = []
    dark_px_list: List[np.ndarray] = []
    cel_px_list: List[np.ndarray] = []

    cx0, cx1 = int(fw * 0.14), int(fw * 0.86)
    cy0, cy1 = int(fh * 0.10), int(fh * 0.90)

    def _extract_used_rgb(p_img: Image.Image) -> List[Tuple[int, int, int]]:
        pal = p_img.getpalette() or []
        out_cols: List[Tuple[int, int, int]] = []
        for p_idx in np.unique(np.asarray(p_img)):
            base = int(p_idx) * 3
            if base + 2 < len(pal):
                out_cols.append((pal[base], pal[base + 1], pal[base + 2]))
        return out_cols

    def _sample_mediancut_subpalette(px_chunks: List[np.ndarray], max_colors: int) -> List[Tuple[int, int, int]]:
        if not px_chunks:
            return []
        all_px = np.concatenate(px_chunks, axis=0)
        if len(all_px) < 1024:
            return []
        n_target = min(max_colors, max(12, len(all_px) // 256))
        side = int(math.sqrt(len(all_px)))
        sub_im = Image.fromarray(all_px[: side * side].reshape(side, side, 3), mode="RGB")
        p_sub = sub_im.quantize(colors=n_target, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        return _extract_used_rgb(p_sub)

    for idx, arr in enumerate(frames_rgb_u8):
        im = Image.fromarray(arr, mode="RGB")
        if sharpen_pct > 0:
            im = im.filter(ImageFilter.UnsharpMask(radius=0.9, percent=sharpen_pct, threshold=4))
        pil_frames.append(im)
        if idx in sampled_idx:
            tiles.append(im.resize((sample_w, sample_h), Image.Resampling.BILINEAR))
            card_crop = im.crop((cx0, cy0, cx1, cy1)).resize((sample_w, sample_h), Image.Resampling.BILINEAR)
            tiles.append(card_crop)
            tiles.append(card_crop)

            arr_s = np.asarray(im, dtype=np.uint8) if sharpen_pct > 0 else arr
            arr_f = arr_s.astype(np.float32)
            lum = 0.299 * arr_f[..., 0] + 0.587 * arr_f[..., 1] + 0.114 * arr_f[..., 2]
            dpx = arr_s[lum < 72.0]
            if len(dpx) > 0:
                step = max(1, len(dpx) // 4096)
                dark_px_list.append(dpx[::step])

            card_arr = arr_s[cy0:cy1, cx0:cx1]
            card_f = arr_f[cy0:cy1, cx0:cx1]
            clum = lum[cy0:cy1, cx0:cx1]
            c_chroma = np.max(card_f, axis=-1) - np.min(card_f, axis=-1)
            cpx = card_arr[(clum >= 72.0) & (c_chroma < 58.0)]
            if len(cpx) > 0:
                cstep = max(1, len(cpx) // 4096)
                cel_px_list.append(cpx[::cstep])

    cols = 8
    rows = math.ceil(len(tiles) / cols)
    mosaic = Image.new("RGB", (cols * sample_w, rows * sample_h))
    for i_t, tile in enumerate(tiles):
        r_idx, c_idx = divmod(i_t, cols)
        mosaic.paste(tile, (c_idx * sample_w, r_idx * sample_h))

    dark_colors = _sample_mediancut_subpalette(dark_px_list, max_colors=54)
    cel_colors = _sample_mediancut_subpalette(cel_px_list, max_colors=36)

    n_main = max(128, 256 - len(set(dark_colors + cel_colors)))
    try:
        p_main = mosaic.quantize(colors=n_main, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    except Exception:
        p_main = mosaic.quantize(colors=n_main, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    p_fallback = mosaic.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)

    seen = set()
    merged: List[int] = []
    for rgb in dark_colors + cel_colors + _extract_used_rgb(p_main) + _extract_used_rgb(p_fallback):
        if rgb not in seen:
            seen.add(rgb)
            merged.extend(rgb)
            if len(seen) >= 256:
                break
    while len(merged) < 768:
        merged.extend((0, 0, 0))

    palette_img = Image.new("P", (16, 16))
    palette_img.putpalette(merged[:768])

    quantized_frames = [
        im.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
        for im in pil_frames
    ]

    quantized_frames[0].save(
        output_path,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )


# ---------------------------------------------------------------------------
# Main Rendering Pipeline
# ---------------------------------------------------------------------------

class RenderCancelledError(Exception):
    """Raised when rendering is cancelled by the user."""


def render_tarot_gif(
    input_path: str,
    output_path: str,
    style: str = "auto",
    pattern: str = "auto",
    metal: str = "auto",
    intensity: str = "medium",
    bg_mode: str = "auto",
    line_etch: str = "medium",
    studio_bg: str = "auto",
    studio_bg_path: Optional[str] = None,
    subject_scale: float = 1.0,
    subject_offset_x: float = 0.0,
    subject_offset_y: float = 0.0,
    card_aspect: str = "auto",
    quality_tier: str = "hd",
    final_width: int = 480,
    n_frames: int = 48,
    supersample: int = 2,
    thickness_layers: int = 8,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, object]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    def _report(step: int, total: int, msg: str) -> None:
        if progress_callback is not None:
            progress_callback(step, total, msg)

    total_steps = n_frames + 2
    _report(0, total_steps, "正在分析原图、智能识别背景与构建 3D 卡片材质...")

    # Load original image respecting EXIF orientation (preserve alpha if present)
    pil_src = ImageOps.exif_transpose(Image.open(input_path))
    src_rgb, subject_alpha, bg_removed = extract_subject_alpha(pil_src, bg_mode=bg_mode)

    # Analyze theme on foreground subject pixels & auto-match style / pattern / metal
    theme = analyze_image_theme(
        src_rgb, style=style, pattern=pattern, metal=metal, intensity=intensity,
        fg_mask=subject_alpha if bg_removed else None,
    )

    ss_canvas_w = final_width * supersample
    # Initial card working width (~70% of supersampled canvas width)
    init_card_w = int(round(ss_canvas_w * 0.70))
    card_data = build_card_base_and_frame(
        src_rgb, init_card_w, theme, subject_alpha=subject_alpha, bg_removed=bg_removed,
        subject_scale=subject_scale, subject_offset_x=subject_offset_x,
        subject_offset_y=subject_offset_y, card_aspect=card_aspect,
    )
    card_h, card_w = card_data["card_rgb"].shape[:2]

    # Perspective focal length & physical card thickness (~1.7% of card width)
    focal = float(card_h * 2.65)
    thickness_total = float(card_w * 0.017)

    # Precompute exact safe canvas size across all 48 frames
    ss_canvas_w, ss_canvas_h, scale_adj = precompute_safe_canvas_size(
        card_w, card_h, n_frames, thickness_total, focal, ss_canvas_w
    )

    # Rebuild card at optimal exact scale so it fills the studio cleanly with safe margins
    if abs(scale_adj - 1.0) > 0.03:
        opt_card_w = int(round(init_card_w * scale_adj))
        card_data = build_card_base_and_frame(
            src_rgb, opt_card_w, theme, subject_alpha=subject_alpha, bg_removed=bg_removed,
            subject_scale=subject_scale, subject_offset_x=subject_offset_x,
            subject_offset_y=subject_offset_y, card_aspect=card_aspect,
        )
        card_h, card_w = card_data["card_rgb"].shape[:2]
        focal = float(card_h * 2.65)
        thickness_total = float(card_w * 0.017)
        ss_canvas_w, ss_canvas_h, _ = precompute_safe_canvas_size(
            card_w, card_h, n_frames, thickness_total, focal, final_width * supersample
        )

    final_h = int(round(ss_canvas_h / supersample))
    final_h += final_h % 2

    # Precompute card-local shader assets (fixed to card surface)
    masks = build_conservative_masks(
        card_data["card_rgb"],
        card_data["outer_frame_mask"],
        card_data["ornament_mask"],
        card_data["photo_mask"],
        bg_cutout_mask=card_data["bg_cutout_mask"],
        subject_alpha_full=card_data["subject_alpha_full"],
        bg_removed=bg_removed,
    )
    etch_tex, grating_tex = build_micro_etching_texture(card_h, card_w, theme.pattern_name)
    brushed_tex = build_brushed_metal_edge_texture(card_h, card_w)
    yy, xx = np.mgrid[0:card_h, 0:card_w].astype(np.float32)
    uv_grid = (xx / max(1, card_w - 1), yy / max(1, card_h - 1))

    anchors = select_anchored_glint_points(
        card_w, card_h, int(card_data["pad"][0]),
        card_data["ornament_mask"], masks["protect_mask"], masks["lum"],
    )

    # Fixed-seed background & foreground particle fields
    bg_particles = init_particles(26, ss_canvas_w, ss_canvas_h, theme, foreground=False, seed=2026)
    fg_particles = init_particles(6, ss_canvas_w, ss_canvas_h, theme, foreground=True, seed=9091)

    canvas_cx = ss_canvas_w * 0.5
    canvas_cy = ss_canvas_h * 0.5
    alpha_mask = card_data["alpha_mask"]

    frames_final: List[np.ndarray] = []

    # Prepare atmospheric studio backdrop behind the floating 3D card
    base_studio_bg: Optional[np.ndarray] = None
    if studio_bg_path and os.path.isfile(studio_bg_path):
        pil_st = ImageOps.exif_transpose(Image.open(studio_bg_path)).convert("RGB")
        st_arr = np.asarray(pil_st, dtype=np.float32) / 255.0
        base_studio_bg = cv2.resize(st_arr, (ss_canvas_w, ss_canvas_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    elif studio_bg != "obsidian_dark":
        from card_templates import STYLE_TO_STUDIO_BG, generate_studio_backdrop
        st_name = STYLE_TO_STUDIO_BG.get(theme.style_name, "starry_indigo") if studio_bg == "auto" else studio_bg
        base_studio_bg = generate_studio_backdrop(st_name, w=ss_canvas_w, h=ss_canvas_h)

    for i in range(n_frames):
        if cancel_event is not None and cancel_event.is_set():
            raise RenderCancelledError("渲染已取消")
        _report(i + 1, total_steps, f"正在渲染第 {i + 1}/{n_frames} 帧...")
        theta = 2.0 * math.pi * i / n_frames
        rx, ry, rz, float_y = compute_frame_pose(theta, card_h)
        R = rotation_matrix_xyz(rx, ry, rz)

        # 1. Studio background + breathing halo
        canvas = render_studio_background(
            ss_canvas_h, ss_canvas_w, theta, float_y, rx, ry, theme, base_studio_bg=base_studio_bg
        )

        # 2. Background stardust particles
        render_particles(canvas, bg_particles, theta)

        # 3. Dynamic floating drop shadow behind/below the card
        # Shadow shifts opposite to card tilt and softens as card floats higher
        float_norm = 0.5 - 0.5 * math.sin(theta + 0.45)  # higher card -> larger float_norm
        shadow_dx = -math.sin(ry) * card_w * 0.14
        shadow_dy = card_h * (0.065 + 0.022 * float_norm) - math.sin(rx) * card_h * 0.08
        H_shadow = plane_homography_for_z(
            card_w, card_h, R, thickness_total, float_y, focal,
            canvas_cx + shadow_dx, canvas_cy + shadow_dy,
        )
        shadow_sil = cv2.warpPerspective(
            alpha_mask, H_shadow, (ss_canvas_w, ss_canvas_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        blur_sigma = (22.0 + 10.0 * float_norm) * supersample
        shadow_soft = cv2.GaussianBlur(shadow_sil, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        shadow_opacity = 0.58 - 0.14 * float_norm
        shadow_alpha_map = np.clip(shadow_soft * shadow_opacity, 0.0, 1.0).astype(np.float32)

        # 4. 8-layer solid metallic thickness slices (Premultiplied-Alpha packed 4-ch warp, in-place)
        edge_base_rgba = np.empty((card_h, card_w, 4), dtype=np.float32)
        edge_base_rgba[..., :3] = (
            shade_metallic_edge_slice(R, theta, card_data, brushed_tex, theme, 1.0)
            * alpha_mask[..., None]
        )
        edge_base_rgba[..., 3] = alpha_mask
        edge_premul = np.zeros((ss_canvas_h, ss_canvas_w, 3), dtype=np.float32)
        edge_a = np.zeros((ss_canvas_h, ss_canvas_w), dtype=np.float32)
        inv_buf = np.empty((ss_canvas_h, ss_canvas_w), dtype=np.float32)
        one32 = np.float32(1.0)
        for k in range(thickness_layers, 0, -1):
            z_loc = thickness_total * (k / float(thickness_layers))
            slice_t = 1.0 - (k / float(thickness_layers))
            H_slice = plane_homography_for_z(
                card_w, card_h, R, z_loc, float_y, focal, canvas_cx, canvas_cy
            )
            w_rgba = cv2.warpPerspective(
                edge_base_rgba, H_slice, (ss_canvas_w, ss_canvas_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0, 0.0, 0.0),
            )
            w_a = np.clip(w_rgba[..., 3], 0.0, 1.0)
            np.subtract(one32, w_a, out=inv_buf)
            edge_premul *= inv_buf[..., None]
            w_rgba[..., :3] *= np.float32(0.72 + 0.28 * slice_t)
            edge_premul += w_rgba[..., :3]
            edge_a *= inv_buf
            edge_a += w_a

        # 5. Front face with tarot holographic shader (Premultiplied-Alpha warp -> Pillow RGBA)
        face_rgb, sweep_center = shade_card_face(
            theta, R, card_data, masks, etch_tex, grating_tex, uv_grid, theme,
            line_etch_scale=LINE_ETCH_SCALES.get(line_etch, 1.0),
        )
        H_front = plane_homography_for_z(
            card_w, card_h, R, 0.0, float_y, focal, canvas_cx, canvas_cy
        )
        face_pil_rgba = warp_premultiplied_to_pil_rgba(
            face_rgb, alpha_mask, H_front, ss_canvas_w, ss_canvas_h
        )

        # 6. Anchored card-face 4-point star glints & foreground bokeh into additive pass
        fx_add = np.zeros((ss_canvas_h, ss_canvas_w, 3), dtype=np.float32)
        for ax, ay, size_scale in anchors:
            u_a = ax / max(1, card_w - 1)
            v_a = ay / max(1, card_h - 1)
            diag_a = u_a * 0.70 + v_a * 0.30
            peak = math.exp(-0.5 * ((diag_a - sweep_center) / 0.13) ** 2)
            glint_int = (0.06 + 0.88 * (peak ** 1.6)) * theme.intensity_scale
            pt = H_front @ np.array([ax, ay, 1.0], dtype=np.float64)
            gx, gy = float(pt[0] / pt[2]), float(pt[1] / pt[2])
            glint_rad = (15.0 + 18.0 * peak) * size_scale * (ss_canvas_w / 920.0)
            draw_star_glint(fx_add, gx, gy, glint_rad, glint_int, theme.metal_bright)

        render_particles(fx_add, fg_particles, theta)

        # 7. Pillow RGBA compositing + Lanczos downsample
        pil_final = compose_studio_frame_pillow(
            canvas, shadow_alpha_map, edge_premul, edge_a, face_pil_rgba, fx_add, final_width, final_h
        )
        frames_final.append(np.asarray(pil_final, dtype=np.uint8))

    # Seam continuity check (Frame 47 -> Frame 0 vs average adjacent frame diff)
    adj_diffs = [
        float(np.mean(np.abs(frames_final[i + 1].astype(np.float32) - frames_final[i].astype(np.float32))))
        for i in range(n_frames - 1)
    ]
    seam_diff = float(np.mean(np.abs(frames_final[0].astype(np.float32) - frames_final[-1].astype(np.float32))))

    if cancel_event is not None and cancel_event.is_set():
        raise RenderCancelledError("渲染已取消")

    _report(n_frames + 1, total_steps, "正在构建共享 256 色调色板并导出 GIF...")
    durations = build_even_durations(n_frames=n_frames, n_50ms=26)
    export_shared_palette_gif(frames_final, output_path, durations, quality_tier=quality_tier)
    _report(total_steps, total_steps, "渲染完成！")

    total_ms = sum(durations)
    info = {
        "output_path": os.path.abspath(output_path),
        "style": theme.style_name,
        "pattern": theme.pattern_name,
        "metal": theme.metal_name,
        "intensity": theme.intensity_name,
        "bg_removed": bg_removed,
        "line_etch": line_etch,
        "output_size": (final_width, final_h),
        "frames": n_frames,
        "total_duration_s": round(total_ms / 1000.0, 2),
        "avg_fps": round(n_frames / (total_ms / 1000.0), 2),
        "avg_adj_diff": round(float(np.mean(adj_diffs)), 3),
        "seam_diff_47_to_0": round(seam_diff, 3),
        "file_size_kb": round(os.path.getsize(output_path) / 1024.0, 1),
    }
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Render 3D holographic floating tarot GIF.")
    parser.add_argument("--input", required=True, help="Path to source image")
    parser.add_argument("--output", default="tarot_holographic_float.gif", help="Output GIF path")
    parser.add_argument("--style", default="auto", choices=[k for k, *_ in STYLE_META])
    parser.add_argument("--pattern", default="auto", choices=[k for k, _ in PATTERN_META])
    parser.add_argument("--metal", default="auto", choices=[k for k, _ in METAL_META])
    parser.add_argument("--intensity", default="medium", choices=["subtle", "medium", "lavish"])
    parser.add_argument("--bg-mode", default="auto", choices=["auto", "remove", "keep"],
                        help="Background handling: auto (detect solid/alpha bg), remove (force cutout), keep")
    parser.add_argument("--line-etch", default="medium", choices=["off", "subtle", "medium", "strong"],
                        help="Metallic foil etching intensity along subject/character linework")
    parser.add_argument("--width", type=int, default=460, help="Final GIF width in pixels")
    parser.add_argument("--frames", type=int, default=48, help="Number of loop frames")
    parser.add_argument("--supersample", type=int, default=2, help="Supersampling factor (2 or 3)")
    args = parser.parse_args()

    info = render_tarot_gif(
        input_path=args.input,
        output_path=args.output,
        style=args.style,
        pattern=args.pattern,
        metal=args.metal,
        intensity=args.intensity,
        bg_mode=args.bg_mode,
        line_etch=args.line_etch,
        final_width=args.width,
        n_frames=args.frames,
        supersample=args.supersample,
    )
    print("Render complete:")
    for k, v in info.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
