#!/usr/bin/env python3
"""
3-Layer Depth-Driven SAR / UR Holographic + Quicksand (流麻) 3D Card Renderer
=============================================================================
Builds a collector-grade 3D floating looping GIF (`holographic_card.gif`) from:
  - Background layer (custom image OR auto-generated celestial template)
  - Midground subject layer (transparent RGBA PNG OR auto-cutout solid-bg art)
  - Foreground ornate frame layer (transparent RGBA PNG OR auto-generated frame)
  - Monocular depth map (optional grayscale depth map aligned via ECC Affine
    Transform + edge extrapolation + Bilateral Filter, OR auto-synthesized)

Key visual features:
  - 3-layer in-card parallax window (zero edge gaps, exact 0° frontal alignment)
  - 8-layer brushed metallic card sidewall + dynamic bevel highlight
  - Relief Mapping (vectorized perspective ray-marching) for 3D midground volume
  - Background Quicksand / 流麻 interlayer (2 mica-sand streams + 300 3D glitters
    with independent random 3D surface normals)
  - Luminous Half-Lambert 3D lighting (clamped slopes, sidewall >= 0.90)
  - SAR ink micro-emboss, metallic specular with smooth roll-off & Fresnel rim
  - Unified cast shadows + Premultiplied-Alpha camera-space Depth of Field
  - 48-frame strictly seamless loop (~22fps) with shared 256-color Median Cut GIF

Dependencies:
    pip install opencv-python-headless numpy pillow
"""

from __future__ import annotations

import argparse
import math
import os
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from card_templates import (
    BG_TEMPLATE_META,
    FG_TEMPLATE_META,
    STUDIO_BG_META,
    STYLE_TO_STUDIO_BG,
    STYLE_TO_TEMPLATES,
    align_and_refine_depth_map,
    ensure_default_templates,
    generate_background_template,
    generate_foreground_frame_template,
    generate_studio_backdrop,
    synthesize_auto_depth_map,
)
from depth_estimator import estimate_depth_ai, estimate_depth_fast, is_ai_depth_available
from render_tarot_holographic import (
    INTENSITY_SCALES,
    LINE_ETCH_SCALES,
    METAL_META,
    PATTERN_META,
    STYLE_META,
    RenderCancelledError,
    ThemeConfig,
    analyze_image_theme,
    build_brushed_metal_edge_texture,
    build_even_durations,
    build_micro_etching_texture,
    build_rounded_rect_sdf,
    draw_star_glint,
    export_shared_palette_gif,
    extract_subject_alpha,
    init_particles,
    plane_homography_for_z,
    render_particles,
    render_studio_background,
    rotation_matrix_xyz,
    shade_metallic_edge_slice,
    spectral_iridescence,
)


# ---------------------------------------------------------------------------
# Periodic 3D Pose (Guarantees exact 0° frontal frame & both +/-17.5° peaks)
# ---------------------------------------------------------------------------

def compute_sar_frame_pose(theta: float, card_h: float) -> Tuple[float, float, float, float]:
    """
    Strictly 2*pi-periodic 3D orbital pose:
      - theta = 0 (frame 0)  -> exact 0° frontal view (rx=0, ry=0, rz=0)
      - theta = pi/2, 3pi/2  -> +/-17.5° Y-axis max tilt peaks
      - X-axis pitch +/-10.5° via sin(2*theta) for natural 3D figure-8 orbit
      - Z-axis roll +/-2.0° via sin(theta)
      - Gentle vertical float bob via sin(theta)
    """
    ry = math.radians(17.5) * math.sin(theta)
    rx = math.radians(10.5) * math.sin(2.0 * theta)
    rz = math.radians(2.0) * math.sin(theta)
    float_y = card_h * 0.022 * math.sin(theta + 0.45)
    return rx, ry, rz, float_y


def precompute_sar_canvas_size(
    card_w: int,
    card_h: int,
    n_frames: int,
    thickness_total: float,
    focal: float,
    target_canvas_w: int,
) -> Tuple[int, int, float]:
    min_x, max_x = 1e9, -1e9
    min_y, max_y = 1e9, -1e9
    cx0 = (card_w - 1.0) * 0.5
    cy0 = (card_h - 1.0) * 0.5
    corners = [(-cx0, -cy0), (cx0, -cy0), (cx0, cy0), (-cx0, cy0)]

    for i in range(n_frames):
        theta = 2.0 * math.pi * i / n_frames
        rx, ry, rz, float_y = compute_sar_frame_pose(theta, card_h)
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
    scale_adj = (target_canvas_w * 0.74) / proj_w
    canvas_h = int(round((proj_h * scale_adj) / 0.76))
    canvas_h += canvas_h % 2
    return target_canvas_w, canvas_h, scale_adj


# ---------------------------------------------------------------------------
# Premultiplied-Alpha Gaussian Blur (Zero Grey Halo / Color Bleed for DoF)
# ---------------------------------------------------------------------------

def blur_premultiplied_rgba(
    rgb: np.ndarray,
    alpha: np.ndarray,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if sigma < 0.25:
        return rgb, alpha
    premult = rgb * alpha[..., None]
    premult_b = cv2.GaussianBlur(premult, (0, 0), sigmaX=sigma, sigmaY=sigma)
    alpha_b = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
    rgb_out = np.where(alpha_b[..., None] > 1e-4, premult_b / np.maximum(alpha_b[..., None], 1e-4), 0.0)
    return np.clip(rgb_out, 0.0, 1.0).astype(np.float32), np.clip(alpha_b, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Background Quicksand (流麻) Interlayer: 2 Mica Streams + 300 3D Glitters
# ---------------------------------------------------------------------------

@dataclass
class QuicksandInterlayer:
    sand_tex1: np.ndarray
    sand_tex2: np.ndarray
    glitter_u: np.ndarray
    glitter_v: np.ndarray
    glitter_z: np.ndarray
    glitter_normals: np.ndarray   # (N, 3) random unit 3D normals
    glitter_radius: np.ndarray
    glitter_amp_u: np.ndarray
    glitter_amp_v: np.ndarray
    glitter_phase: np.ndarray
    glitter_color: np.ndarray     # (N, 3)
    glitter_is_star: np.ndarray   # (N,) bool for shaped 4-pt star/diamond sequins


def init_quicksand_interlayer(
    card_h: int,
    card_w: int,
    theme: ThemeConfig,
    n_glitters: int = 300,
    seed: int = 777,
) -> QuicksandInterlayer:
    """
    Build two tileable fine pearlescent mica-sand flow textures and ~300
    suspended 3D micro-glitter flakes (including ~48 holographic star/diamond
    sequins) with independent random 3D unit normals.
    """
    rng = np.random.default_rng(seed)

    # Fine pearlescent mica-sand wave textures (smooth, tileable via np.roll)
    yy, xx = np.mgrid[0:card_h, 0:card_w].astype(np.float32)
    u = xx / max(1, card_w)
    v = yy / max(1, card_h)

    # Stream 1: diagonal silky sand veins + fine shimmer grain
    vein1 = (0.5 + 0.5 * np.sin((u * 3.0 + v * 2.0 + 0.18 * np.sin(v * 6.0 * math.pi)) * 2.0 * math.pi)) ** 2.6
    grain1 = rng.uniform(0.0, 1.0, size=(card_h // 2, card_w // 2)).astype(np.float32)
    grain1 = cv2.resize(grain1, (card_w, card_h), interpolation=cv2.INTER_LINEAR)
    grain1 = cv2.GaussianBlur(grain1, (0, 0), sigmaX=1.1)
    sand1 = np.clip(vein1 * 0.55 + (grain1 ** 2.2) * vein1 * 0.95, 0.0, 1.0).astype(np.float32)

    # Stream 2: counter-flowing silk shimmer current
    vein2 = (0.5 + 0.5 * np.sin((u * 2.0 - v * 3.0 + 0.16 * np.cos(u * 4.0 * math.pi)) * 2.0 * math.pi)) ** 2.8
    grain2 = rng.uniform(0.0, 1.0, size=(card_h // 2, card_w // 2)).astype(np.float32)
    grain2 = cv2.resize(grain2, (card_w, card_h), interpolation=cv2.INTER_LINEAR)
    grain2 = cv2.GaussianBlur(grain2, (0, 0), sigmaX=1.3)
    sand2 = np.clip(vein2 * 0.48 + (grain2 ** 2.5) * vein2 * 0.95, 0.0, 1.0).astype(np.float32)

    # 300 3D micro-glitter flakes with independent random 3D surface normals
    g_u = rng.uniform(0.04, 0.96, size=n_glitters).astype(np.float32)
    g_v = rng.uniform(0.04, 0.96, size=n_glitters).astype(np.float32)
    g_z = rng.uniform(0.28, 0.92, size=n_glitters).astype(np.float32)

    # Random unit normals biased slightly toward front hemisphere (-Z) so glitters catch light nicely
    raw_n = rng.normal(0.0, 1.0, size=(n_glitters, 3)).astype(np.float32)
    raw_n[:, 2] = -np.abs(raw_n[:, 2]) - 0.32
    raw_n /= np.linalg.norm(raw_n, axis=1, keepdims=True) + 1e-6

    scale = card_w / 680.0
    g_is_star = rng.random(size=n_glitters) < 0.16
    g_rad = np.where(
        g_is_star,
        rng.uniform(2.6, 4.2, size=n_glitters),
        rng.uniform(1.3, 2.5, size=n_glitters),
    ).astype(np.float32) * scale
    g_amp_u = rng.uniform(0.008, 0.025, size=n_glitters).astype(np.float32)
    g_amp_v = rng.uniform(0.010, 0.028, size=n_glitters).astype(np.float32)
    g_phase = rng.uniform(0.0, 2.0 * math.pi, size=n_glitters).astype(np.float32)

    # Harmonize glitter colors: pearlescent gold / icy silver / cyan-violet accents
    w1 = rng.uniform(0.0, 1.0, size=(n_glitters, 1)).astype(np.float32)
    w2 = rng.uniform(0.0, 0.45, size=(n_glitters, 1)).astype(np.float32)
    pearlescent = np.array([0.82, 0.93, 1.00], dtype=np.float32)[None, :]
    g_col = theme.metal_bright[None, :] * w1 + theme.halo_secondary[None, :] * (1.0 - w1)
    g_col = g_col * (1.0 - w2) + pearlescent * w2
    g_col = np.clip(g_col * 0.85 + 0.18, 0.0, 1.0)

    return QuicksandInterlayer(
        sand_tex1=sand1,
        sand_tex2=sand2,
        glitter_u=g_u,
        glitter_v=g_v,
        glitter_z=g_z,
        glitter_normals=raw_n,
        glitter_radius=g_rad,
        glitter_amp_u=g_amp_u,
        glitter_amp_v=g_amp_v,
        glitter_phase=g_phase,
        glitter_color=g_col,
        glitter_is_star=g_is_star,
    )


def render_quicksand_layer(
    canvas: np.ndarray,
    qs: QuicksandInterlayer,
    theta: float,
    R: np.ndarray,
    px: float,
    py: float,
    half_vec: np.ndarray,
    light_dir: np.ndarray,
    shadow_map: np.ndarray,
    theme: ThemeConfig,
    strength: float,
) -> None:
    """
    Composite the two flowing mica-sand streams and 300 3D normal-lit micro-glitters
    (with chromatic normal-driven iridescence & shaped star sequins) into the
    in-card interlayer, modulated by subject shadow for realistic depth.
    """
    if strength <= 0.01:
        return
    h, w = canvas.shape[:2]

    # 1. Two periodic + gravity/tilt-coupled mica-sand flow layers
    sh1_x = int(round((-px * 0.052 + 0.024 * math.sin(theta)) * w))
    sh1_y = int(round((-py * 0.052 + 0.020 * math.cos(theta)) * h))
    sh2_x = int(round((-px * 0.036 - 0.020 * math.cos(theta)) * w))
    sh2_y = int(round((-py * 0.036 + 0.026 * math.sin(theta)) * h))

    s1 = np.roll(np.roll(qs.sand_tex1, sh1_y, axis=0), sh1_x, axis=1)
    s2 = np.roll(np.roll(qs.sand_tex2, sh2_y, axis=0), sh2_x, axis=1)

    shadow_atten = (1.0 - shadow_map * 0.58)[..., None]
    sand_col1 = theme.metal_bright * 0.65 + np.array([0.92, 0.88, 0.78], dtype=np.float32) * 0.35
    sand_col2 = theme.halo_primary * 0.45 + np.array([0.74, 0.90, 0.98], dtype=np.float32) * 0.55

    sand_add = (
        sand_col1[None, None, :] * (s1 * 0.22 * strength)[..., None]
        + sand_col2[None, None, :] * (s2 * 0.18 * strength)[..., None]
    ) * shadow_atten
    canvas += sand_add

    # 2. 300 suspended 3D glitters with independent random 3D normals + chromatic shift
    rot_n = (R @ qs.glitter_normals.T).T  # (N, 3)
    ndoth = np.clip(rot_n @ half_vec, 0.0, 1.0)
    ndotl = np.clip(rot_n @ light_dir, 0.0, 1.0)
    sparkle = (ndoth ** 11.0) * 1.25 + (ndotl ** 4.0) * 0.24 + 0.05

    # Per-flake iridescent hue shift driven by rotated 3D normal orientation
    g_phase_col = (rot_n[:, 0] * 3.8 - rot_n[:, 1] * 3.2 + qs.glitter_phase)
    irid_flake = np.stack([
        0.58 + 0.40 * np.cos(g_phase_col),
        0.56 + 0.38 * np.cos(g_phase_col + 2.05),
        0.62 + 0.38 * np.cos(g_phase_col + 4.05),
    ], axis=-1).astype(np.float32)
    flake_cols = np.clip(qs.glitter_color * 0.58 + irid_flake * 0.48, 0.0, 1.0)

    # Strictly periodic gentle drift + stratified 3D depth parallax
    cur_u = qs.glitter_u - px * 0.072 * qs.glitter_z + qs.glitter_amp_u * np.sin(theta + qs.glitter_phase)
    cur_v = qs.glitter_v - py * 0.072 * qs.glitter_z + qs.glitter_amp_v * np.cos(theta + qs.glitter_phase * 1.2)
    cur_u = np.mod(cur_u, 1.0)
    cur_v = np.mod(cur_v, 1.0)

    gx_arr = cur_u * (w - 1)
    gy_arr = cur_v * (h - 1)

    for i in range(len(gx_arr)):
        inten_i = float(sparkle[i]) * strength
        if inten_i < 0.07:
            continue
        cx = float(gx_arr[i])
        cy = float(gy_arr[i])
        ix = int(np.clip(round(cx), 0, w - 1))
        iy = int(np.clip(round(cy), 0, h - 1))
        sh_a = 1.0 - float(shadow_map[iy, ix]) * 0.58
        inten_i *= sh_a

        is_star = bool(qs.glitter_is_star[i])
        rad = float(qs.glitter_radius[i]) * (0.85 + 0.45 * min(1.0, inten_i))
        r_int = max(2, int(math.ceil(rad * (2.3 if is_star else 1.8))))
        x0 = max(0, ix - r_int)
        x1 = min(w, ix + r_int + 1)
        y0 = max(0, iy - r_int)
        y1 = min(h, iy + r_int + 1)
        if x1 <= x0 or y1 <= y0:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx = xx - cx
        dy = yy - cy
        d2 = dx * dx + dy * dy
        if is_star:
            # Faceted holographic 4-point star / diamond sequin
            core = np.exp(-d2 / (2.0 * (rad * 0.36) ** 2))
            cross = (
                np.exp(-np.abs(dy) / max(0.75, rad * 0.18)) * np.exp(-np.abs(dx) / (rad * 1.25))
                + np.exp(-np.abs(dx) / max(0.75, rad * 0.18)) * np.exp(-np.abs(dy) / (rad * 1.25))
            ) * 0.72
            dot = np.clip(core + cross, 0.0, 1.25)
        else:
            dot = np.exp(-d2 / (2.0 * (rad * 0.52) ** 2))
            if inten_i > 0.62:
                cross = (
                    np.exp(-np.abs(dy) / 0.8) * np.exp(-np.abs(dx) / (rad * 1.5))
                    + np.exp(-np.abs(dx) / 0.8) * np.exp(-np.abs(dy) / (rad * 1.5))
                ) * 0.40
                dot = np.clip(dot + cross, 0.0, 1.2)
        canvas[y0:y1, x0:x1] += (dot * min(1.10, inten_i))[..., None] * flake_cols[i][None, None, :]


# ---------------------------------------------------------------------------
# Perspective Ray-Marching (Relief Mapping) for 3D Midground Subject
# ---------------------------------------------------------------------------

def prepare_edge_extended_subject(
    subj_rgb: np.ndarray,
    subj_alpha: np.ndarray,
    pad_px: int = 5,
) -> np.ndarray:
    """
    Extrapolate valid subject silhouette colors outward by `pad_px` pixels via
    nearest-neighbor distance transform so narrow 3D relief sidewalls always
    inherit clean edge colors instead of black/transparent bleed.
    """
    valid = subj_alpha > 0.35
    if not np.any(valid) or np.all(valid):
        return subj_rgb.copy()
    inv = (~valid).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    coords = np.argwhere(valid)
    idx = np.clip(labels - 1, 0, len(coords) - 1)
    ext_rgb = subj_rgb[coords[idx, 0], coords[idx, 1]]
    k = max(3, pad_px * 2 + 1)
    near_edge = cv2.dilate(valid.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    return np.where(valid[..., None], subj_rgb, np.where(near_edge[..., None], ext_rgb, subj_rgb)).astype(np.float32)


def relief_map_subject(
    depth_map: np.ndarray,
    subj_rgb_ext: np.ndarray,
    subj_alpha: np.ndarray,
    depth_nx: np.ndarray,
    depth_ny: np.ndarray,
    line_mask: np.ndarray,
    line_gx: np.ndarray,
    line_gy: np.ndarray,
    metal_mask: np.ndarray,
    px: float,
    py: float,
    relief_px: float,
    base_xx: np.ndarray,
    base_yy: np.ndarray,
    n_steps: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Vectorized Relief Mapping (Perspective Ray-Marching + 2-step binary refinement):
      - At 0° frontal view (px=py=0): exact 1:1 pixel identity with original art.
      - At tilted angles: produces genuine perspective parallax across body parts,
        limb/head/wing self-occlusion, and clean edge-color-extended sidewalls.
    """
    dx_max = px * relief_px
    dy_max = py * relief_px

    if abs(dx_max) < 0.05 and abs(dy_max) < 0.05:
        return {
            "rgb": subj_rgb_ext.copy(),
            "alpha": subj_alpha.copy(),
            "nx": depth_nx.copy(),
            "ny": depth_ny.copy(),
            "line_mask": line_mask.copy(),
            "line_gx": line_gx.copy(),
            "line_gy": line_gy.copy(),
            "metal_mask": metal_mask.copy(),
            "sidewall": np.zeros_like(subj_alpha),
            "u_hit": base_xx.copy(),
            "v_hit": base_yy.copy(),
            "hit_h": depth_map.copy(),
        }

    h, w = depth_map.shape[:2]
    hit_t = np.ones((h, w), dtype=np.float32)
    prev_t = np.zeros((h, w), dtype=np.float32)
    has_hit = np.zeros((h, w), dtype=bool)

    # Subtle radial perspective expansion tied to tilt magnitude (0 at exact 0° frontal view!)
    tilt_mag = min(1.0, math.hypot(px, py) / 0.32)
    rad_scale = 0.032 * tilt_mag * (relief_px / max(1.0, w * 0.135))
    rad_dx = (base_xx - (w - 1) * 0.5) * rad_scale
    rad_dy = (base_yy - (h - 1) * 0.5) * rad_scale

    # Coarse linear ray-march from front (h=1.0, t=0) to back (h=0.0, t=1)
    # Pivot around mid-depth t=0.5 with physically correct perspective sign (off = 0.5 - t):
    # near parts (h=1, off=+0.5) shift with foreground parallax & subtly loom closer!
    dt = 1.0 / float(n_steps)
    for s in range(n_steps + 1):
        t = s * dt
        ray_h = 1.0 - t
        off = 0.5 - t
        map_x = (base_xx + off * dx_max - off * rad_dx).astype(np.float32)
        map_y = (base_yy + off * dy_max - off * rad_dy).astype(np.float32)
        h_sample = cv2.remap(
            depth_map, map_x, map_y,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        newly_hit = (~has_hit) & (h_sample >= ray_h) & (h_sample > 0.04)
        if np.any(newly_hit):
            hit_t[newly_hit] = t
            prev_t[newly_hit] = max(0.0, t - dt)
            has_hit |= newly_hit

    # 3-step binary refinement for silky smooth, artifact-free surface intersections
    lo = prev_t
    hi = hit_t
    for _ in range(3):
        mid = 0.5 * (lo + hi)
        ray_h = 1.0 - mid
        off = 0.5 - mid
        map_x = (base_xx + off * dx_max - off * rad_dx).astype(np.float32)
        map_y = (base_yy + off * dy_max - off * rad_dy).astype(np.float32)
        h_sample = cv2.remap(
            depth_map, map_x, map_y,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        hit_mid = has_hit & (h_sample >= ray_h)
        hi = np.where(hit_mid, mid, hi)
        lo = np.where(hit_mid, lo, mid)

    final_t = np.where(has_hit, hi, 0.5)
    final_off = 0.5 - final_t
    u_hit = (base_xx + final_off * dx_max - final_off * rad_dx).astype(np.float32)
    v_hit = (base_yy + final_off * dy_max - final_off * rad_dy).astype(np.float32)

    w_rgb = cv2.remap(subj_rgb_ext, u_hit, v_hit, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    w_alpha = cv2.remap(subj_alpha, u_hit, v_hit, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    w_alpha = np.where(has_hit, w_alpha, 0.0).astype(np.float32)

    # Identify narrow sidewall grazing pixels (where ray hit steep depth jump)
    h_hit = cv2.remap(depth_map, u_hit, v_hit, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    sidewall = np.clip((h_hit - (1.0 - final_t)) * 5.0, 0.0, 1.0) * w_alpha

    return {
        "rgb": w_rgb,
        "alpha": w_alpha,
        "nx": cv2.remap(depth_nx, u_hit, v_hit, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0),
        "ny": cv2.remap(depth_ny, u_hit, v_hit, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0),
        "line_mask": cv2.remap(line_mask, u_hit, v_hit, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0),
        "line_gx": cv2.remap(line_gx, u_hit, v_hit, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0),
        "line_gy": cv2.remap(line_gy, u_hit, v_hit, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0),
        "metal_mask": cv2.remap(metal_mask, u_hit, v_hit, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0),
        "sidewall": sidewall.astype(np.float32),
        "u_hit": u_hit,
        "v_hit": v_hit,
        "hit_h": h_hit.astype(np.float32),
    }


def compute_heightfield_self_shadow(
    depth_map: np.ndarray,
    u_hit: np.ndarray,
    v_hit: np.ndarray,
    hit_h: np.ndarray,
    lx: float,
    ly: float,
    step_span_px: float,
) -> np.ndarray:
    """
    Fast 5-tap screen-space heightfield self-shadowing: marches from each visible
    3D surface point toward the rotating studio light (lx, ly) so foreground limbs,
    hands, bangs, sleeves, and front skirt folds cast soft 3D contact shadows onto
    deeper parts (rear hair, inner wings, dragon tail).
    Returns shadow intensity in [0, 1].
    """
    occ = np.zeros_like(hit_h, dtype=np.float32)
    for k in range(1, 6):
        frac = k / 5.0
        sx = (u_hit + lx * step_span_px * frac).astype(np.float32)
        sy = (v_hit + ly * step_span_px * frac).astype(np.float32)
        h_s = cv2.remap(depth_map, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        # Higher occluder along light direction blocks light softly
        delta = h_s - (hit_h + 0.055 + 0.045 * frac)
        occ = np.maximum(occ, np.clip(delta * 3.8, 0.0, 1.0) * (1.0 - 0.12 * (k - 1)))
    return cv2.GaussianBlur(occ, (0, 0), sigmaX=1.8).astype(np.float32)


# ---------------------------------------------------------------------------
# Main 3-Layer + Depth + Quicksand SAR Rendering Pipeline
# ---------------------------------------------------------------------------

def _auto_select_templates(style_name: str) -> Tuple[str, str]:
    return STYLE_TO_TEMPLATES.get(style_name, ("classical_arcana", "classical_filigree"))


def render_3layer_sar_gif(
    subject_path: Optional[str],
    output_path: str = "holographic_card.gif",
    bg_path: Optional[str] = None,
    fg_path: Optional[str] = None,
    depth_path: Optional[str] = None,
    bg_template: str = "auto",
    fg_template: str = "auto",
    studio_bg: str = "auto",
    studio_bg_path: Optional[str] = None,
    style: str = "auto",
    pattern: str = "auto",
    metal: str = "auto",
    intensity: str = "medium",
    bg_mode: str = "auto",
    line_etch: str = "medium",
    depth_engine: str = "auto",
    quicksand_strength: float = 1.0,
    relief_strength: float = 1.0,
    enable_dof: bool = True,
    final_width: int = 460,
    n_frames: int = 48,
    supersample: int = 2,
    thickness_layers: int = 8,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    single_frame_theta: Optional[float] = None,
) -> Dict[str, object]:
    """
    Render a 3-layer depth-driven SAR / UR Holographic + Quicksand (流麻) 3D
    floating looping GIF. When `single_frame_theta` is provided, renders only
    that single 3D angle and returns `{"preview_image": PIL.Image, ...}` without
    writing a GIF file (used for instant WYSIWYG GUI previews).
    """
    if single_frame_theta is None and (not subject_path or not os.path.exists(subject_path)):
        raise FileNotFoundError(f"Subject image not found: {subject_path}")

    def _report(step: int, total: int, msg: str) -> None:
        if progress_callback is not None:
            progress_callback(step, total, msg)

    total_steps = n_frames + 3
    _report(0, total_steps, "正在加载图层、准备默认模板与智能对齐单目深度图...")

    # Ensure reusable template files exist on disk
    tpl_paths = ensure_default_templates()

    # 1. Load midground subject & extract clean alpha (or empty preview subject if none selected yet)
    if subject_path and os.path.isfile(subject_path):
        pil_subj = ImageOps.exif_transpose(Image.open(subject_path))
        subj_rgb_raw, subj_alpha_raw, bg_removed = extract_subject_alpha(pil_subj, bg_mode=bg_mode)
        src_h, src_w = subj_rgb_raw.shape[:2]
    else:
        src_w, src_h = 560, 896
        subj_rgb_raw = np.zeros((src_h, src_w, 3), dtype=np.float32)
        subj_alpha_raw = np.zeros((src_h, src_w), dtype=np.float32)
        bg_removed = True

    # Analyze theme from foreground subject pixels (or fallback neutral if empty preview)
    theme = analyze_image_theme(
        subj_rgb_raw if np.any(subj_alpha_raw > 0.1) else np.full((32, 32, 3), 0.5, dtype=np.float32),
        style=style, pattern=pattern, metal=metal, intensity=intensity,
        fg_mask=subj_alpha_raw if (bg_removed and np.any(subj_alpha_raw > 0.1)) else None,
    )
    inten = theme.intensity_scale
    line_etch_scale = LINE_ETCH_SCALES.get(line_etch, 1.0)

    auto_bg_tpl, auto_fg_tpl = _auto_select_templates(theme.style_name)
    chosen_bg_tpl = auto_bg_tpl if bg_template == "auto" else bg_template
    chosen_fg_tpl = auto_fg_tpl if fg_template == "auto" else fg_template

    # 2. Compute optimal supersampled card & canvas dimensions
    ss_canvas_w = final_width * supersample
    init_card_w = int(round(ss_canvas_w * 0.70))
    pad = max(16, int(round(init_card_w * 0.045)))
    inner_w = init_card_w - 2 * pad
    inner_h = int(round(inner_w * (src_h / float(src_w))))
    card_w = inner_w + 2 * pad
    card_h = inner_h + 2 * pad

    focal = float(card_h * 2.65)
    thickness_total = float(card_w * 0.017)
    ss_canvas_w, ss_canvas_h, scale_adj = precompute_sar_canvas_size(
        card_w, card_h, n_frames, thickness_total, focal, ss_canvas_w
    )
    if abs(scale_adj - 1.0) > 0.03:
        opt_card_w = int(round(init_card_w * scale_adj))
        pad = max(16, int(round(opt_card_w * 0.045)))
        inner_w = opt_card_w - 2 * pad
        inner_h = int(round(inner_w * (src_h / float(src_w))))
        card_w = inner_w + 2 * pad
        card_h = inner_h + 2 * pad
        focal = float(card_h * 2.65)
        thickness_total = float(card_w * 0.017)
        ss_canvas_w, ss_canvas_h, _ = precompute_sar_canvas_size(
            card_w, card_h, n_frames, thickness_total, focal, final_width * supersample
        )

    final_h = int(round(ss_canvas_h / supersample))
    final_h += final_h % 2

    # 3. Prepare card-local geometry (anti-aliased rounded corners & metallic bevel)
    radius = float(max(14, int(round(min(card_w, card_h) * 0.044))))
    sdf, edge_nx, edge_ny = build_rounded_rect_sdf(card_h, card_w, radius)
    alpha_mask = np.clip(0.5 - sdf / 1.5, 0.0, 1.0).astype(np.float32)
    bevel_width = max(3.5, card_w * 0.0085)
    bevel_mask = np.clip(1.0 - np.abs(sdf + bevel_width * 0.55) / (bevel_width * 0.65), 0.0, 1.0) * alpha_mask

    # 4. Prepare Layer 1: Background image (custom or template)
    if bg_path and os.path.isfile(bg_path):
        pil_bg = ImageOps.exif_transpose(Image.open(bg_path)).convert("RGB")
        bg_src = np.asarray(pil_bg, dtype=np.float32) / 255.0
        bg_u8 = cv2.resize(np.clip(bg_src * 255, 0, 255).astype(np.uint8), (card_w, card_h), interpolation=cv2.INTER_AREA)
        bg_rgb = bg_u8.astype(np.float32) / 255.0
        bg_source_desc = os.path.basename(bg_path)
    else:
        bg_rgb = generate_background_template(chosen_bg_tpl, w=card_w, h=card_h)
        bg_source_desc = f"template:{chosen_bg_tpl}"

    # Pre-blur deep background slightly for camera-space Depth of Field (focus on midground)
    bg_rgb_dof = cv2.GaussianBlur(bg_rgb, (0, 0), sigmaX=max(0.8, 1.15 * supersample)) if enable_dof else bg_rgb

    # Background dark/low-detail mask for Pokémon-grade zoned background holo
    bg_lum = (0.299 * bg_rgb[..., 0] + 0.587 * bg_rgb[..., 1] + 0.114 * bg_rgb[..., 2]).astype(np.float32)
    bg_gx = cv2.Sobel(bg_lum, cv2.CV_32F, 1, 0, ksize=3)
    bg_gy = cv2.Sobel(bg_lum, cv2.CV_32F, 0, 1, ksize=3)
    bg_edge = np.sqrt(bg_gx * bg_gx + bg_gy * bg_gy)
    bg_dark_mask = np.clip((0.55 - bg_lum) / 0.40, 0.0, 1.0) * np.clip(1.0 - bg_edge * 1.4, 0.25, 1.0)
    bg_dark_mask = cv2.GaussianBlur(bg_dark_mask, (0, 0), sigmaX=max(3.0, card_w * 0.008))

    # 5. Prepare Layer 2: Midground Subject + Depth Map (ECC aligned or auto-synthesized)
    subj_u8 = cv2.resize(np.clip(subj_rgb_raw * 255, 0, 255).astype(np.uint8), (inner_w, inner_h), interpolation=cv2.INTER_AREA)
    subj_inner_rgb = subj_u8.astype(np.float32) / 255.0
    subj_inner_a = np.clip(cv2.resize(subj_alpha_raw, (inner_w, inner_h), interpolation=cv2.INTER_AREA), 0.0, 1.0).astype(np.float32)

    subj_rgb = np.zeros((card_h, card_w, 3), dtype=np.float32)
    subj_alpha = np.zeros((card_h, card_w), dtype=np.float32)
    subj_rgb[pad:pad + inner_h, pad:pad + inner_w] = subj_inner_rgb
    subj_alpha[pad:pad + inner_h, pad:pad + inner_w] = subj_inner_a

    # Edge-color-extended subject RGB prevents dark bleed on narrow 3D relief sidewalls
    subj_rgb_ext = prepare_edge_extended_subject(subj_rgb, subj_alpha, pad_px=max(5, card_w // 110))

    # Monocular depth map: ECC-aligned custom depth OR auto-generated from 2D subject
    if depth_path and os.path.isfile(depth_path):
        pil_depth = ImageOps.exif_transpose(Image.open(depth_path)).convert("L")
        raw_d = np.asarray(pil_depth, dtype=np.float32) / 255.0
        inner_d = align_and_refine_depth_map(raw_d, subj_inner_a, use_ecc=True)
        depth_source_desc = f"ecc_aligned:{os.path.basename(depth_path)}"
    else:
        use_ai = False
        ai_variant = "auto"
        if depth_engine == "ai_large":
            use_ai = True
            ai_variant = "large"
        elif depth_engine == "ai":
            use_ai = True
            ai_variant = "small"
        elif depth_engine == "auto":
            use_ai, _ = is_ai_depth_available()
            ai_variant = "auto"

        if use_ai:
            try:
                raw_ai = estimate_depth_ai(subj_inner_rgb, subj_inner_a, model_variant=ai_variant)
                inner_d = align_and_refine_depth_map(raw_ai, subj_inner_a, use_ecc=False)
                depth_source_desc = f"ai_monocular_{ai_variant}"
            except Exception:
                inner_d = synthesize_auto_depth_map(subj_inner_rgb, subj_inner_a)
                inner_d = align_and_refine_depth_map(inner_d, subj_inner_a, use_ecc=False)
                depth_source_desc = "auto_fast_fallback"
        else:
            inner_d = estimate_depth_fast(subj_inner_rgb, subj_inner_a)
            inner_d = align_and_refine_depth_map(inner_d, subj_inner_a, use_ecc=False)
            depth_source_desc = "auto_fast"

    depth_map = np.zeros((card_h, card_w), dtype=np.float32)
    depth_map[pad:pad + inner_h, pad:pad + inner_w] = inner_d

    # Smooth clamped surface normals from depth map (resolution-independent physical 3D slope!)
    # Height increases toward camera (-Z), so outward XY normal is (-dH/dx, -dH/dy)
    d_for_norm = cv2.GaussianBlur(depth_map, (0, 0), sigmaX=max(2.5, card_w * 0.006))
    dnx_raw = cv2.Sobel(d_for_norm, cv2.CV_32F, 1, 0, ksize=3)
    dny_raw = cv2.Sobel(d_for_norm, cv2.CV_32F, 0, 1, ksize=3)
    norm_scale = card_w * 0.055
    depth_nx = (-np.tanh(dnx_raw * norm_scale) * 0.62).astype(np.float32)
    depth_ny = (-np.tanh(dny_raw * norm_scale) * 0.62).astype(np.float32)

    # Extract SAR character linework + metallic/jewelry highlight zones (protecting face/eyes!)
    inner_lum = (0.299 * subj_inner_rgb[..., 0] + 0.587 * subj_inner_rgb[..., 1] + 0.114 * subj_inner_rgb[..., 2]).astype(np.float32)
    bh_k = max(5, int(round(inner_w * 0.011)) | 1)
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_k, bh_k))
    local_hi = cv2.morphologyEx(inner_lum, cv2.MORPH_CLOSE, bh_kernel)
    blackhat = np.clip((local_hi - inner_lum) * 3.2, 0.0, 1.0)
    inner_line = np.clip((blackhat - 0.14) / 0.55, 0.0, 1.0) ** 0.85

    inner_ycrcb = cv2.cvtColor(subj_u8, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    inner_skin = (
        np.exp(-0.5 * ((inner_ycrcb[..., 1] - 146.0) / 14.0) ** 2)
        * np.exp(-0.5 * ((inner_ycrcb[..., 2] - 112.0) / 14.0) ** 2)
    )
    iy, _ = np.mgrid[0:inner_h, 0:inner_w].astype(np.float32)
    skin_blur = cv2.GaussianBlur(inner_skin, (0, 0), sigmaX=max(6.0, inner_w * 0.025))
    face_atten = np.clip(skin_blur * 2.3 * np.clip(1.25 - iy / max(1.0, inner_h * 0.45), 0.0, 1.0), 0.0, 0.88)
    inner_line = (inner_line * (1.0 - face_atten * 0.80)).astype(np.float32)

    # Metallic/jewelry ornament zones on subject (sharp gradient + high local specular contrast, non-skin)
    sgx = cv2.Sobel(inner_lum, cv2.CV_32F, 1, 0, ksize=3)
    sgy = cv2.Sobel(inner_lum, cv2.CV_32F, 0, 1, ksize=3)
    sgrad = np.sqrt(sgx * sgx + sgy * sgy)
    inner_hsv = cv2.cvtColor(subj_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    is_gold_hue = ((inner_hsv[..., 0] >= 12) & (inner_hsv[..., 0] <= 34) & (inner_hsv[..., 1] > 45)).astype(np.float32)
    metal_inner = np.clip(
        (inner_line * 0.65 + np.clip(sgrad * 1.6, 0.0, 1.0) * 0.45 + is_gold_hue * 0.55)
        * (1.0 - face_atten * 0.92),
        0.0,
        1.0,
    )
    metal_inner = cv2.GaussianBlur(metal_inner, (0, 0), sigmaX=1.2)

    line_mask = np.zeros((card_h, card_w), dtype=np.float32)
    metal_mask = np.zeros((card_h, card_w), dtype=np.float32)
    face_protect_full = np.zeros((card_h, card_w), dtype=np.float32)
    line_mask[pad:pad + inner_h, pad:pad + inner_w] = inner_line
    metal_mask[pad:pad + inner_h, pad:pad + inner_w] = metal_inner
    face_protect_full[pad:pad + inner_h, pad:pad + inner_w] = face_atten
    line_mask *= subj_alpha
    metal_mask *= subj_alpha
    line_gx = cv2.Sobel(line_mask, cv2.CV_32F, 1, 0, ksize=3)
    line_gy = cv2.Sobel(line_mask, cv2.CV_32F, 0, 1, ksize=3)

    # 6. Prepare Layer 3: Foreground Transparent Decorative Frame
    has_fg = True
    if fg_path and os.path.isfile(fg_path):
        pil_fg = ImageOps.exif_transpose(Image.open(fg_path)).convert("RGBA")
        fg_arr = np.asarray(pil_fg, dtype=np.float32) / 255.0
        fg_rgb = cv2.resize(fg_arr[..., :3], (card_w, card_h), interpolation=cv2.INTER_AREA)
        fg_alpha = np.clip(cv2.resize(fg_arr[..., 3], (card_w, card_h), interpolation=cv2.INTER_AREA), 0.0, 1.0)
        fg_source_desc = os.path.basename(fg_path)
    elif chosen_fg_tpl == "none":
        fg_rgb = np.zeros((card_h, card_w, 3), dtype=np.float32)
        fg_alpha = np.zeros((card_h, card_w), dtype=np.float32)
        has_fg = False
        fg_source_desc = "none"
    else:
        fg_rgba = generate_foreground_frame_template(
            chosen_fg_tpl, w=card_w, h=card_h, metal_rgb=tuple(float(x) for x in theme.metal_base)
        )
        fg_rgb = fg_rgba[..., :3]
        fg_alpha = fg_rgba[..., 3]
        fg_source_desc = f"template:{chosen_fg_tpl}"

    # Premultiplied-alpha DoF for foreground outer frame (zero grey halo!)
    if has_fg and enable_dof:
        fg_rgb_dof, fg_alpha_dof = blur_premultiplied_rgba(fg_rgb, fg_alpha, sigma=max(0.5, 0.72 * supersample))
    else:
        fg_rgb_dof, fg_alpha_dof = fg_rgb, fg_alpha
    fg_gx = cv2.Sobel(fg_alpha, cv2.CV_32F, 1, 0, ksize=3)
    fg_gy = cv2.Sobel(fg_alpha, cv2.CV_32F, 0, 1, ksize=3)

    # 7. Shared Card-Local Holographic Etch, Brushed Sidewall & Quicksand Interlayer
    etch_tex, grating_tex = build_micro_etching_texture(card_h, card_w, theme.pattern_name)
    brushed_tex = build_brushed_metal_edge_texture(card_h, card_w)
    quicksand = init_quicksand_interlayer(card_h, card_w, theme, n_glitters=300, seed=777)

    base_yy, base_xx = np.mgrid[0:card_h, 0:card_w].astype(np.float32)
    u_norm = base_xx / max(1, card_w - 1)
    v_norm = base_yy / max(1, card_h - 1)

    # Multi-group anchored 4-point cross star glints:
    # foreground frame tips/corners + top/bottom crests + 2 subject metallic highlights (avoiding face!)
    m_in = max(20, int(round(card_w * 0.062)))
    anchors: List[Tuple[float, float, float]] = [
        (m_in * 0.85, m_in * 0.85, 1.10),
        (card_w - 1 - m_in * 0.85, m_in * 0.85, 1.05),
        (m_in * 0.85, card_h - 1 - m_in * 0.85, 1.00),
        (card_w - 1 - m_in * 0.85, card_h - 1 - m_in * 0.85, 1.10),
        (card_w * 0.50, m_in * 0.95, 0.92),
        (card_w * 0.50, card_h - 1 - m_in * 0.95, 0.92),
    ]
    # Add up to 2 crisp non-face subject metallic highlight anchors
    metal_score = cv2.GaussianBlur(metal_mask * (1.0 - face_protect_full), (0, 0), sigmaX=4.0)
    # Exclude border margin so they don't duplicate frame corners
    metal_score[:int(card_h * 0.22), :] = 0.0
    metal_score[-int(card_h * 0.12):, :] = 0.0
    metal_score[:, :int(card_w * 0.14)] = 0.0
    metal_score[:, -int(card_w * 0.14):] = 0.0
    for _ in range(2):
        idx = int(np.argmax(metal_score))
        my, mx = divmod(idx, card_w)
        if metal_score[my, mx] < 0.18:
            break
        anchors.append((float(mx), float(my), 0.78))
        cv2.circle(metal_score, (mx, my), int(min(card_w, card_h) * 0.20), 0.0, -1)

    # Studio background & foreground parallax particles
    bg_particles = init_particles(28, ss_canvas_w, ss_canvas_h, theme, foreground=False, seed=2026)
    fg_particles = init_particles(7, ss_canvas_w, ss_canvas_h, theme, foreground=True, seed=9091)

    card_edge_data = {
        "edge_nx": edge_nx,
        "edge_ny": edge_ny,
    }

    canvas_cx = ss_canvas_w * 0.5
    canvas_cy = ss_canvas_h * 0.5
    # Parallax depths (in pixels at working card resolution — tuned for unmistakable 3D pop!)
    bg_parallax_px = card_w * 0.095
    fg_parallax_px = card_w * 0.028
    relief_px = card_w * 0.135 * max(0.0, relief_strength)

    # Precompute atmospheric studio backdrop behind the floating 3D card (non-pure-black!)
    base_studio_bg: Optional[np.ndarray] = None
    if studio_bg_path and os.path.isfile(studio_bg_path):
        pil_st = ImageOps.exif_transpose(Image.open(studio_bg_path)).convert("RGB")
        st_arr = np.asarray(pil_st, dtype=np.float32) / 255.0
        base_studio_bg = cv2.resize(st_arr, (ss_canvas_w, ss_canvas_h), interpolation=cv2.INTER_AREA).astype(np.float32)
        studio_source_desc = os.path.basename(studio_bg_path)
    elif studio_bg == "obsidian_dark":
        studio_source_desc = "obsidian_dark"
    else:
        st_name = STYLE_TO_STUDIO_BG.get(theme.style_name, "starry_indigo") if studio_bg == "auto" else studio_bg
        base_studio_bg = generate_studio_backdrop(st_name, w=ss_canvas_w, h=ss_canvas_h)
        studio_source_desc = f"studio:{st_name}"

    frames_final: List[np.ndarray] = []
    frame_thetas = [float(single_frame_theta)] if single_frame_theta is not None else [2.0 * math.pi * idx / n_frames for idx in range(n_frames)]

    for i, theta in enumerate(frame_thetas):
        if cancel_event is not None and cancel_event.is_set():
            raise RenderCancelledError("渲染已取消")
        if single_frame_theta is None:
            _report(i + 1, total_steps, f"正在渲染第 {i + 1}/{n_frames} 帧 (三层景深 + 流麻 + Relief Mapping)...")

        rx, ry, rz, float_y = compute_sar_frame_pose(theta, card_h)
        R = rotation_matrix_xyz(rx, ry, rz)

        # Studio key light & view vector in world and card-local space
        light_dir = np.array([
            0.38 + 0.30 * math.sin(theta + 0.35),
            -0.50 + 0.20 * math.cos(theta),
            -0.78,
        ], dtype=np.float64)
        light_dir /= np.linalg.norm(light_dir)
        view_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        half_vec = light_dir + view_dir
        half_vec /= np.linalg.norm(half_vec)

        local_light = R.T @ light_dir
        local_view = R.T @ view_dir
        local_half = R.T @ half_vec
        lx, ly = float(local_light[0]), float(local_light[1])

        # Exact in-card parallax slope (0, 0 at exact 0° frontal frame!)
        px = float(local_view[0] / max(0.25, -local_view[2]))
        py = float(local_view[1] / max(0.25, -local_view[2]))

        # ===================================================================
        # IN-CARD 3-LAYER COMPOSITING (Sealed inside rounded card window!)
        # ===================================================================

        # --- Layer 1: Recessed Background (edge-extended, zero border gap) ---
        bg_map_x = (base_xx - px * bg_parallax_px).astype(np.float32)
        bg_map_y = (base_yy - py * bg_parallax_px).astype(np.float32)
        card_face = cv2.remap(bg_rgb_dof, bg_map_x, bg_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        bg_dark_w = cv2.remap(bg_dark_mask, bg_map_x, bg_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        etch_w = cv2.remap(etch_tex, bg_map_x, bg_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        grat_w = cv2.remap(grating_tex, bg_map_x, bg_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        # Shared two-layer iridescent foil waves
        diag1 = u_norm * 0.76 + v_norm * 0.44
        phase1 = (
            theme.irid_phase_shift
            + diag1 * 5.2
            + float(local_half[0]) * 6.8
            + float(local_half[1]) * 4.2
            + grat_w * 0.70
        )
        diag2 = u_norm * 0.38 - v_norm * 0.82
        phase2 = (
            theme.irid_phase_shift + 1.85
            + diag2 * 6.4
            - float(local_half[0]) * 5.4
            + float(local_half[1]) * 5.8
            + etch_w * 1.15
        )
        irid_mix = (spectral_iridescence(phase1) * 0.58 + spectral_iridescence(phase2) * 0.42) * 0.74 + theme.metal_base[None, None, :] * 0.26

        # Seamless periodic glass specular sweep & moving spotlight
        diag_coord = u_norm * 0.70 + v_norm * 0.30
        sweep_center = 0.50 + 0.64 * math.sin(theta - 0.35)
        sweep_band = np.exp(-0.5 * ((diag_coord - sweep_center) / 0.125) ** 2).astype(np.float32)

        spot_x = 0.50 + 0.30 * math.sin(theta + 0.15)
        spot_y = 0.46 - 0.22 * math.cos(theta + 0.25)
        spot = np.exp(-0.5 * (((u_norm - spot_x) / 0.36) ** 2 + ((v_norm - spot_y) / 0.38) ** 2)).astype(np.float32)

        # Apply Pokémon-grade zoned background holographics (mainly in dark/low-detail regions)
        angle_boost = 0.68 + 0.32 * abs(math.sin(theta + 0.25))
        bg_foil_w = bg_dark_w * (0.16 + 0.34 * etch_w + 0.16 * grat_w) * angle_boost * inten
        card_face = card_face + irid_mix * bg_foil_w[..., None] * (1.0 - card_face * 0.35)

        # Sweep-ignited background etching grooves
        bg_etch_glint = etch_w * bg_dark_w * (0.22 + 0.78 * np.clip(sweep_band * 1.25 + spot * 0.55, 0.0, 1.0)) * inten
        card_face = card_face + (irid_mix * 0.58 + theme.metal_bright[None, None, :] * 0.42) * (bg_etch_glint * 0.34)[..., None]

        # --- Unified Directional Cast Shadow of Subject onto Deep Background & 流麻 ---
        sh_dx = -lx * card_w * 0.036 - px * bg_parallax_px * 0.45
        sh_dy = -ly * card_h * 0.028 - py * bg_parallax_px * 0.45
        sh_map_x = (base_xx - sh_dx).astype(np.float32)
        sh_map_y = (base_yy - sh_dy).astype(np.float32)
        subj_shadow = cv2.remap(subj_alpha, sh_map_x, sh_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        subj_shadow = cv2.GaussianBlur(subj_shadow, (0, 0), sigmaX=max(8.0, card_w * 0.022))

        # Apply subject cast shadow onto deep background
        card_face *= (1.0 - subj_shadow[..., None] * 0.52)

        # --- Interlayer: Background Quicksand (流麻) + 300 3D Normal-Lit Glitters ---
        render_quicksand_layer(
            card_face, quicksand, theta, R, px, py,
            half_vec.astype(np.float32), light_dir.astype(np.float32),
            subj_shadow, theme, strength=quicksand_strength * inten,
        )

        # --- Inner 3D Acrylic Diorama Box Wall Refraction & Cavity Shadow ---
        # Gives unmistakable physical "looking into a deep crystal shadowbox" depth!
        cavity_sdf = cv2.remap(
            sdf,
            (base_xx - px * bg_parallax_px * 0.70).astype(np.float32),
            (base_yy - py * bg_parallax_px * 0.70).astype(np.float32),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        # Inner wall region visible inside the front glass edge when tilted
        cavity_wall = np.clip((cavity_sdf + 2.0) / max(4.0, card_w * 0.015), 0.0, 1.0) * alpha_mask
        cavity_wall_lit = np.clip(edge_nx * lx + edge_ny * ly, -1.0, 1.0)
        # Lit inner acrylic wall catches subtle iridescent crystal refraction; shadow wall deepens cavity
        wall_prism = (theme.metal_bright[None, None, :] * 0.55 + irid_mix * 0.45) * np.clip(cavity_wall_lit * 0.5 + 0.5, 0.0, 1.0)[..., None]
        card_face = card_face * (1.0 - cavity_wall[..., None] * 0.36) + wall_prism * (cavity_wall[..., None] * 0.24)
        # Soft interior box shadow cast by the raised outer rim onto the recessed background & 流麻
        rim_cavity_sh = np.clip(1.0 + cavity_sdf / max(10.0, card_w * 0.045), 0.0, 1.0) ** 1.8 * alpha_mask
        card_face *= (1.0 - rim_cavity_sh[..., None] * 0.32)
        card_face = np.clip(card_face, 0.0, 1.0)

        # --- Layer 2: Midground 3D Relief-Mapped Subject ---
        rm = relief_map_subject(
            depth_map, subj_rgb_ext, subj_alpha, depth_nx, depth_ny,
            line_mask, line_gx, line_gy, metal_mask,
            px, py, relief_px, base_xx, base_yy, n_steps=16,
        )
        s_rgb = rm["rgb"]
        s_a = rm["alpha"]
        s_nx = rm["nx"]
        s_ny = rm["ny"]
        s_nz = -np.sqrt(np.clip(1.0 - s_nx * s_nx - s_ny * s_ny, 0.25, 1.0))

        # Screen-space 3D Heightfield Self-Shadowing (foreground hands/head/sleeves cast soft cool shadow onto rear wings/tail!)
        self_sh = compute_heightfield_self_shadow(
            depth_map, rm["u_hit"], rm["v_hit"], rm["hit_h"],
            lx, ly, step_span_px=max(10.0, card_w * 0.038),
        ) * s_a

        # Sculptural Warm/Cool Hemispheric Half-Lambert 3D shading (keeps sidewall clean & >= 0.89!)
        ndotl_s = np.clip(s_nx * lx + s_ny * ly + s_nz * float(local_light[2]), -1.0, 1.0)
        half_lambert = 0.5 * ndotl_s + 0.5
        lit_factor = 0.88 + 0.20 * half_lambert - rm["sidewall"] * 0.04
        lit_factor = np.clip(lit_factor, 0.89, 1.08)
        # Subtle warm key / cool bounce tinting for rich cinema 3D volume (never muddy grey!)
        cool_bounce = np.array([0.93, 0.95, 1.02], dtype=np.float32)
        warm_key = np.array([1.02, 1.01, 0.98], dtype=np.float32)
        hemi_tint = cool_bounce[None, None, :] * (1.0 - half_lambert[..., None]) + warm_key[None, None, :] * half_lambert[..., None]
        s_shaded = s_rgb * lit_factor[..., None] * hemi_tint

        # Apply soft cool-tinted 3D self-shadowing (preserves crisp bright white dress while popping 3D overlap!)
        sh_tint = np.array([0.86, 0.88, 0.96], dtype=np.float32)
        s_shaded = s_shaded * (1.0 - self_sh[..., None] * 0.55) + (s_shaded * sh_tint[None, None, :]) * (self_sh[..., None] * 0.55)

        # Soft 3D curvature satin/pearl highlight + crisp metallic jewelry specular
        ndoth_s = np.clip(s_nx * float(local_half[0]) + s_ny * float(local_half[1]) + s_nz * float(local_half[2]), 0.0, 1.0)
        curv_spec = (ndoth_s ** 8.0) * np.clip(rm["hit_h"], 0.0, 1.0) * (1.0 - self_sh * 0.7) * 0.075 * inten
        metal_spec = (ndoth_s ** 18.0) * rm["metal_mask"] * (0.45 + 0.55 * sweep_band) * 0.55 * inten
        s_shaded = s_shaded + (theme.metal_bright[None, None, :] * 0.72 + irid_mix * 0.28) * (metal_spec + curv_spec)[..., None]

        # SAR Linework Metallic Foil Etching & 3D Directional Line Relief
        if line_etch_scale > 0.01:
            lm = rm["line_mask"]
            ldot = np.tanh((rm["line_gx"] * lx + rm["line_gy"] * ly) * 1.9)
            g_sh = np.clip(-ldot, 0.0, 1.0) * 0.26 * line_etch_scale * inten
            g_lit = np.clip(ldot, 0.0, 1.0) * 0.20 * line_etch_scale * inten
            s_shaded = s_shaded * (1.0 - g_sh[..., None] * np.clip(lm * 1.2, 0.0, 1.0)[..., None])

            sat_irid = np.clip((irid_mix - 0.48) * 1.85 + 0.50, 0.0, 1.0)
            foil_col = np.clip(theme.metal_base[None, None, :] * 0.42 + sat_irid * 0.58, 0.0, 1.0)
            foil_hot = np.clip(foil_col * 0.65 + theme.metal_bright[None, None, :] * 0.45, 0.0, 1.0)
            angle_rip = 0.58 + 0.42 * np.sin(phase1 * 1.15 - phase2 * 0.65)
            sw_hit = np.clip((sweep_band * 1.35 + spot * 0.35 - 0.22) / 0.78, 0.0, 1.0) ** 1.25
            lf_a = np.clip(lm * (0.14 + 0.78 * sw_hit * angle_rip) * line_etch_scale * inten, 0.0, 0.88)
            s_shaded = s_shaded * (1.0 - lf_a[..., None]) + foil_col * lf_a[..., None]
            s_shaded = s_shaded + foil_hot * (g_lit * (0.35 + 0.65 * sw_hit) + (sweep_band ** 2.6) * lm * 0.38 * line_etch_scale * inten)[..., None] * (0.45 + 0.55 * lm[..., None])

        # True 3D View-Dependent Fresnel Rim Light + Silhouette Pearlescence
        eroded_a = cv2.erode((s_a > 0.45).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float32)
        fresnel_edge = np.clip(s_a - cv2.GaussianBlur(eroded_a, (0, 0), sigmaX=1.6), 0.0, 1.0)
        ndotv_s = np.clip(s_nx * float(local_view[0]) + s_ny * float(local_view[1]) + s_nz * float(local_view[2]), 0.0, 1.0)
        fresnel_3d = np.clip(1.0 - ndotv_s, 0.0, 1.0) ** 2.2
        rim_w = np.clip(fresnel_edge * 0.55 + fresnel_3d * 0.36, 0.0, 0.80) * s_a * (0.55 + 0.45 * sweep_band) * inten
        s_shaded = s_shaded + (irid_mix * 0.52 + theme.metal_bright[None, None, :] * 0.48) * (rim_w * 0.30)[..., None]

        # Smooth highlight roll-off on subject so white dress/face/helmet never clips or loses linework
        s_shaded = np.where(s_shaded > 0.88, 0.88 + 0.12 * np.tanh((s_shaded - 0.88) / 0.20), s_shaded)
        s_shaded = np.clip(s_shaded, 0.0, 1.0)

        # Composite midground 3D subject cleanly over background + 流麻 interlayer
        card_face = card_face * (1.0 - s_a[..., None]) + s_shaded * s_a[..., None]

        # --- Layer 3: Raised Foreground Ornate Decorative Frame (Double-Layer Emboss + DoF) ---
        if has_fg:
            fg_map_x = (base_xx + px * fg_parallax_px).astype(np.float32)
            fg_map_y = (base_yy + py * fg_parallax_px).astype(np.float32)

            # Soft subtle local shadow cast by foreground frame onto subject (never dirties colors!)
            fg_sh_x = (fg_map_x - lx * card_w * 0.010).astype(np.float32)
            fg_sh_y = (fg_map_y - ly * card_h * 0.008).astype(np.float32)
            fg_shadow = cv2.remap(fg_alpha, fg_sh_x, fg_sh_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            fg_shadow = cv2.GaussianBlur(fg_shadow, (0, 0), sigmaX=max(3.0, card_w * 0.007))
            card_face *= (1.0 - fg_shadow[..., None] * 0.18)

            # Double-layer embossed metallic underside thickness
            th_off_x = px * fg_parallax_px * 0.35 + 1.6 * supersample * (0.4 + 0.6 * lx)
            th_off_y = py * fg_parallax_px * 0.35 + 1.6 * supersample * (0.4 + 0.6 * ly)
            fg_under_a = cv2.remap(
                fg_alpha, (fg_map_x - th_off_x).astype(np.float32), (fg_map_y - th_off_y).astype(np.float32),
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
            )
            under_col = theme.metal_shadow * 0.82
            card_face = card_face * (1.0 - fg_under_a[..., None] * 0.78) + under_col[None, None, :] * (fg_under_a[..., None] * 0.78)

            # Raised top foreground metallic layer (with Premultiplied Alpha DoF!)
            fg_w_rgb = cv2.remap(fg_rgb_dof, fg_map_x, fg_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0, 0.0))
            fg_w_a = cv2.remap(fg_alpha_dof, fg_map_x, fg_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            fg_w_gx = cv2.remap(fg_gx, fg_map_x, fg_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            fg_w_gy = cv2.remap(fg_gy, fg_map_x, fg_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

            # High-intensity angle-driven gilded/silver/violet-gold reflection on foreground ornament
            fg_emb = np.tanh((fg_w_gx * lx + fg_w_gy * ly) * 1.4) * 0.24
            fg_foil_shimmer = (0.35 + 0.65 * np.clip(sweep_band * 1.30 + spot * 0.55, 0.0, 1.0)) * inten
            fg_lit = (
                fg_w_rgb * (0.85 + 0.25 * angle_boost)
                + (theme.metal_bright[None, None, :] * 0.65 + irid_mix * 0.35) * (fg_foil_shimmer * 0.55)[..., None]
                + np.where(fg_emb[..., None] >= 0, theme.metal_bright[None, None, :], theme.metal_shadow[None, None, :]) * np.abs(fg_emb)[..., None]
            )
            fg_lit = np.clip(fg_lit, 0.0, 1.0)
            card_face = card_face * (1.0 - fg_w_a[..., None]) + fg_lit * fg_w_a[..., None]

        # --- Full-Card Diagonal Glass Specular Sweep & Moving Spotlight ---
        # Automatically attenuate white-on-white sweep over very bright subject fabrics/face so fine linework stays crisp
        s_lum = (0.299 * s_rgb[..., 0] + 0.587 * s_rgb[..., 1] + 0.114 * s_rgb[..., 2]).astype(np.float32)
        bright_cloth_protect = s_a * np.clip((s_lum - 0.68) / 0.30, 0.0, 0.86)
        glass_atten = np.clip((1.0 - face_protect_full * 0.76) * (1.0 - bright_cloth_protect), 0.08, 1.0)
        sweep_col = theme.metal_bright * 0.55 + np.array([1.0, 1.0, 1.0], dtype=np.float32) * 0.45
        card_face = card_face + sweep_col[None, None, :] * (sweep_band * glass_atten * 0.15 * inten)[..., None]
        card_face = card_face + (theme.metal_bright * 0.40 + 0.60)[None, None, :] * (spot * glass_atten * 0.055 * inten)[..., None]

        # --- Dynamic Directional Metallic Bevel Rim on Outer Card Shell ---
        rim_dot = edge_nx * lx + edge_ny * ly
        rim_spec = np.clip(rim_dot * 0.5 + 0.5, 0.0, 1.0) ** 2.6
        rim_col = theme.metal_shadow[None, None, :] * (1.0 - rim_spec[..., None]) + theme.metal_bright[None, None, :] * rim_spec[..., None]
        card_face = card_face * (1.0 - bevel_mask[..., None] * 0.90) + rim_col * (bevel_mask[..., None] * 0.90)

        # Global soft highlight roll-off
        card_face = np.where(card_face > 0.88, 0.88 + 0.12 * np.tanh((card_face - 0.88) / 0.22), card_face)
        card_face = np.clip(card_face, 0.0, 1.0).astype(np.float32)

        # ===================================================================
        # STUDIO SCENE: Background, Shadow, 8-Layer Thickness, 3D Warp, Glints
        # ===================================================================
        canvas = render_studio_background(
            ss_canvas_h, ss_canvas_w, theta, float_y, rx, ry, theme, base_studio_bg=base_studio_bg
        )
        render_particles(canvas, bg_particles, theta)

        # Floating drop shadow (moves opposite tilt, scales/blurs with float height)
        float_norm = 0.5 - 0.5 * math.sin(theta + 0.45)
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
        canvas *= (1.0 - shadow_soft[..., None] * (0.58 - 0.14 * float_norm))

        # 8-layer brushed metallic thickness slices (back-to-front)
        for k in range(thickness_layers, 0, -1):
            z_loc = thickness_total * (k / float(thickness_layers))
            slice_t = 1.0 - (k / float(thickness_layers))
            edge_rgb = shade_metallic_edge_slice(R, theta, card_edge_data, brushed_tex, theme, slice_t)
            H_slice = plane_homography_for_z(card_w, card_h, R, z_loc, float_y, focal, canvas_cx, canvas_cy)
            w_rgb = cv2.warpPerspective(edge_rgb, H_slice, (ss_canvas_w, ss_canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0, 0.0))
            w_a = cv2.warpPerspective(alpha_mask, H_slice, (ss_canvas_w, ss_canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)[..., None]
            canvas = canvas * (1.0 - w_a) + w_rgb * w_a

        # Front 3-layer composited card window
        H_front = plane_homography_for_z(card_w, card_h, R, 0.0, float_y, focal, canvas_cx, canvas_cy)
        w_face = cv2.warpPerspective(card_face, H_front, (ss_canvas_w, ss_canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0, 0.0))
        w_face_a = cv2.warpPerspective(alpha_mask, H_front, (ss_canvas_w, ss_canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)[..., None]
        canvas = canvas * (1.0 - w_face_a) + w_face * w_face_a

        # Anchored 4-point cross star glints synced to diagonal sweep
        for ax, ay, size_scale in anchors:
            u_a = ax / max(1, card_w - 1)
            v_a = ay / max(1, card_h - 1)
            diag_a = u_a * 0.70 + v_a * 0.30
            peak = math.exp(-0.5 * ((diag_a - sweep_center) / 0.13) ** 2)
            glint_int = (0.06 + 0.90 * (peak ** 1.5)) * inten
            pt = H_front @ np.array([ax, ay, 1.0], dtype=np.float64)
            gx, gy = float(pt[0] / pt[2]), float(pt[1] / pt[2])
            glint_rad = (15.0 + 20.0 * peak) * size_scale * (ss_canvas_w / 920.0)
            draw_star_glint(canvas, gx, gy, glint_rad, glint_int, theme.metal_bright)

        # Foreground parallax soft-focus particles
        render_particles(canvas, fg_particles, theta)

        # Downsample cleanly with Lanczos
        ss_u8 = np.clip(canvas * 255.0 + 0.5, 0, 255).astype(np.uint8)
        pil_final = Image.fromarray(ss_u8, mode="RGB").resize((final_width, final_h), Image.Resampling.LANCZOS)
        if single_frame_theta is not None:
            return {
                "preview_image": pil_final,
                "style": theme.style_name,
                "pattern": theme.pattern_name,
                "metal": theme.metal_name,
                "bg_source": bg_source_desc,
                "fg_source": fg_source_desc,
                "studio_source": studio_source_desc,
                "depth_source": depth_source_desc,
                "output_size": (final_width, final_h),
            }
        frames_final.append(np.asarray(pil_final, dtype=np.uint8))

    if cancel_event is not None and cancel_event.is_set():
        raise RenderCancelledError("渲染已取消")

    _report(n_frames + 1, total_steps, "正在构建共享 256 色 Median Cut 调色板并导出 GIF...")
    adj_diffs = [
        float(np.mean(np.abs(frames_final[i + 1].astype(np.float32) - frames_final[i].astype(np.float32))))
        for i in range(n_frames - 1)
    ]
    seam_diff = float(np.mean(np.abs(frames_final[0].astype(np.float32) - frames_final[-1].astype(np.float32))))

    durations = build_even_durations(n_frames=n_frames, n_50ms=26)
    export_shared_palette_gif(frames_final, output_path, durations)
    _report(total_steps, total_steps, "三层 3D SAR 流麻闪卡渲染完成！")

    total_ms = sum(durations)
    return {
        "output_path": os.path.abspath(output_path),
        "mode": "3layer_sar",
        "style": theme.style_name,
        "pattern": theme.pattern_name,
        "metal": theme.metal_name,
        "intensity": theme.intensity_name,
        "bg_source": bg_source_desc,
        "fg_source": fg_source_desc,
        "studio_source": studio_source_desc,
        "depth_source": depth_source_desc,
        "bg_removed": bg_removed,
        "line_etch": line_etch,
        "quicksand_strength": quicksand_strength,
        "relief_strength": relief_strength,
        "output_size": (final_width, final_h),
        "frames": n_frames,
        "total_duration_s": round(total_ms / 1000.0, 2),
        "avg_fps": round(n_frames / (total_ms / 1000.0), 2),
        "avg_adj_diff": round(float(np.mean(adj_diffs)), 3),
        "seam_diff_47_to_0": round(seam_diff, 3),
        "file_size_kb": round(os.path.getsize(output_path) / 1024.0, 1),
    }


def render_live_preview_frame(
    subject_path: Optional[str],
    angle_preset: str = "right",
    preview_width: int = 320,
    **kwargs,
) -> Dict[str, object]:
    """Fast single-frame WYSIWYG 3D preview renderer for GUI interactive previews."""
    theta_map = {
        "front": 0.0,
        "right": math.pi * 0.5,      # +17.5 deg Y-axis tilt
        "left": math.pi * 1.5,       # -17.5 deg Y-axis tilt
        "diag": math.pi * 0.25,      # +12.4 deg Y & +10.5 deg X
        "tilt": math.pi * 0.25,      # alias used by Tauri UI
    }
    theta = theta_map.get(angle_preset, math.pi * 0.5)
    # Fast preview uses supersample=1, 4 thickness layers, and fast depth unless custom depth_path is already set
    kw = dict(kwargs)
    kw.setdefault("depth_engine", "fast")
    if not kw.get("depth_path"):
        kw["depth_engine"] = "fast"
    kw["final_width"] = preview_width
    kw["supersample"] = 1
    kw["thickness_layers"] = 4
    kw["n_frames"] = 16
    kw["single_frame_theta"] = theta
    return render_3layer_sar_gif(subject_path=subject_path, **kw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render 3-Layer Depth-Driven SAR + Quicksand (流麻) 3D GIF.")
    parser.add_argument("--subject", "--input", dest="subject", required=True, help="Midground subject image path")
    parser.add_argument("--bg", default=None, help="Optional custom background image path")
    parser.add_argument("--fg", default=None, help="Optional custom transparent foreground frame PNG path")
    parser.add_argument("--depth", default=None, help="Optional custom monocular grayscale depth map path")
    parser.add_argument("--studio-bg-image", default=None, help="Optional custom outer studio backdrop image path")
    parser.add_argument("--output", default="holographic_card.gif", help="Output GIF path")
    parser.add_argument("--bg-template", default="auto", choices=["auto"] + [k for k, _ in BG_TEMPLATE_META])
    parser.add_argument("--fg-template", default="auto", choices=["auto", "none"] + [k for k, _ in FG_TEMPLATE_META])
    parser.add_argument("--studio-bg", default="auto", choices=["auto", "obsidian_dark"] + [k for k, _ in STUDIO_BG_META])
    parser.add_argument("--style", default="auto", choices=[k for k, *_ in STYLE_META])
    parser.add_argument("--pattern", default="auto", choices=[k for k, _ in PATTERN_META])
    parser.add_argument("--metal", default="auto", choices=[k for k, _ in METAL_META])
    parser.add_argument("--intensity", default="medium", choices=["subtle", "medium", "lavish"])
    parser.add_argument("--bg-mode", default="auto", choices=["auto", "remove", "keep"])
    parser.add_argument("--line-etch", default="medium", choices=["off", "subtle", "medium", "strong"])
    parser.add_argument("--depth-engine", default="auto", choices=["fast", "ai", "ai_large", "auto"],
                        help="2D->Depth generator when --depth is omitted: auto | ai_large | ai | fast")
    parser.add_argument("--quicksand", type=float, default=1.0, help="Quicksand (流麻) intensity (0.0 to 1.6)")
    parser.add_argument("--relief", type=float, default=1.0, help="3D Relief Mapping parallax strength (0.0 to 1.6)")
    parser.add_argument("--no-dof", action="store_true", help="Disable camera-space Depth of Field")
    parser.add_argument("--width", type=int, default=460, help="Final GIF width in pixels")
    parser.add_argument("--frames", type=int, default=48, help="Number of loop frames")
    parser.add_argument("--supersample", type=int, default=2, help="Supersampling factor (1, 2, or 3)")
    args = parser.parse_args()

    info = render_3layer_sar_gif(
        subject_path=args.subject,
        output_path=args.output,
        bg_path=args.bg,
        fg_path=args.fg,
        depth_path=args.depth,
        bg_template=args.bg_template,
        fg_template=args.fg_template,
        studio_bg=args.studio_bg,
        studio_bg_path=args.studio_bg_image,
        style=args.style,
        pattern=args.pattern,
        metal=args.metal,
        intensity=args.intensity,
        bg_mode=args.bg_mode,
        line_etch=args.line_etch,
        depth_engine=args.depth_engine,
        quicksand_strength=args.quicksand,
        relief_strength=args.relief,
        enable_dof=not args.no_dof,
        final_width=args.width,
        n_frames=args.frames,
        supersample=args.supersample,
    )
    print("3-Layer SAR Render Complete:")
    for k, v in info.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
