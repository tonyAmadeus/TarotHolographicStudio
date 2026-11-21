#!/usr/bin/env python3
"""
10-Style Collector Tarot / SAR / Art Nouveau / Guofeng / Cyber Card Templates
=============================================================================
Procedurally generates 10 high-resolution background templates (RGB) and
10 matching transparent foreground ornate frame templates (RGBA) — plus a
minimalist astral hairline frame — into `./templates/`.

Inspired by collector tarot decks, Alphonse Mucha Art Nouveau arches, Pokémon
SAR/UR prismatic foil borders, Chinese Guofeng cloud-fretwork cards, Cyberpunk
HUD frames, and Japanese Sakura twilight quicksand acrylic cards.

Also provides ECC Affine Alignment + edge depth extrapolation + Bilateral
filtering for monocular depth maps.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from depth_estimator import _guided_filter_gray, estimate_depth_fast


# ---------------------------------------------------------------------------
# Template Registry (10 Cohesive Pairs + Minimalist Bonus Frame)
# ---------------------------------------------------------------------------

BG_TEMPLATE_META: List[Tuple[str, str]] = [
    ("classical_arcana", "古典神秘 · 秘仪星盘 (Classical Arcana)"),
    ("moon_astrolabe", "月夜星象 · 子夜星盘 (Moon Astrolabe)"),
    ("sun_sanctuary", "太阳圣辉 · 圣光玫瑰窗 (Sun Sanctuary)"),
    ("gothic_eclipse", "暗黑哥特 · 血金蚀月 (Gothic Eclipse)"),
    ("dream_mirage", "梦境水光 · 极光冰湖 (Dream Mirage)"),
    ("art_nouveau_halo", "新艺术穆夏 · 圣环彩窗 (Art Nouveau Halo)"),
    ("prism_shard", "宝可梦 SAR · 碎钻棱镜 (Prismatic Shards)"),
    ("guofeng_jade", "东方国风 · 青绿仙云 (Guofeng Jade Cloud)"),
    ("cyber_holo_grid", "赛博全息 · 霓虹矩阵 (Cyber Holo Grid)"),
    ("sakura_twilight", "绯夜樱吹 · 暮月流樱 (Sakura Twilight)"),
]

FG_TEMPLATE_META: List[Tuple[str, str]] = [
    ("classical_filigree", "古典神秘 · 鎏金秘仪花框 (Classical Filigree)"),
    ("moon_arch", "月夜星象 · 月华冰银拱框 (Moonlit Arch)"),
    ("sun_baroque", "太阳圣辉 · 巴洛克日冕框 (Solar Baroque)"),
    ("gothic_cathedral", "暗黑哥特 · 尖拱花窗框 (Gothic Cathedral)"),
    ("dream_crystal", "梦境水光 · 水波晶棱框 (Dream Crystal)"),
    ("art_nouveau_arch", "新艺术穆夏 · 穹顶花拱框 (Mucha Arch Frame)"),
    ("sar_prismatic_border", "宝可梦 SAR · 碎钻切角框 (SAR Prismatic)"),
    ("guofeng_cloud_frame", "东方国风 · 回纹祥云窗棂 (Guofeng Fretwork)"),
    ("cyber_hud_frame", "赛博全息 · 机能 HUD 装甲框 (Cyber HUD)"),
    ("sakura_lace_frame", "绯夜樱吹 · 和风樱花蕾丝框 (Sakura Lace)"),
    ("minimal_astral", "通用百搭 · 极简星轨细框 (Minimal Astral)"),
]

STUDIO_BG_META: List[Tuple[str, str]] = [
    ("starry_indigo", "星夜靛蓝丝绒棚"),
    ("royal_velvet", "古典酒红鎏金棚"),
    ("emerald_mist", "青绿云雾玉殿"),
    ("sakura_dusk", "绯樱暮紫柔焦棚"),
    ("aurora_teal", "极光冰青梦境"),
    ("champagne_gallery", "香槟暖灰美术馆"),
    ("cyber_studio", "赛博双色霓虹棚"),
]

STYLE_TO_TEMPLATES: Dict[str, Tuple[str, str]] = {
    "classical": ("classical_arcana", "classical_filigree"),
    "moon": ("moon_astrolabe", "moon_arch"),
    "sun": ("sun_sanctuary", "sun_baroque"),
    "gothic": ("gothic_eclipse", "gothic_cathedral"),
    "dream": ("dream_mirage", "dream_crystal"),
    "guofeng": ("guofeng_jade", "guofeng_cloud_frame"),
    "sakura": ("sakura_twilight", "sakura_lace_frame"),
    "cyber": ("cyber_holo_grid", "cyber_hud_frame"),
    "prism": ("prism_shard", "sar_prismatic_border"),
    "nouveau": ("art_nouveau_halo", "art_nouveau_arch"),
    "abyss": ("moon_astrolabe", "dream_crystal"),
}

STYLE_TO_STUDIO_BG: Dict[str, str] = {
    "classical": "champagne_gallery",
    "moon": "starry_indigo",
    "sun": "royal_velvet",
    "gothic": "sakura_dusk",
    "dream": "aurora_teal",
    "guofeng": "emerald_mist",
    "sakura": "sakura_dusk",
    "cyber": "cyber_studio",
    "prism": "aurora_teal",
    "nouveau": "champagne_gallery",
    "abyss": "starry_indigo",
}

_BG_ALIASES = {
    "astrolabe_night": "moon_astrolabe",
    "solar_sanctuary": "sun_sanctuary",
    "astral_mirage": "dream_mirage",
    "gothic_nebula": "gothic_eclipse",
}
_FG_ALIASES = {
    "celestial_gold": "classical_filigree",
    "moonlit_silver": "moon_arch",
    "baroque_solar": "sun_baroque",
}


def _default_template_dir() -> str:
    import sys

    env_root = os.environ.get("TAROT_APP_ROOT", "").strip()
    if env_root:
        return os.path.join(env_root, "templates")
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        parent_tpl = os.path.join(os.path.dirname(exe_dir), "templates")
        if os.path.isdir(parent_tpl):
            return parent_tpl
        return os.path.join(exe_dir, "templates")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


# ---------------------------------------------------------------------------
# Procedural Texture Helpers
# ---------------------------------------------------------------------------

def _fractal_cloud(h: int, w: int, seed: int) -> np.ndarray:
    """Fast multi-octave smooth procedural velvet/nebula cloud in [0, 1]."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), dtype=np.float32)
    weight_sum = 0.0
    for oct_w, oct_h, amp in [(6, 10, 0.50), (14, 22, 0.30), (30, 48, 0.14), (64, 96, 0.06)]:
        grid = rng.uniform(0.0, 1.0, size=(oct_h, oct_w)).astype(np.float32)
        up = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        acc += up * amp
        weight_sum += amp
    acc /= max(1e-6, weight_sum)
    acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=max(3.0, w * 0.015))
    lo, hi = float(acc.min()), float(acc.max())
    return np.clip((acc - lo) / max(1e-5, hi - lo), 0.0, 1.0).astype(np.float32)


def _voronoi_facets(h: int, w: int, n_sites: int = 44, seed: int = 515) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast vectorized Voronoi shattered-prism facet map (for Pokémon SAR / Terastal style).
    Computes on a downscaled grid then upsamples crisp facet boundaries + flat shard IDs.
    Returns (shard_shade, edge_highlight, shard_hue_offset) in [0, 1].
    """
    sw, sh = 180, max(180, int(round(180 * h / float(w))))
    rng = np.random.default_rng(seed)
    sx = rng.uniform(0.0, 1.0, size=n_sites).astype(np.float32)
    sy = rng.uniform(0.0, 1.0, size=n_sites).astype(np.float32)
    s_shade = rng.uniform(0.15, 1.0, size=n_sites).astype(np.float32)
    s_hue = rng.uniform(0.0, 1.0, size=n_sites).astype(np.float32)

    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    u = xx / float(sw)
    v = yy / float(sh)
    aspect = sh / float(sw)

    # Distance to all sites: (sh, sw, n_sites)
    dx = u[..., None] - sx[None, None, :]
    dy = (v[..., None] - sy[None, None, :]) * aspect
    d2 = dx * dx + dy * dy
    order = np.argpartition(d2, 2, axis=-1)
    idx0 = order[..., 0]
    idx1 = order[..., 1]
    d0 = np.sqrt(np.take_along_axis(d2, idx0[..., None], axis=-1)[..., 0])
    d1 = np.sqrt(np.take_along_axis(d2, idx1[..., None], axis=-1)[..., 0])

    # Crisp shard boundary where d1 - d0 is small
    edge = np.exp(-(((d1 - d0) / 0.014) ** 2)).astype(np.float32)
    shade = s_shade[idx0]
    hue = s_hue[idx0]

    shade_up = cv2.resize(shade, (w, h), interpolation=cv2.INTER_NEAREST)
    shade_up = cv2.GaussianBlur(shade_up, (3, 3), sigmaX=0.8)
    edge_up = cv2.resize(edge, (w, h), interpolation=cv2.INTER_CUBIC)
    hue_up = cv2.resize(hue, (w, h), interpolation=cv2.INTER_NEAREST)
    return shade_up.astype(np.float32), np.clip(edge_up, 0.0, 1.0).astype(np.float32), hue_up.astype(np.float32)


# ---------------------------------------------------------------------------
# 10 Collector Background Templates (RGB)
# ---------------------------------------------------------------------------

def generate_background_template(name: str, w: int = 720, h: int = 1152) -> np.ndarray:
    """Generate one of the 10 collector tarot / SAR / community backgrounds (RGB float32)."""
    name = _BG_ALIASES.get(name, name)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx / max(1, w - 1)) * 2.0 - 1.0
    v = (yy / max(1, h - 1)) * 2.0 - 1.0
    aspect = h / float(w)
    v_iso = v * aspect

    cy = -0.18 * aspect
    dx = u
    dy = v_iso - cy
    r = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)

    if name == "classical_arcana":
        c_deep = np.array([0.044, 0.038, 0.058], dtype=np.float32)
        c_mid = np.array([0.118, 0.095, 0.132], dtype=np.float32)
        c_gold = np.array([0.84, 0.70, 0.44], dtype=np.float32)
        c_indigo = np.array([0.42, 0.44, 0.68], dtype=np.float32)
        cloud = _fractal_cloud(h, w, seed=101)
        bg = c_deep + c_mid * (cloud[..., None] * 0.72)
        bg += c_gold * (np.exp(-(r ** 2) / 0.56) * 0.28)[..., None]
        bg += c_indigo * (cloud * np.exp(-((r - 0.78) ** 2) / 0.32) * 0.22)[..., None]
        ring1 = np.exp(-((r - 0.56) ** 2) / 0.00014)
        ring2 = np.exp(-((r - 0.72) ** 2) / 0.00012)
        oct_r = 0.64 + 0.045 * np.cos(ang * 8.0)
        oct_line = np.exp(-((r - oct_r) ** 2) / 0.00015)
        diag_lat = np.maximum(
            np.exp(-((np.sin((u + v_iso) * 10.5) ** 2) / 0.014)),
            np.exp(-((np.sin((u - v_iso) * 10.5) ** 2) / 0.014)),
        ) * 0.24 * np.exp(-r * 0.55)
        axis_cross = np.maximum(
            np.exp(-(u * u) / 0.000032),
            np.exp(-((v_iso - cy) ** 2) / 0.000032),
        ) * np.exp(-r * 0.9) * 0.42
        orn = np.clip(ring1 * 0.85 + ring2 * 0.70 + oct_line * 0.72 + diag_lat + axis_cross, 0.0, 1.0)
        bg += c_gold * (orn * 0.36)[..., None]

    elif name == "sun_sanctuary":
        c_deep = np.array([0.058, 0.030, 0.036], dtype=np.float32)
        c_mid = np.array([0.172, 0.098, 0.062], dtype=np.float32)
        c_glow = np.array([0.90, 0.66, 0.30], dtype=np.float32)
        c_sec = np.array([0.58, 0.22, 0.26], dtype=np.float32)
        cloud = _fractal_cloud(h, w, seed=201)
        rays = ((0.5 + 0.5 * np.cos(ang * 16.0)) ** 3.0) * np.exp(-r * 1.10)
        fine_rays = ((0.5 + 0.5 * np.cos(ang * 32.0)) ** 6.0) * np.exp(-r * 1.45) * 0.45
        bg = c_deep * (1.0 - cloud[..., None] * 0.55) + c_mid * (cloud[..., None] * 0.78)
        bg += c_glow * (np.exp(-(r ** 2) / 0.54) * 0.36 + (rays + fine_rays) * 0.24)[..., None]
        bg += c_sec * (np.exp(-((r - 0.85) ** 2) / 0.26) * cloud * 0.26)[..., None]
        ring1 = np.exp(-((r - 0.60) ** 2) / 0.00018)
        ring2 = np.exp(-((r - 0.75) ** 2) / 0.00014)
        petals = np.exp(-((r - (0.675 + 0.058 * np.cos(ang * 12.0))) ** 2) / 0.00016)
        orn = np.clip(ring1 * 0.85 + ring2 * 0.68 + petals * 0.80, 0.0, 1.0)
        bg += c_glow * (orn * 0.38)[..., None]

    elif name == "dream_mirage":
        c_deep = np.array([0.022, 0.048, 0.092], dtype=np.float32)
        c_mid = np.array([0.062, 0.152, 0.215], dtype=np.float32)
        c_glow = np.array([0.42, 0.82, 0.88], dtype=np.float32)
        c_sec = np.array([0.58, 0.44, 0.86], dtype=np.float32)
        cloud = _fractal_cloud(h, w, seed=303)
        aurora = (0.5 + 0.5 * np.sin(u * 4.5 + np.sin(v_iso * 2.6) * 1.7 + cloud * 3.0)) ** 2.4
        silk = (0.5 + 0.5 * np.sin(v_iso * 6.5 - np.cos(u * 3.2) * 1.4)) ** 3.2 * 0.32
        bg = c_deep * (1.0 - cloud[..., None] * 0.48) + c_mid * (cloud[..., None] * 0.72)
        bg += c_glow * (aurora * np.exp(-(r ** 2) / 1.45) * 0.30 + np.exp(-(r ** 2) / 0.64) * 0.24)[..., None]
        bg += c_sec * ((cloud * np.exp(-((r - 0.72) ** 2) / 0.32) + silk) * 0.28)[..., None]
        ring1 = np.exp(-((r - 0.68) ** 2) / 0.00015)
        ripples = np.exp(-((np.sin(r * 10.5) ** 2) / 0.016)) * np.exp(-r * 0.82) * 0.42
        bg += (c_glow * 0.68 + 0.32) * (np.clip(ring1 * 0.75 + ripples, 0.0, 1.0) * 0.32)[..., None]

    elif name == "gothic_eclipse":
        c_deep = np.array([0.038, 0.018, 0.048], dtype=np.float32)
        c_mid = np.array([0.145, 0.045, 0.125], dtype=np.float32)
        c_crimson = np.array([0.76, 0.26, 0.44], dtype=np.float32)
        c_gold = np.array([0.85, 0.64, 0.36], dtype=np.float32)
        cloud = _fractal_cloud(h, w, seed=409)
        cloud2 = _fractal_cloud(h, w, seed=419)
        bg = c_deep + c_mid * (cloud[..., None] * 0.85)
        bg += c_crimson * (np.exp(-((r - 0.56) ** 2) / 0.045) * 0.36 + (cloud * cloud2) * 0.30)[..., None]
        core_darken = 1.0 - np.exp(-(r ** 2) / 0.16) * 0.38
        bg *= core_darken[..., None]
        eclipse = np.exp(-((r - 0.56) ** 2) / 0.00022)
        ring_out = np.exp(-((r - 0.78) ** 2) / 0.00014)
        spikes = ((0.5 + 0.5 * np.cos(ang * 8.0)) ** 26.0) * np.exp(-((r - 0.70) ** 2) / 0.025)
        bg += c_gold * (np.clip(eclipse * 0.95 + ring_out * 0.62 + spikes * 0.82, 0.0, 1.0) * 0.38)[..., None]

    elif name == "art_nouveau_halo":
        # 新艺术穆夏风格：深祖母绿与古铜暖金过渡 + 经典穆夏头部圣环 + 彩绘玻璃放射扇格
        cy_m = -0.34 * aspect
        r_m = np.sqrt(u * u + (v_iso - cy_m) ** 2)
        ang_m = np.arctan2(v_iso - cy_m, u)
        c_deep = np.array([0.036, 0.072, 0.068], dtype=np.float32)
        c_teal = np.array([0.088, 0.185, 0.165], dtype=np.float32)
        c_gold = np.array([0.88, 0.74, 0.44], dtype=np.float32)
        c_peach = np.array([0.78, 0.54, 0.38], dtype=np.float32)
        cloud = _fractal_cloud(h, w, seed=521)
        bg = c_deep + c_teal * (cloud[..., None] * 0.70)
        # Signature Mucha luminous medallion halo behind character head
        halo_fill = np.clip(1.0 - (r_m / 0.68) ** 4.0, 0.0, 1.0)
        bg = bg * (1.0 - halo_fill[..., None] * 0.35) + (c_peach * 0.32 + c_gold * 0.22)[None, None, :] * halo_fill[..., None]
        # Stained-glass fan petals & dual ornamental Mucha halo rings
        m_ring1 = np.exp(-((r_m - 0.66) ** 2) / 0.00022)
        m_ring2 = np.exp(-((r_m - 0.75) ** 2) / 0.00018)
        m_ring3 = np.exp(-((r_m - 0.56) ** 2) / 0.00012)
        mosaic_band = ((0.5 + 0.5 * np.cos(ang_m * 24.0)) ** 14.0) * np.exp(-((r_m - 0.705) ** 2) / 0.0014)
        # Subtle vertical Art-Nouveau fluted columns near left/right sides
        cols = np.exp(-((np.abs(u) - 0.78) ** 2) / 0.00035) * np.clip(1.0 - np.abs(v) * 0.4, 0.0, 1.0) * 0.55
        orn = np.clip(m_ring1 * 0.92 + m_ring2 * 0.85 + m_ring3 * 0.55 + mosaic_band * 0.78 + cols, 0.0, 1.0)
        bg += c_gold * (orn * 0.44)[..., None]

    elif name == "prism_shard":
        # 宝可梦 SAR / Terastal 碎钻棱镜：深曜石蓝紫 + Voronoi 晶体碎钻折射面 + 菱形星辉爆发
        shade, edge, hue_f = _voronoi_facets(h, w, n_sites=48, seed=617)
        c_base = np.array([0.032, 0.038, 0.088], dtype=np.float32)
        # Iridescent per-facet tinting (sapphire / amethyst / cyan / gold)
        f_r = 0.25 + 0.55 * (0.5 + 0.5 * np.cos(hue_f * 6.283 + 0.2))
        f_g = 0.28 + 0.52 * (0.5 + 0.5 * np.cos(hue_f * 6.283 + 2.2))
        f_b = 0.45 + 0.50 * (0.5 + 0.5 * np.cos(hue_f * 6.283 + 4.1))
        facet_col = np.stack([f_r, f_g, f_b], axis=-1).astype(np.float32)
        radial_env = np.exp(-(r ** 2) / 1.45) * 0.85 + 0.15
        bg = c_base + facet_col * (shade * 0.26 * radial_env)[..., None]
        # Crisp crystalline fracture highlight lines
        bg += (facet_col * 0.65 + 0.35) * (edge * 0.36 * radial_env)[..., None]
        # Central Terastal diamond burst
        d_diamond = np.abs(u) + np.abs(v_iso - cy)
        burst = np.exp(-((d_diamond - 0.72) ** 2) / 0.00028) * 0.75 + np.exp(-((d_diamond - 0.92) ** 2) / 0.00022) * 0.50
        bg += np.array([0.82, 0.90, 1.0], dtype=np.float32)[None, None, :] * (burst * 0.42)[..., None]

    elif name == "guofeng_jade":
        # 东方国风青绿仙云：深黛青 + 矿彩青绿层叠山岚祥云 + 鎏金圆日与朱砂点翠星尘
        c_sky_top = np.array([0.028, 0.068, 0.078], dtype=np.float32)
        c_sky_bot = np.array([0.055, 0.135, 0.125], dtype=np.float32)
        c_jade = np.array([0.16, 0.44, 0.38], dtype=np.float32)
        c_gold = np.array([0.88, 0.73, 0.40], dtype=np.float32)
        c_vermilion = np.array([0.76, 0.26, 0.22], dtype=np.float32)
        v01 = np.clip((v + 1.0) * 0.5, 0.0, 1.0)
        bg = c_sky_top * (1.0 - v01[..., None]) + c_sky_bot * v01[..., None]
        cloud = _fractal_cloud(h, w, seed=733)
        bg += c_jade * (cloud * 0.26 * (1.0 - v01 * 0.35))[..., None]
        # Layered ink-wash Guofeng mountain silhouettes + golden mist ridges near bottom half
        for idx, (base_v, amp, freq, phase, tint, gold_rim) in enumerate([
            (0.28, 0.11, 2.4, 0.5, c_jade * 0.55, 0.35),
            (0.46, 0.12, 3.1, 2.1, c_jade * 0.40, 0.42),
            (0.66, 0.10, 2.8, 4.0, np.array([0.035, 0.095, 0.095], dtype=np.float32), 0.48),
        ]):
            ridge = base_v + amp * np.sin(u * freq + phase) + 0.045 * np.cos(u * freq * 2.1 - phase)
            below = np.clip((v - ridge) * 18.0, 0.0, 1.0)
            rim = np.exp(-((v - ridge) ** 2) / 0.00055)
            bg = bg * (1.0 - below[..., None] * 0.55) + tint[None, None, :] * (below[..., None] * 0.55)
            bg += c_gold * (rim * gold_rim)[..., None]
        # Classical gilded solar/lunar disc behind upper torso + delicate swirling auspicious cloud ribbons
        disc = np.exp(-((r - 0.58) ** 2) / 0.00018) * 0.85 + np.exp(-(r ** 2) / 0.38) * 0.18
        bg += (c_gold * 0.82 + c_vermilion * 0.18)[None, None, :] * (disc * 0.42)[..., None]

    elif name == "cyber_holo_grid":
        # 赛博全息霓虹矩阵：深电光紫黑 + 六边形科技蜂巢光栅 + 透视全息扫描环 + 青粉双霓虹
        c_deep = np.array([0.026, 0.024, 0.068], dtype=np.float32)
        c_cyan = np.array([0.16, 0.86, 0.96], dtype=np.float32)
        c_magenta = np.array([0.92, 0.24, 0.68], dtype=np.float32)
        bg = np.ones((h, w, 3), dtype=np.float32) * c_deep[None, None, :]
        # Dual neon atmospheric corner glows
        g_left = np.exp(-(((u + 0.65) ** 2) / 0.85 + ((v + 0.25) ** 2) / 1.1))
        g_right = np.exp(-(((u - 0.65) ** 2) / 0.85 + ((v - 0.20) ** 2) / 1.1))
        bg += c_cyan * (g_left * 0.22)[..., None] + c_magenta * (g_right * 0.22)[..., None]
        # Hex / triangular holographic tech lattice
        s3 = 1.73205
        gx1 = np.exp(-((np.sin(u * 20.0) ** 2) / 0.015))
        gx2 = np.exp(-((np.sin((u * 0.5 + v_iso * (s3 * 0.5)) * 20.0) ** 2) / 0.015))
        gx3 = np.exp(-((np.sin((u * 0.5 - v_iso * (s3 * 0.5)) * 20.0) ** 2) / 0.015))
        tech_grid = np.clip(np.maximum(gx1, np.maximum(gx2, gx3)), 0.0, 1.0) * 0.22 * np.exp(-r * 0.42)
        # Segmented HUD radar rings around center
        hud_r1 = np.exp(-((r - 0.62) ** 2) / 0.00018) * (0.5 + 0.5 * np.sign(np.cos(ang * 6.0) + 0.25))
        hud_r2 = np.exp(-((r - 0.78) ** 2) / 0.00014) * ((0.5 + 0.5 * np.cos(ang * 36.0)) ** 6.0)
        # Subtle horizontal scanlines
        scan = (0.5 + 0.5 * np.sin(yy * 0.9)) * 0.05
        bg += c_cyan * (tech_grid * 0.65 + hud_r1 * 0.36 + scan)[..., None]
        bg += c_magenta * (tech_grid * 0.45 + hud_r2 * 0.40)[..., None]

    elif name == "sakura_twilight":
        # 绯夜樱吹流光：子夜靛蓝过渡到绯樱暮紫 + 柔辉满月 + 梦幻散景樱瓣光斑
        c_top = np.array([0.042, 0.036, 0.095], dtype=np.float32)
        c_bot = np.array([0.165, 0.072, 0.138], dtype=np.float32)
        c_sakura = np.array([0.96, 0.64, 0.78], dtype=np.float32)
        c_gold = np.array([0.95, 0.84, 0.64], dtype=np.float32)
        v01 = np.clip((v + 1.0) * 0.5, 0.0, 1.0)
        cloud = _fractal_cloud(h, w, seed=827)
        bg = c_top * (1.0 - v01[..., None]) + c_bot * v01[..., None]
        bg += c_sakura * (cloud * 0.20)[..., None]
        # Soft luminous twilight moon behind upper center
        moon_glow = np.exp(-(r ** 2) / 0.45) * 0.34
        moon_rim = np.exp(-((r - 0.52) ** 2) / 0.00028) * 0.48
        bg += (c_gold * 0.60 + c_sakura * 0.40)[None, None, :] * (moon_glow + moon_rim)[..., None]
        # Soft bokeh sakura petals drifting diagonally
        rng_p = np.random.default_rng(919)
        petals = np.zeros((h, w), dtype=np.float32)
        for _ in range(42):
            px = int(rng_p.integers(24, w - 24))
            py = int(rng_p.integers(24, h - 24))
            rx_p = int(rng_p.integers(6, 16))
            ry_p = max(3, int(rx_p * rng_p.uniform(0.45, 0.70)))
            rot = float(rng_p.uniform(0, 180))
            val = float(rng_p.uniform(0.25, 0.65))
            cv2.ellipse(petals, (px, py), (rx_p, ry_p), rot, 0, 360, val, -1, cv2.LINE_AA)
        petals = cv2.GaussianBlur(petals, (0, 0), sigmaX=3.2)
        bg += c_sakura * (petals * 0.48)[..., None]

    else:  # moon_astrolabe
        c_deep = np.array([0.026, 0.032, 0.074], dtype=np.float32)
        c_mid = np.array([0.078, 0.090, 0.190], dtype=np.float32)
        c_glow = np.array([0.46, 0.60, 0.94], dtype=np.float32)
        c_sec = np.array([0.74, 0.58, 0.92], dtype=np.float32)
        cloud = _fractal_cloud(h, w, seed=107)
        cloud2 = _fractal_cloud(h, w, seed=127)
        bg = c_deep + c_mid * (cloud[..., None] * 0.78)
        bg += c_glow * (np.exp(-(r ** 2) / 0.60) * 0.34)[..., None]
        bg += c_sec * ((cloud * cloud2) * np.exp(-(r ** 2) / 1.55) * 0.32)[..., None]
        ring_a = np.exp(-((r - 0.64) ** 2) / 0.00015)
        ring_b = np.exp(-((r - 0.74) ** 2) / 0.00011)
        ring_c = np.exp(-((r - 0.98) ** 2) / 0.00016)
        ticks = ((0.5 + 0.5 * np.cos(ang * 12.0)) ** 22.0) * np.exp(-((r - 0.69) ** 2) / 0.0020)
        axis_v = np.exp(-(u * u) / 0.000035) * np.clip(1.0 - np.abs(v) * 0.75, 0.0, 1.0) * 0.45
        orn = np.clip(ring_a * 0.85 + ring_b * 0.68 + ring_c * 0.45 + ticks * 0.75 + axis_v, 0.0, 1.0)
        bg += (c_glow * 0.55 + c_sec * 0.45) * (orn * 0.38)[..., None]

    # Subtle starfield dust
    rng = np.random.default_rng(abs(hash(name)) % 10000 + 77)
    stars = np.zeros((h, w), dtype=np.float32)
    for _ in range(140):
        sx = int(rng.integers(12, w - 12))
        sy = int(rng.integers(12, h - 12))
        sr = int(rng.choice([1, 1, 1, 2]))
        val = float(rng.uniform(0.32, 0.92))
        cv2.circle(stars, (sx, sy), sr, val, -1, cv2.LINE_AA)
    stars = cv2.GaussianBlur(stars, (0, 0), sigmaX=0.85)
    warm_bg = name in ("classical_arcana", "sun_sanctuary", "art_nouveau_halo", "guofeng_jade")
    star_tint = np.array([1.0, 0.92, 0.78] if warm_bg else [0.88, 0.90, 1.0], dtype=np.float32)
    bg += stars[..., None] * star_tint[None, None, :] * 0.52

    vig = np.clip(1.0 - 0.32 * (u * u * 0.85 + v * v * 0.70), 0.42, 1.0)[..., None]
    return np.clip(bg * vig, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 10 Collector Transparent Foreground Ornate Frame Templates (RGBA)
# ---------------------------------------------------------------------------

def _draw_four_point_star(canvas: np.ndarray, cx: int, cy: int, rad: int, val: int = 255) -> None:
    pts = np.array([
        [cx, cy - rad],
        [cx + max(2, rad // 4), cy - max(2, rad // 4)],
        [cx + rad, cy],
        [cx + max(2, rad // 4), cy + max(2, rad // 4)],
        [cx, cy + rad],
        [cx - max(2, rad // 4), cy + max(2, rad // 4)],
        [cx - rad, cy],
        [cx - max(2, rad // 4), cy - max(2, rad // 4)],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], val, lineType=cv2.LINE_AA)


def _draw_diamond(canvas: np.ndarray, cx: int, cy: int, rx: int, ry: int, val: int = 255, thickness: int = -1) -> None:
    pts = np.array([[cx, cy - ry], [cx + rx, cy], [cx, cy + ry], [cx - rx, cy]], dtype=np.int32)
    if thickness < 0:
        cv2.fillPoly(canvas, [pts], val, lineType=cv2.LINE_AA)
    else:
        cv2.polylines(canvas, [pts], True, val, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_sakura_blossom(canvas: np.ndarray, cx: int, cy: int, rad: int, val: int = 255) -> None:
    for k in range(5):
        ang = math.radians(k * 72.0 - 90.0)
        px = int(round(cx + math.cos(ang) * rad * 0.52))
        py = int(round(cy + math.sin(ang) * rad * 0.52))
        cv2.ellipse(canvas, (px, py), (max(3, int(rad * 0.48)), max(2, int(rad * 0.28))),
                    k * 72.0 - 90.0, 0, 360, val, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), max(2, rad // 5), max(120, val - 60), -1, cv2.LINE_AA)


def generate_foreground_frame_template(
    name: str,
    w: int = 720,
    h: int = 1152,
    metal_rgb: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    """
    Generate one of the 10 collector transparent RGBA frames (+ `minimal_astral`),
    leaving the central illustration window crystal-clear.
    Returns (H, W, 4) float32 in [0, 1].
    """
    name = _FG_ALIASES.get(name, name)
    mask_u8 = np.zeros((h, w), dtype=np.uint8)
    highlight_u8 = np.zeros((h, w), dtype=np.uint8)

    m_out = max(14, int(round(w * 0.032)))
    m_in = max(28, int(round(w * 0.062)))
    lw_thick = max(2, int(round(w * 0.0055)))
    lw_thin = max(1, int(round(w * 0.0028)))

    if name not in ("sar_prismatic_border", "cyber_hud_frame"):
        cv2.rectangle(mask_u8, (m_out, m_out), (w - 1 - m_out, h - 1 - m_out), 235, thickness=lw_thick, lineType=cv2.LINE_AA)
        cv2.rectangle(highlight_u8, (m_out, m_out), (w - 1 - m_out, h - 1 - m_out), 255, thickness=max(1, lw_thick // 2), lineType=cv2.LINE_AA)

    if name == "minimal_astral":
        cv2.rectangle(mask_u8, (m_in, m_in), (w - 1 - m_in, h - 1 - m_in), 205, thickness=lw_thin, lineType=cv2.LINE_AA)
        d = max(7, int(round(w * 0.016)))
        for cx, cy in [(m_in, m_in), (w - 1 - m_in, m_in), (m_in, h - 1 - m_in), (w - 1 - m_in, h - 1 - m_in)]:
            _draw_four_point_star(mask_u8, cx, cy, d, 255)
            _draw_four_point_star(highlight_u8, cx, cy, max(4, int(d * 0.7)), 255)

    elif name == "moon_arch":
        cv2.rectangle(mask_u8, (m_in, m_in), (w - 1 - m_in, h - 1 - m_in), 220, thickness=lw_thin, lineType=cv2.LINE_AA)
        arc_r = max(34, int(round(w * 0.095)))
        for cx, cy, a0, a1 in [
            (m_out, m_out, 0, 90), (w - 1 - m_out, m_out, 90, 180),
            (w - 1 - m_out, h - 1 - m_out, 180, 270), (m_out, h - 1 - m_out, 270, 360),
        ]:
            cv2.ellipse(mask_u8, (cx, cy), (arc_r, arc_r), 0, a0, a1, 245, thickness=lw_thick, lineType=cv2.LINE_AA)
            cv2.ellipse(mask_u8, (cx, cy), (int(arc_r * 0.76), int(arc_r * 0.76)), 0, a0, a1, 200, thickness=lw_thin, lineType=cv2.LINE_AA)
            cv2.ellipse(highlight_u8, (cx, cy), (arc_r, arc_r), 0, a0, a1, 255, thickness=lw_thin, lineType=cv2.LINE_AA)
        cr_r = max(10, int(round(w * 0.024)))
        for cy_c, sign in [(int((m_out + m_in) * 0.58), 1), (h - 1 - int((m_out + m_in) * 0.58), -1)]:
            cv2.circle(mask_u8, (w // 2, cy_c), cr_r, 240, thickness=lw_thin + 1, lineType=cv2.LINE_AA)
            _draw_four_point_star(mask_u8, w // 2, cy_c + sign * (cr_r + 6), max(8, int(cr_r * 0.85)), 255)
            _draw_four_point_star(highlight_u8, w // 2, cy_c + sign * (cr_r + 6), max(5, int(cr_r * 0.55)), 255)

    elif name == "sun_baroque":
        ch = max(32, int(round(w * 0.092)))
        poly_in = np.array([
            [m_in + ch, m_in], [w - 1 - m_in - ch, m_in],
            [w - 1 - m_in, m_in + ch], [w - 1 - m_in, h - 1 - m_in - ch],
            [w - 1 - m_in - ch, h - 1 - m_in], [m_in + ch, h - 1 - m_in],
            [m_in, h - 1 - m_in - ch], [m_in, m_in + ch],
        ], dtype=np.int32)
        cv2.polylines(mask_u8, [poly_in], True, 242, thickness=lw_thick, lineType=cv2.LINE_AA)
        cv2.polylines(highlight_u8, [poly_in], True, 255, thickness=lw_thin, lineType=cv2.LINE_AA)
        for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
            bx = m_out if sx == 1 else w - 1 - m_out
            by = m_out if sy == 1 else h - 1 - m_out
            for t in np.linspace(0.16, 0.84, 5):
                cv2.line(mask_u8, (bx, by), (int(bx + sx * ch * t), int(by + sy * ch * (1.0 - t))), 220, thickness=lw_thin, lineType=cv2.LINE_AA)
            _draw_four_point_star(mask_u8, int(bx + sx * ch * 0.36), int(by + sy * ch * 0.36), max(9, ch // 4), 255)
        for cx, cy in [(w // 2, m_in), (w // 2, h - 1 - m_in), (m_in, h // 2), (w - 1 - m_in, h // 2)]:
            _draw_four_point_star(mask_u8, cx, cy, max(11, int(w * 0.025)), 255)
            _draw_four_point_star(highlight_u8, cx, cy, max(6, int(w * 0.015)), 255)

    elif name == "gothic_cathedral":
        cv2.rectangle(mask_u8, (m_in, m_in), (w - 1 - m_in, h - 1 - m_in), 228, thickness=lw_thin, lineType=cv2.LINE_AA)
        span = max(38, int(round(w * 0.115)))
        for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
            bx = m_out if sx == 1 else w - 1 - m_out
            by = m_out if sy == 1 else h - 1 - m_out
            pts_arch = np.array([
                [bx + sx * span, by],
                [int(bx + sx * span * 0.62), int(by + sy * span * 0.22)],
                [int(bx + sx * span * 0.72), int(by + sy * span * 0.72)],
                [int(bx + sx * span * 0.22), int(by + sy * span * 0.62)],
                [bx, by + sy * span],
            ], dtype=np.int32)
            cv2.polylines(mask_u8, [pts_arch], False, 248, thickness=lw_thick, lineType=cv2.LINE_AA)
            cv2.polylines(highlight_u8, [pts_arch], False, 255, thickness=lw_thin, lineType=cv2.LINE_AA)
            cv2.line(mask_u8, (bx, by), (int(bx + sx * span * 0.85), int(by + sy * span * 0.85)), 235, thickness=lw_thin, lineType=cv2.LINE_AA)
            _draw_diamond(mask_u8, int(bx + sx * span * 0.42), int(by + sy * span * 0.42), max(7, span // 6), max(7, span // 6), 255, -1)
        for cx, cy, rx, ry in [
            (w // 2, m_in, max(8, int(w * 0.018)), max(16, int(w * 0.036))),
            (w // 2, h - 1 - m_in, max(8, int(w * 0.018)), max(16, int(w * 0.036))),
            (m_in, h // 2, max(14, int(w * 0.030)), max(7, int(w * 0.016))),
            (w - 1 - m_in, h // 2, max(14, int(w * 0.030)), max(7, int(w * 0.016))),
        ]:
            _draw_diamond(mask_u8, cx, cy, rx, ry, 255, -1)
            _draw_diamond(highlight_u8, cx, cy, max(4, rx // 2), max(4, ry // 2), 255, -1)

    elif name == "dream_crystal":
        cv2.rectangle(mask_u8, (m_in, m_in), (w - 1 - m_in, h - 1 - m_in), 215, thickness=lw_thin, lineType=cv2.LINE_AA)
        span = max(36, int(round(w * 0.105)))
        for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
            bx = m_in if sx == 1 else w - 1 - m_in
            by = m_in if sy == 1 else h - 1 - m_in
            c1x, c1y = int(bx + sx * span * 0.32), int(by + sy * span * 0.32)
            _draw_diamond(mask_u8, c1x, c1y, max(10, span // 3), max(14, int(span * 0.42)), 245, lw_thin + 1)
            _draw_diamond(highlight_u8, c1x, c1y, max(6, span // 5), max(8, span // 4), 255, -1)
            _draw_diamond(mask_u8, int(bx + sx * span * 0.78), by, max(6, span // 6), max(4, span // 8), 240, -1)
            _draw_diamond(mask_u8, bx, int(by + sy * span * 0.78), max(4, span // 8), max(6, span // 6), 240, -1)
            cv2.line(mask_u8, (int(bx + sx * span * 0.78), by), (c1x, c1y), 205, thickness=lw_thin, lineType=cv2.LINE_AA)
            cv2.line(mask_u8, (bx, int(by + sy * span * 0.78)), (c1x, c1y), 205, thickness=lw_thin, lineType=cv2.LINE_AA)
        for cx, cy in [(w // 2, m_in), (w // 2, h - 1 - m_in)]:
            _draw_diamond(mask_u8, cx, cy, max(14, int(w * 0.032)), max(9, int(w * 0.020)), 245, lw_thin + 1)
            _draw_four_point_star(mask_u8, cx, cy, max(9, int(w * 0.020)), 255)
            _draw_four_point_star(highlight_u8, cx, cy, max(5, int(w * 0.012)), 255)

    elif name == "art_nouveau_arch":
        # 新艺术穆夏穹顶拱窗框：顶部优雅大圆拱 + 细长双柱侧边 + 顶部圆环花饰 + 底部花瓣托
        arch_rx = (w - 2 * m_in) // 2
        arch_ry = int(round(arch_rx * 0.72))
        arch_cy = m_in + arch_ry
        # Top sweeping Art Nouveau dome arch
        cv2.ellipse(mask_u8, (w // 2, arch_cy), (arch_rx, arch_ry), 0, 180, 360, 248, thickness=lw_thick, lineType=cv2.LINE_AA)
        cv2.ellipse(highlight_u8, (w // 2, arch_cy), (arch_rx, arch_ry), 0, 180, 360, 255, thickness=lw_thin, lineType=cv2.LINE_AA)
        cv2.ellipse(mask_u8, (w // 2, arch_cy), (arch_rx - 8, arch_ry - 8), 0, 180, 360, 205, thickness=lw_thin, lineType=cv2.LINE_AA)
        # Side vertical rails down from arch spring-line & bottom rail
        cv2.line(mask_u8, (m_in, arch_cy), (m_in, h - 1 - m_in), 232, thickness=lw_thin + 1, lineType=cv2.LINE_AA)
        cv2.line(mask_u8, (w - 1 - m_in, arch_cy), (w - 1 - m_in, h - 1 - m_in), 232, thickness=lw_thin + 1, lineType=cv2.LINE_AA)
        cv2.line(mask_u8, (m_in, h - 1 - m_in), (w - 1 - m_in, h - 1 - m_in), 232, thickness=lw_thin + 1, lineType=cv2.LINE_AA)
        # Upper corner Mucha circular rosette medallions tucked cleanly in the spandrels
        med_r = max(14, int(round(w * 0.036)))
        for mx in (m_out + med_r + 10, w - 1 - m_out - med_r - 10):
            my = m_out + med_r + 10
            cv2.circle(mask_u8, (mx, my), med_r, 240, thickness=lw_thin + 1, lineType=cv2.LINE_AA)
            cv2.circle(mask_u8, (mx, my), max(5, med_r // 2), 215, thickness=lw_thin, lineType=cv2.LINE_AA)
            _draw_four_point_star(mask_u8, mx, my, max(7, int(med_r * 0.55)), 255)
            _draw_four_point_star(highlight_u8, mx, my, max(4, int(med_r * 0.35)), 255)
        # Bottom corners organic whip-curve arcs
        b_arc = max(24, int(round(w * 0.068)))
        cv2.ellipse(mask_u8, (m_in, h - 1 - m_in), (b_arc, b_arc), 0, 270, 360, 235, thickness=lw_thin + 1, lineType=cv2.LINE_AA)
        cv2.ellipse(mask_u8, (w - 1 - m_in, h - 1 - m_in), (b_arc, b_arc), 0, 180, 270, 235, thickness=lw_thin + 1, lineType=cv2.LINE_AA)
        _draw_four_point_star(mask_u8, w // 2, m_in, max(11, int(w * 0.025)), 255)
        _draw_four_point_star(mask_u8, w // 2, h - 1 - m_in, max(11, int(w * 0.025)), 255)

    elif name == "sar_prismatic_border":
        # 宝可梦 SAR 碎钻切角银金框：现代双斜切外框 + 四角菱钻切面 + 边缘棱镜刻度
        ch_o = max(22, int(round(w * 0.062)))
        ch_i = max(28, int(round(w * 0.080)))
        poly_o = np.array([
            [m_out + ch_o, m_out], [w - 1 - m_out - ch_o, m_out],
            [w - 1 - m_out, m_out + ch_o], [w - 1 - m_out, h - 1 - m_out - ch_o],
            [w - 1 - m_out - ch_o, h - 1 - m_out], [m_out + ch_o, h - 1 - m_out],
            [m_out, h - 1 - m_out - ch_o], [m_out, m_out + ch_o],
        ], dtype=np.int32)
        poly_i = np.array([
            [m_in + ch_i, m_in], [w - 1 - m_in - ch_i, m_in],
            [w - 1 - m_in, m_in + ch_i], [w - 1 - m_in, h - 1 - m_in - ch_i],
            [w - 1 - m_in - ch_i, h - 1 - m_in], [m_in + ch_i, h - 1 - m_in],
            [m_in, h - 1 - m_in - ch_i], [m_in, m_in + ch_i],
        ], dtype=np.int32)
        cv2.polylines(mask_u8, [poly_o], True, 245, thickness=lw_thick + 1, lineType=cv2.LINE_AA)
        cv2.polylines(highlight_u8, [poly_o], True, 255, thickness=lw_thin, lineType=cv2.LINE_AA)
        cv2.polylines(mask_u8, [poly_i], True, 215, thickness=lw_thin, lineType=cv2.LINE_AA)
        # Faceted prism corner trusses & gems
        for p0, p1 in zip(poly_o, poly_i):
            cv2.line(mask_u8, tuple(p0), tuple(p1), 235, thickness=lw_thin, lineType=cv2.LINE_AA)
        for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
            cx_g = int((m_out if sx == 1 else w - 1 - m_out) + sx * ch_o * 0.62)
            cy_g = int((m_out if sy == 1 else h - 1 - m_out) + sy * ch_o * 0.62)
            _draw_diamond(mask_u8, cx_g, cy_g, max(8, ch_o // 4), max(8, ch_o // 4), 255, -1)
            _draw_diamond(highlight_u8, cx_g, cy_g, max(4, ch_o // 7), max(4, ch_o // 7), 255, -1)
        # Subtle side & top/bottom prismatic accent dashes
        dash = max(18, int(w * 0.055))
        cv2.line(mask_u8, (w // 2 - dash, m_out + 7), (w // 2 + dash, m_out + 7), 245, thickness=lw_thin + 1, lineType=cv2.LINE_AA)
        cv2.line(mask_u8, (w // 2 - dash, h - 1 - m_out - 7), (w // 2 + dash, h - 1 - m_out - 7), 245, thickness=lw_thin + 1, lineType=cv2.LINE_AA)

    elif name == "guofeng_cloud_frame":
        # 东方国风回纹祥云窗棂框：经典四角回字纹（Meander/Key Fret）+ 雅致双框 + 菱形玉佩饰扣
        cv2.rectangle(mask_u8, (m_in, m_in), (w - 1 - m_in, h - 1 - m_in), 225, thickness=lw_thin, lineType=cv2.LINE_AA)
        u_step = max(12, int(round(w * 0.030)))
        for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
            bx = m_out if sx == 1 else w - 1 - m_out
            by = m_out if sy == 1 else h - 1 - m_out
            # Classic Chinese key-fret spiral corner motif (回纹)
            fret = np.array([
                [bx + sx * int(u_step * 3.6), by],
                [bx + sx * int(u_step * 3.6), by + sy * int(u_step * 1.4)],
                [bx + sx * int(u_step * 2.2), by + sy * int(u_step * 1.4)],
                [bx + sx * int(u_step * 2.2), by + sy * int(u_step * 0.6)],
                [bx + sx * int(u_step * 2.9), by + sy * int(u_step * 0.6)],
                [bx + sx * int(u_step * 2.9), by + sy * int(u_step * 1.0)],
            ], dtype=np.int32)
            fret_v = np.array([
                [bx, by + sy * int(u_step * 3.6)],
                [bx + sx * int(u_step * 1.4), by + sy * int(u_step * 3.6)],
                [bx + sx * int(u_step * 1.4), by + sy * int(u_step * 2.2)],
                [bx + sx * int(u_step * 0.6), by + sy * int(u_step * 2.2)],
                [bx + sx * int(u_step * 0.6), by + sy * int(u_step * 2.9)],
                [bx + sx * int(u_step * 1.0), by + sy * int(u_step * 2.9)],
            ], dtype=np.int32)
            cv2.polylines(mask_u8, [fret], False, 248, thickness=lw_thin + 1, lineType=cv2.LINE_AA)
            cv2.polylines(mask_u8, [fret_v], False, 248, thickness=lw_thin + 1, lineType=cv2.LINE_AA)
            cv2.polylines(highlight_u8, [fret], False, 255, thickness=lw_thin, lineType=cv2.LINE_AA)
            cv2.polylines(highlight_u8, [fret_v], False, 255, thickness=lw_thin, lineType=cv2.LINE_AA)
            # Corner square jade seal box
            cv2.rectangle(
                mask_u8,
                (bx + sx * int(u_step * 0.4), by + sy * int(u_step * 0.4)),
                (bx + sx * int(u_step * 1.55), by + sy * int(u_step * 1.55)),
                245, thickness=lw_thin + 1, lineType=cv2.LINE_AA,
            )
        # Top/bottom & side Chinese jade-pendant diamond medallions
        for cx, cy in [(w // 2, m_in), (w // 2, h - 1 - m_in)]:
            _draw_diamond(mask_u8, cx, cy, max(16, int(w * 0.036)), max(10, int(w * 0.022)), 250, lw_thin + 1)
            _draw_diamond(highlight_u8, cx, cy, max(8, int(w * 0.018)), max(5, int(w * 0.011)), 255, -1)

    elif name == "cyber_hud_frame":
        # 赛博机能 HUD 装甲框：45° 硬朗切角 + 四角战术括号 + 精细侧边微刻度（无乱码文字）
        ch = max(26, int(round(w * 0.072)))
        poly_o = np.array([
            [m_out + ch, m_out], [w - 1 - m_out - ch, m_out],
            [w - 1 - m_out, m_out + ch], [w - 1 - m_out, h - 1 - m_out - ch],
            [w - 1 - m_out - ch, h - 1 - m_out], [m_out + ch, h - 1 - m_out],
            [m_out, h - 1 - m_out - ch], [m_out, m_out + ch],
        ], dtype=np.int32)
        cv2.polylines(mask_u8, [poly_o], True, 238, thickness=lw_thin + 1, lineType=cv2.LINE_AA)
        cv2.polylines(highlight_u8, [poly_o], True, 255, thickness=lw_thin, lineType=cv2.LINE_AA)
        # Heavy tactical corner armor brackets
        arm = max(46, int(round(w * 0.13)))
        for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
            bx = m_out if sx == 1 else w - 1 - m_out
            by = m_out if sy == 1 else h - 1 - m_out
            br = np.array([
                [bx + sx * arm, by],
                [bx + sx * ch, by],
                [bx, by + sy * ch],
                [bx, by + sy * arm],
            ], dtype=np.int32)
            cv2.polylines(mask_u8, [br], False, 255, thickness=lw_thick + 2, lineType=cv2.LINE_AA)
            cv2.polylines(highlight_u8, [br], False, 255, thickness=lw_thin + 1, lineType=cv2.LINE_AA)
            # Inner corner HUD crosshair
            cx_h = int(bx + sx * ch * 0.95)
            cy_h = int(by + sy * ch * 0.95)
            cv2.line(mask_u8, (cx_h - 6, cy_h), (cx_h + 6, cy_h), 245, thickness=lw_thin, lineType=cv2.LINE_AA)
            cv2.line(mask_u8, (cx_h, cy_h - 6), (cx_h, cy_h + 6), 245, thickness=lw_thin, lineType=cv2.LINE_AA)
        # Fine telemetry tick marks along left & right vertical rails
        for ty in np.linspace(h * 0.32, h * 0.68, 9):
            t_len = 12 if int(ty) % 2 == 0 else 7
            cv2.line(mask_u8, (m_out + 4, int(ty)), (m_out + 4 + t_len, int(ty)), 230, thickness=lw_thin, lineType=cv2.LINE_AA)
            cv2.line(mask_u8, (w - 1 - m_out - 4, int(ty)), (w - 1 - m_out - 4 - t_len, int(ty)), 230, thickness=lw_thin, lineType=cv2.LINE_AA)

    elif name == "sakura_lace_frame":
        # 绯夜樱吹和风蕾丝框：双细线框 + 四角五瓣樱花饰徽 + 柔美扇形蕾丝边弧
        cv2.rectangle(mask_u8, (m_in, m_in), (w - 1 - m_in, h - 1 - m_in), 215, thickness=lw_thin, lineType=cv2.LINE_AA)
        fl_r = max(14, int(round(w * 0.034)))
        for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
            bx = m_in if sx == 1 else w - 1 - m_in
            by = m_in if sy == 1 else h - 1 - m_in
            _draw_sakura_blossom(mask_u8, int(bx + sx * fl_r * 0.45), int(by + sy * fl_r * 0.45), fl_r, 250)
            _draw_sakura_blossom(highlight_u8, int(bx + sx * fl_r * 0.45), int(by + sy * fl_r * 0.45), max(7, int(fl_r * 0.62)), 255)
            # Small twin blossom buds along edges
            _draw_sakura_blossom(mask_u8, int(bx + sx * fl_r * 1.9), by, max(7, int(fl_r * 0.48)), 235)
            _draw_sakura_blossom(mask_u8, bx, int(by + sy * fl_r * 1.9), max(7, int(fl_r * 0.48)), 235)
        # Delicate scalloped lace arcs on top & bottom centers
        for cy_l, a0, a1 in [(m_in, 0, 180), (h - 1 - m_in, 180, 360)]:
            for off_x in (-24, 0, 24):
                cv2.ellipse(mask_u8, (w // 2 + off_x, cy_l), (12, 8), 0, a0, a1, 225, thickness=lw_thin, lineType=cv2.LINE_AA)
            _draw_sakura_blossom(mask_u8, w // 2, cy_l, max(9, int(fl_r * 0.65)), 255)

    else:  # classical_filigree
        cv2.rectangle(mask_u8, (m_in, m_in), (w - 1 - m_in, h - 1 - m_in), 228, thickness=lw_thin, lineType=cv2.LINE_AA)
        c_box = max(32, int(round(w * 0.090)))
        for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
            bx = m_out if sx == 1 else w - 1 - m_out
            by = m_out if sy == 1 else h - 1 - m_out
            pts_l = np.array([
                [bx + sx * c_box, by],
                [bx + sx * c_box, by + sy * int(c_box * 0.34)],
                [bx + sx * int(c_box * 0.34), by + sy * int(c_box * 0.34)],
                [bx + sx * int(c_box * 0.34), by + sy * c_box],
                [bx, by + sy * c_box],
            ], dtype=np.int32)
            cv2.polylines(mask_u8, [pts_l], False, 245, thickness=lw_thin + 1, lineType=cv2.LINE_AA)
            cv2.polylines(highlight_u8, [pts_l], False, 255, thickness=lw_thin, lineType=cv2.LINE_AA)
            cx_s = int(bx + sx * c_box * 0.56)
            cy_s = int(by + sy * c_box * 0.56)
            _draw_four_point_star(mask_u8, cx_s, cy_s, max(10, int(c_box * 0.28)), 255)
            _draw_four_point_star(highlight_u8, cx_s, cy_s, max(6, int(c_box * 0.18)), 255)
            cv2.line(mask_u8, (bx + sx * c_box, by + sy * int(c_box * 0.34)),
                     (bx + sx * int(c_box * 0.34), by + sy * c_box), 210, thickness=lw_thin, lineType=cv2.LINE_AA)
        for cx, cy, sz in [
            (w // 2, m_in, max(12, int(w * 0.028))),
            (w // 2, h - 1 - m_in, max(12, int(w * 0.028))),
            (m_in, h // 2, max(9, int(w * 0.021))),
            (w - 1 - m_in, h // 2, max(9, int(w * 0.021))),
        ]:
            _draw_four_point_star(mask_u8, cx, cy, sz, 255)
            _draw_four_point_star(highlight_u8, cx, cy, max(5, int(sz * 0.6)), 255)

    alpha = mask_u8.astype(np.float32) / 255.0
    hi = highlight_u8.astype(np.float32) / 255.0

    default_metals = {
        "classical_filigree": (0.88, 0.78, 0.56),
        "moon_arch": (0.80, 0.85, 0.92),
        "sun_baroque": (0.90, 0.72, 0.36),
        "gothic_cathedral": (0.82, 0.64, 0.50),
        "dream_crystal": (0.76, 0.86, 0.94),
        "art_nouveau_arch": (0.88, 0.75, 0.45),
        "sar_prismatic_border": (0.84, 0.88, 0.94),
        "guofeng_cloud_frame": (0.89, 0.75, 0.42),
        "cyber_hud_frame": (0.48, 0.88, 0.96),
        "sakura_lace_frame": (0.94, 0.74, 0.82),
        "minimal_astral": (0.88, 0.80, 0.60),
    }
    base = np.array(metal_rgb if metal_rgb is not None else default_metals.get(name, (0.86, 0.74, 0.48)), dtype=np.float32)
    shadow = base * 0.48
    bright = np.clip(base * 1.18 + 0.12, 0.0, 1.0)
    rgb = shadow[None, None, :] * (1.0 - alpha[..., None] * 0.55) + base[None, None, :] * (alpha[..., None] * 0.55)
    rgb = rgb * (1.0 - hi[..., None] * 0.65) + bright[None, None, :] * (hi[..., None] * 0.65)

    rgba = np.concatenate([np.clip(rgb, 0.0, 1.0), np.clip(alpha, 0.0, 1.0)[..., None]], axis=-1)
    return rgba.astype(np.float32)


# ---------------------------------------------------------------------------
# 7 Atmospheric Non-Black Studio Backdrop Presets (Behind Floating 3D Card)
# ---------------------------------------------------------------------------

def generate_studio_backdrop(name: str, w: int = 460, h: int = 766) -> np.ndarray:
    """
    Generate a rich, atmospheric, non-pure-black studio backdrop (RGB float32 [0,1])
    designed to frame the floating 3D holographic card with depth, velvet lighting,
    subtle volumetric mist/bokeh, and studio presence while keeping the card focal.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx / max(1, w - 1)) * 2.0 - 1.0
    v = (yy / max(1, h - 1)) * 2.0 - 1.0
    v01 = (v + 1.0) * 0.5
    r2 = u * u * 0.90 + v * v * 0.72
    rad_pool = np.exp(-r2 * 0.95)[..., None]
    top_spot = np.exp(-((u * 0.85) ** 2 + ((v + 0.55) * 0.72) ** 2) * 1.15)[..., None]

    if name == "royal_velvet":
        # 古典酒红鎏金丝绒棚：深勃艮第酒红渐变 + 暖金舞台聚光 + 丝绒云纹
        c_edge = np.array([0.115, 0.042, 0.068], dtype=np.float32)
        c_mid = np.array([0.275, 0.098, 0.145], dtype=np.float32)
        c_glow = np.array([0.62, 0.42, 0.20], dtype=np.float32)
        cloud = _fractal_cloud(h, w, seed=911)
        bg = c_edge * (1.0 - rad_pool) + c_mid * rad_pool
        bg += c_glow * (top_spot * 0.28 + cloud[..., None] * rad_pool * 0.12)
        # Soft velvet draped light folds
        folds = (0.5 + 0.5 * np.sin(u * 5.2 + v * 0.8)) ** 2.2 * 0.06
        bg += np.array([0.38, 0.16, 0.20], dtype=np.float32)[None, None, :] * folds[..., None]

    elif name == "emerald_mist":
        # 青绿云雾玉殿：深黛青绿渐变 + 流动仙雾 + 暖玉顶光
        c_edge = np.array([0.042, 0.105, 0.108], dtype=np.float32)
        c_mid = np.array([0.095, 0.245, 0.225], dtype=np.float32)
        c_glow = np.array([0.48, 0.68, 0.52], dtype=np.float32)
        cloud = _fractal_cloud(h, w, seed=923)
        mist = (0.5 + 0.5 * np.sin(v * 3.8 + np.sin(u * 2.4) * 1.1 + cloud * 2.5)) ** 2.0
        bg = c_edge * (1.0 - rad_pool) + c_mid * rad_pool
        bg += c_glow * (top_spot * 0.22 + mist[..., None] * 0.14)

    elif name == "sakura_dusk":
        # 绯樱暮紫柔焦棚：暮色紫罗兰到暖樱粉渐变 + 柔和散景光斑
        c_top = np.array([0.135, 0.088, 0.215], dtype=np.float32)
        c_bot = np.array([0.265, 0.135, 0.228], dtype=np.float32)
        c_bokeh = np.array([0.82, 0.56, 0.72], dtype=np.float32)
        cloud = _fractal_cloud(h, w, seed=937)
        bg = c_top * (1.0 - v01[..., None]) + c_bot * v01[..., None]
        bg += np.array([0.22, 0.12, 0.24], dtype=np.float32)[None, None, :] * (rad_pool * 0.55 + cloud[..., None] * 0.20)
        rng = np.random.default_rng(941)
        bokeh = np.zeros((h, w), dtype=np.float32)
        for _ in range(22):
            bx = int(rng.integers(int(w * 0.05), int(w * 0.95)))
            by = int(rng.integers(int(h * 0.05), int(h * 0.95)))
            br = int(rng.integers(max(8, w // 36), max(16, w // 14)))
            cv2.circle(bokeh, (bx, by), br, float(rng.uniform(0.18, 0.45)), -1, cv2.LINE_AA)
        bokeh = cv2.GaussianBlur(bokeh, (0, 0), sigmaX=max(6.0, w * 0.022))
        bg += c_bokeh * (bokeh * 0.36)[..., None]

    elif name == "aurora_teal":
        # 极光冰青梦境：深海蓝青渐变 + 极光丝带雾 + 蓝紫双色柔光
        c_edge = np.array([0.048, 0.095, 0.168], dtype=np.float32)
        c_mid = np.array([0.095, 0.235, 0.315], dtype=np.float32)
        c_aur1 = np.array([0.30, 0.76, 0.82], dtype=np.float32)
        c_aur2 = np.array([0.48, 0.36, 0.78], dtype=np.float32)
        cloud = _fractal_cloud(h, w, seed=953)
        ribbon = (0.5 + 0.5 * np.sin(u * 3.4 - v * 2.2 + cloud * 3.2)) ** 2.6
        bg = c_edge * (1.0 - rad_pool) + c_mid * rad_pool
        bg += c_aur1 * (ribbon[..., None] * 0.19) + c_aur2 * ((1.0 - ribbon[..., None]) * cloud[..., None] * 0.16)

    elif name == "champagne_gallery":
        # 香槟暖灰美术馆：高级莫兰迪暖炭灰 + 柔和画廊聚光墙面 + 微暖香槟底晕
        c_edge = np.array([0.105, 0.096, 0.112], dtype=np.float32)
        c_mid = np.array([0.225, 0.205, 0.215], dtype=np.float32)
        c_gold = np.array([0.66, 0.56, 0.40], dtype=np.float32)
        cloud = _fractal_cloud(h, w, seed=967)
        # Subtle gallery plaster/linen wall micro-texture
        bg = c_edge * (1.0 - rad_pool) + c_mid * rad_pool
        bg += c_gold * (top_spot * 0.22 + cloud[..., None] * 0.08)

    elif name == "cyber_studio":
        # 赛博双色霓虹棚：深电光靛紫 + 左右青/洋红霓虹氛围墙 + 极淡透视光栅
        c_base = np.array([0.075, 0.068, 0.148], dtype=np.float32)
        c_cyan = np.array([0.16, 0.68, 0.86], dtype=np.float32)
        c_pink = np.array([0.78, 0.22, 0.58], dtype=np.float32)
        g_l = np.exp(-(((u + 0.80) ** 2) / 0.95 + ((v + 0.15) ** 2) / 1.25))[..., None]
        g_r = np.exp(-(((u - 0.80) ** 2) / 0.95 + ((v - 0.15) ** 2) / 1.25))[..., None]
        bg = np.ones((h, w, 3), dtype=np.float32) * c_base[None, None, :]
        bg += c_cyan * (g_l * 0.30) + c_pink * (g_r * 0.28)
        # Very soft distant perspective horizon glow
        horiz = np.exp(-((v - 0.55) ** 2) / 0.08)[..., None] * 0.10
        bg += (c_cyan * 0.5 + c_pink * 0.5)[None, None, :] * horiz

    else:  # starry_indigo
        # 星夜靛蓝丝绒棚：深皇家靛蓝渐变 + 柔焦蓝紫银河云 + 细碎远处星点
        c_edge = np.array([0.058, 0.068, 0.145], dtype=np.float32)
        c_mid = np.array([0.135, 0.155, 0.295], dtype=np.float32)
        c_neb = np.array([0.46, 0.38, 0.72], dtype=np.float32)
        cloud = _fractal_cloud(h, w, seed=977)
        cloud2 = _fractal_cloud(h, w, seed=983)
        bg = c_edge * (1.0 - rad_pool) + c_mid * rad_pool
        bg += c_neb * ((cloud * cloud2)[..., None] * 0.30 + top_spot * 0.12)
        rng = np.random.default_rng(991)
        stars = np.zeros((h, w), dtype=np.float32)
        for _ in range(70):
            sx = int(rng.integers(6, w - 6))
            sy = int(rng.integers(6, h - 6))
            cv2.circle(stars, (sx, sy), 1, float(rng.uniform(0.25, 0.70)), -1, cv2.LINE_AA)
        stars = cv2.GaussianBlur(stars, (0, 0), sigmaX=0.8)
        bg += stars[..., None] * np.array([0.82, 0.88, 1.0], dtype=np.float32)[None, None, :] * 0.42

    # Gentle studio corner vignette (keeps edges rich colored, never pitch black!)
    vig = np.clip(1.0 - 0.18 * (u * u + v * v), 0.68, 1.0)[..., None]
    return np.clip(bg * vig, 0.0, 1.0).astype(np.float32)


def ensure_default_templates(template_dir: Optional[str] = None, force_regenerate: bool = False) -> Dict[str, Dict[str, str]]:
    """
    Ensure all 10 collector card background templates, 11 transparent foreground
    frame templates, and 7 studio backdrop templates exist in `./templates/`.
    """
    tdir = template_dir or _default_template_dir()
    os.makedirs(tdir, exist_ok=True)
    paths: Dict[str, Dict[str, str]] = {"bg": {}, "fg": {}, "studio": {}}

    for name, _ in BG_TEMPLATE_META:
        p = os.path.join(tdir, f"bg_{name}.png")
        if force_regenerate or not os.path.exists(p):
            arr = generate_background_template(name, w=720, h=1152)
            Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGB").save(p)
        paths["bg"][name] = p

    for name, _ in FG_TEMPLATE_META:
        p = os.path.join(tdir, f"fg_{name}.png")
        if force_regenerate or not os.path.exists(p):
            rgba = generate_foreground_frame_template(name, w=720, h=1152)
            Image.fromarray(np.clip(rgba * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGBA").save(p)
        paths["fg"][name] = p

    for name, _ in STUDIO_BG_META:
        p = os.path.join(tdir, f"studio_{name}.png")
        if force_regenerate or not os.path.exists(p):
            arr = generate_studio_backdrop(name, w=460, h=736)
            Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGB").save(p)
        paths["studio"][name] = p

    return paths


# ---------------------------------------------------------------------------
# Monocular Depth Synthesis Wrapper & ECC Affine Alignment
# ---------------------------------------------------------------------------

def synthesize_auto_depth_map(
    subject_rgb: np.ndarray,
    subject_alpha: np.ndarray,
) -> np.ndarray:
    """Delegate to the upgraded multi-cue 2D->depth estimator in depth_estimator.py."""
    return estimate_depth_fast(subject_rgb, subject_alpha)


def load_depth_map_f32(path: str) -> Tuple[np.ndarray, str]:
    """
    Safely load ANY 8-bit, 16-bit (`I;16` / `uint16`), or 32-bit float depth image
    into a normalized float32 array in [0, 1] without Pillow `.convert('L')` clamping
    16-bit pixels (>=255) to solid pure white!
    Returns (depth_f32_2d, bit_depth_desc).
    """
    # Use np.fromfile + cv2.imdecode so Unicode/Chinese Windows file paths work reliably
    raw_bytes = np.fromfile(path, dtype=np.uint8)
    arr = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
    if arr is None:
        pil_im = Image.open(path)
        arr = np.asarray(pil_im)

    orig_dtype = arr.dtype
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            # Respect alpha if depth is stored as RGBA
            alpha_ch = arr[..., 3].astype(np.float32)
            max_a = 65535.0 if orig_dtype == np.uint16 else (255.0 if orig_dtype == np.uint8 else 1.0)
            alpha_f = np.clip(alpha_ch / max(1.0, max_a), 0.0, 1.0)
            gray = (0.299 * arr[..., 2] + 0.587 * arr[..., 1] + 0.114 * arr[..., 0]).astype(np.float32)
            gray = gray * (alpha_f > 0.05).astype(np.float32)
        else:
            # OpenCV decodes BGR
            gray = (0.299 * arr[..., 2] + 0.587 * arr[..., 1] + 0.114 * arr[..., 0]).astype(np.float32)
    else:
        gray = arr.astype(np.float32)

    if orig_dtype == np.uint16:
        bit_desc = "16-bit uint16"
        out = gray / 65535.0
    elif orig_dtype == np.uint8:
        bit_desc = "8-bit uint8"
        out = gray / 255.0
    elif orig_dtype in (np.int32, np.uint32):
        bit_desc = f"32-bit {orig_dtype}"
        g_max = float(gray.max())
        out = gray / (65535.0 if g_max <= 65535.0 and g_max > 255.0 else max(1.0, g_max))
    else:
        bit_desc = f"float ({orig_dtype})"
        g_min, g_max = float(gray.min()), float(gray.max())
        if g_max > 1.001 or g_min < -0.001:
            out = (gray - g_min) / max(1e-6, g_max - g_min)
        else:
            out = gray

    return np.clip(out, 0.0, 1.0).astype(np.float32), bit_desc


def align_and_refine_depth_map(
    raw_depth_gray: np.ndarray,
    subject_alpha: np.ndarray,
    use_ecc: bool = True,
    guide_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Precision depth-to-artwork fusion:
      1. Smart IoU-gated ECC Affine Alignment: only warps externally supplied
         depth maps when their silhouette actually diverges (IoU < 0.92), never
         drifting an already pixel-aligned AI depth map.
      2. Interior-only nearest-neighbor hole fill: repairs dark interior parts
         (horns/hair tips) strictly inside `subject_alpha`, NEVER creating an
         invisible outer dilation skirt outside the silhouette.
      3. RGB-Guided edge-snapping filter (`_guided_filter_gray`) guided by the
         artwork itself so depth jumps lock pixel-accurately onto real drawn
         contours (hands, sleeves, dress folds, wings, hair) with zero bleeding.
    Returns clean smooth height field H in [0, 1] (0 outside subject silhouette).
    """
    h, w = subject_alpha.shape[:2]
    d = cv2.resize(raw_depth_gray.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    d_lo, d_hi = float(d.min()), float(d.max())
    if d_hi > d_lo + 1e-5:
        d = (d - d_lo) / (d_hi - d_lo)
    else:
        return np.zeros((h, w), dtype=np.float32)

    valid_fg = subject_alpha > 0.12
    if not np.any(valid_fg):
        return np.zeros((h, w), dtype=np.float32)

    # 1. Smart IoU-gated ECC alignment (only for misaligned external depth maps)
    if use_ecc and float(np.mean(valid_fg)) < 0.96:
        fg_bin = subject_alpha > 0.25
        d_bin = d > 0.06
        inter = float(np.sum(fg_bin & d_bin))
        union = float(np.sum(fg_bin | d_bin))
        iou = inter / max(1.0, union)
        if iou < 0.92:
            scale = min(1.0, 320.0 / max(h, w))
            sw, sh = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
            tpl = cv2.GaussianBlur(
                cv2.resize(fg_bin.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA),
                (0, 0), sigmaX=2.0,
            )
            src_sil = cv2.GaussianBlur(
                cv2.resize(d_bin.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA),
                (0, 0), sigmaX=2.0,
            )
            warp = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-4)
            try:
                _, warp = cv2.findTransformECC(tpl, src_sil, warp, cv2.MOTION_AFFINE, criteria)
                warp_full = warp.copy()
                warp_full[0, 2] /= scale
                warp_full[1, 2] /= scale
                det = float(warp_full[0, 0] * warp_full[1, 1] - warp_full[0, 1] * warp_full[1, 0])
                if 0.82 <= det <= 1.22 and abs(warp_full[0, 2]) < w * 0.12 and abs(warp_full[1, 2]) < h * 0.12:
                    d_cand = cv2.warpAffine(
                        d, warp_full, (w, h),
                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
                    )
                    cand_bin = d_cand > 0.06
                    cand_iou = float(np.sum(fg_bin & cand_bin)) / max(1.0, float(np.sum(fg_bin | cand_bin)))
                    if cand_iou > iou:
                        d = d_cand
            except cv2.error:
                pass

    # 2. Repair interior holes ONLY inside subject_alpha (never dilate depth outside alpha!)
    has_depth = valid_fg & (d > 0.05)
    if np.any(has_depth):
        inv_mask = (~has_depth).astype(np.uint8)
        _, labels = cv2.distanceTransformWithLabels(inv_mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        valid_coords = np.argwhere(has_depth)
        idx = np.clip(labels - 1, 0, len(valid_coords) - 1)
        d_nearest = d[valid_coords[idx, 0], valid_coords[idx, 1]]
        # Keep extrapolated values everywhere temporarily solely so the guided/bilateral
        # filter doesn't bleed zero background depth inward across silhouette edges!
        d_for_filter = np.where(has_depth, d, d_nearest).astype(np.float32)
    else:
        d_for_filter = d.copy()

    # 3. RGB-Guided edge-snapping filter: locks blurry AI depth edges onto drawn contours,
    #    while preserving ultra-crisp edges already present in 16-bit/studio depth maps!
    if guide_rgb is not None and guide_rgb.shape[:2] == (h, w):
        guide_l = (
            0.299 * guide_rgb[..., 0] + 0.587 * guide_rgb[..., 1] + 0.114 * guide_rgb[..., 2]
        ).astype(np.float32)
        # Extrapolate guide_l outward cleanly so silhouette boundary stays razor sharp
        if np.any(valid_fg) and not np.all(valid_fg):
            inv_fg = (~valid_fg).astype(np.uint8)
            _, lbl_g = cv2.distanceTransformWithLabels(inv_fg, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            fg_coords = np.argwhere(valid_fg)
            g_idx = np.clip(lbl_g - 1, 0, len(fg_coords) - 1)
            guide_l = np.where(valid_fg, guide_l, guide_l[fg_coords[g_idx, 0], fg_coords[g_idx, 1]])
        rad = max(3, int(round(w * 0.009)))
        d_snapped = _guided_filter_gray(guide_l, d_for_filter, radius=rad, eps=0.0025)
        # Preserve crisp native depth boundaries (e.g., white-on-white fingers/lace in 16-bit depth!)
        dgx = cv2.Sobel(d_for_filter, cv2.CV_32F, 1, 0, ksize=3)
        dgy = cv2.Sobel(d_for_filter, cv2.CV_32F, 0, 1, ksize=3)
        d_grad = np.sqrt(dgx * dgx + dgy * dgy)
        preserve_sharp = np.clip((d_grad - 0.04) * 5.0, 0.0, 0.80)
        d_mixed = d_snapped * (1.0 - preserve_sharp) + d_for_filter * preserve_sharp
        d_smooth = cv2.bilateralFilter(d_mixed, d=5, sigmaColor=0.08, sigmaSpace=max(3.0, w * 0.007))
    else:
        d_smooth = cv2.bilateralFilter(d_for_filter, d=5, sigmaColor=0.08, sigmaSpace=max(3.0, w * 0.007))

    # 4. Clean foreground dynamic-range normalization (faithful to AI depth, zero pillow bloat!)
    fg_vals = d_smooth[valid_fg]
    p_lo, p_hi = float(np.percentile(fg_vals, 2.0)), float(np.percentile(fg_vals, 98.5))
    if p_hi > p_lo + 1e-4:
        d_norm = np.clip((d_smooth - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    else:
        d_norm = np.full((h, w), 0.5, dtype=np.float32)

    # Map strictly inside valid silhouette to [0.06, 1.0], and strictly 0.0 outside!
    d_out = np.where(valid_fg, 0.06 + 0.94 * d_norm, 0.0).astype(np.float32)
    return np.clip(d_out, 0.0, 1.0).astype(np.float32)
