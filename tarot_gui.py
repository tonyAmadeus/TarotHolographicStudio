#!/usr/bin/env python3
"""
典藏级鎏金塔罗牌 / 宝可梦 SAR 三层流麻 3D 闪卡生成器 — 可视化图形界面 (GUI)
=============================================================================
特性：
  - 所有风格、外层摄影棚背景、卡内后景模板、前景透明花框、蚀刻底纹、金属色调
    均以真实可视化缩略图 / 色卡呈现，点击即选并实时刷新右侧 3D 预览！
  - 右侧支持「0° 正视 / +17.5° 右倾 / -17.5° 左倾」实时 3D WYSIWYG 预览，
    以及「原图 / 深度图 / 最终循环 GIF 动图」一键切换。
  - 支持单图一键自动抠图 + 本地 GPU AI 单目深度估计（Depth-Anything-V2），
    也支持自定义三层原图与外部深度图（ECC 仿射对齐）。

依赖：
    pip install opencv-python-headless numpy pillow

启动方式：
    双击 tarot_gui.pyw 或运行 python tarot_gui.py（自动隐藏黑色命令终端）
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk

from card_templates import (
    BG_TEMPLATE_META,
    FG_TEMPLATE_META,
    STUDIO_BG_META,
    ensure_default_templates,
)
from depth_estimator import generate_depth_from_image, is_ai_depth_available
from render_sar_card import render_3layer_sar_gif, render_live_preview_frame
from render_tarot_holographic import (
    METAL_META,
    METAL_PALETTES,
    PATTERN_META,
    STYLE_META,
    RenderCancelledError,
    build_micro_etching_texture,
    render_tarot_gif,
)


# ---------------------------------------------------------------------------
# Option Definitions
# ---------------------------------------------------------------------------

RENDER_MODE_OPTIONS: List[Tuple[str, str]] = [
    ("三层景深 + 流麻 + 3D深度浮雕 (SAR 推荐)", "3layer_sar"),
    ("经典单层全息塔罗牌模式", "classic_tarot"),
]

BG_MODE_OPTIONS: List[Tuple[str, str]] = [
    ("自动识别抠图", "auto"),
    ("强制抠除纯色底", "remove"),
    ("保留原图背景", "keep"),
]

LINE_ETCH_OPTIONS: List[Tuple[str, str]] = [
    ("中等金属线蚀刻", "medium"),
    ("强烈浮雕金属线", "strong"),
    ("轻微线条流光", "subtle"),
    ("关闭线条蚀刻", "off"),
]

DEPTH_ENGINE_OPTIONS: List[Tuple[str, str]] = [
    ("自动优先 AI 单目深度 (Large/Small 自动)", "auto"),
    ("Depth-Anything-V2-Large (旗舰高精度 GPU)", "ai_large"),
    ("Depth-Anything-V2-Small (高速轻量 AI)", "ai"),
    ("极速本地算法 (离线免模型)", "fast"),
]

SUPERSAMPLE_OPTIONS: List[Tuple[str, int]] = [
    ("2x (推荐)", 2),
    ("3x (高精)", 3),
    ("1x (快速)", 1),
]

INTENSITY_CHOICES: List[Tuple[str, str]] = [
    ("克制", "subtle"),
    ("中等 (推荐)", "medium"),
    ("华丽", "lavish"),
]

STYLE_VISUALS: List[Tuple[str, str, Tuple[int, int, int], Tuple[int, int, int]]] = list(STYLE_META)
PATTERN_VISUALS: List[Tuple[str, str]] = list(PATTERN_META)
METAL_VISUALS: List[Tuple[str, str]] = list(METAL_META)


# ---------------------------------------------------------------------------
# Color Theme (Luxury Dark Velvet Studio + Champagne Gold)
# ---------------------------------------------------------------------------

BG_MAIN = "#111018"
BG_CARD = "#1A1825"
BG_SUB = "#15131F"
BG_INPUT = "#242133"
BG_PREVIEW = "#0D0C14"
BORDER_SUBTLE = "#312C45"
GOLD_PRIMARY = "#DFBF7C"
GOLD_HOVER = "#EDD196"
TEXT_PRIMARY = "#F1ECE2"
TEXT_MUTED = "#9E97AE"


# ---------------------------------------------------------------------------
# Procedural Thumbnail Builders for Visual Pickers
# ---------------------------------------------------------------------------

def _make_badge_thumb(w: int, h: int, title: str, c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> Image.Image:
    """Create a smooth diagonal gradient swatch card."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = np.clip((xx / max(1, w - 1)) * 0.55 + (yy / max(1, h - 1)) * 0.45, 0.0, 1.0)[..., None]
    arr = np.array(c1, dtype=np.float32) * (1.0 - t) + np.array(c2, dtype=np.float32) * t
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(im)
    draw.rectangle([2, 2, w - 3, h - 3], outline=(255, 245, 220), width=1)
    return im


def _make_metal_thumb(key: str, w: int = 84, h: int = 44) -> Image.Image:
    if key == "auto":
        return _make_badge_thumb(w, h, "Auto", (95, 86, 130), (225, 198, 132))
    m_base, m_bright, m_shadow = METAL_PALETTES[key]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    diag = (xx / max(1, w - 1)) * 0.72 + (yy / max(1, h - 1)) * 0.28
    # Simulate brushed metallic specular sheen band
    spec = np.exp(-((diag - 0.48) ** 2) / 0.045)[..., None]
    base_t = np.clip(diag * 0.9, 0.0, 1.0)[..., None]
    col = np.array(m_shadow) * (1.0 - base_t) + np.array(m_base) * base_t + np.array(m_bright) * spec * 0.65
    # Subtle brushed lines
    rng = np.random.default_rng(abs(hash(key)) % 9999)
    brush = rng.normal(0.0, 0.035, size=(h, 1, 1)).astype(np.float32)
    col = np.clip((col + brush) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(col, mode="RGB")


def _make_pattern_thumb(key: str, w: int = 84, h: int = 56) -> Image.Image:
    if key == "auto":
        return _make_badge_thumb(w, h, "Auto", (52, 48, 84), (118, 96, 152))
    etch, grat = build_micro_etching_texture(h * 2, w * 2, key)
    # Render high-contrast holographic preview tile
    yy, xx = np.mgrid[0:h * 2, 0:w * 2].astype(np.float32)
    phase = (xx / (w * 2)) * 4.5 + (yy / (h * 2)) * 2.5
    irid = np.stack([
        0.55 + 0.42 * np.cos(phase),
        0.52 + 0.40 * np.cos(phase + 2.0),
        0.60 + 0.40 * np.cos(phase + 4.0),
    ], axis=-1).astype(np.float32)
    base = np.array([0.07, 0.07, 0.12], dtype=np.float32)
    rgb = base + irid * (etch[..., None] * 0.78 + grat[..., None] * 0.14)
    im = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    return im.resize((w, h), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# Reusable Visual Thumbnail Grid Widget
# ---------------------------------------------------------------------------

class VisualSelectorGrid(tk.Frame):
    """Grid of clickable visual thumbnail tiles with gold active-selection outline."""

    def __init__(
        self,
        parent: tk.Widget,
        items: List[Tuple[str, str, Image.Image]],
        cols: int,
        initial_key: str,
        on_select: Callable[[str], None],
    ) -> None:
        super().__init__(parent, bg=BG_CARD)
        self.on_select = on_select
        self.selected_key = initial_key
        self._photos: List[ImageTk.PhotoImage] = []
        self._tiles: Dict[str, tk.Frame] = {}
        self._labels: Dict[str, tk.Label] = {}

        for c in range(cols):
            self.columnconfigure(c, weight=1)

        for idx, (key, title, pil_img) in enumerate(items):
            r, c = divmod(idx, cols)
            photo = ImageTk.PhotoImage(pil_img)
            self._photos.append(photo)

            tile = tk.Frame(
                self,
                bg=BG_SUB,
                highlightthickness=2,
                highlightbackground=GOLD_PRIMARY if key == initial_key else BORDER_SUBTLE,
                padx=3,
                pady=3,
                cursor="hand2",
            )
            tile.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")

            img_lbl = tk.Label(tile, image=photo, bg=BG_SUB, bd=0, cursor="hand2")
            img_lbl.pack(pady=(1, 2))
            txt_lbl = tk.Label(
                tile,
                text=title,
                bg=BG_SUB,
                fg=GOLD_PRIMARY if key == initial_key else TEXT_PRIMARY,
                font=("Microsoft YaHei UI", 8),
                wraplength=pil_img.width + 14,
                cursor="hand2",
            )
            txt_lbl.pack()

            for wdg in (tile, img_lbl, txt_lbl):
                wdg.bind("<Button-1>", lambda _e, k=key: self.select(k))

            self._tiles[key] = tile
            self._labels[key] = txt_lbl

    def select(self, key: str, trigger: bool = True) -> None:
        if key not in self._tiles:
            return
        old = self.selected_key
        if old in self._tiles:
            self._tiles[old].configure(highlightbackground=BORDER_SUBTLE, bg=BG_SUB)
            self._labels[old].configure(fg=TEXT_PRIMARY, bg=BG_SUB)
        self.selected_key = key
        self._tiles[key].configure(highlightbackground=GOLD_PRIMARY, bg="#262238")
        self._labels[key].configure(fg=GOLD_PRIMARY, bg="#262238")
        if trigger:
            self.on_select(key)


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class TarotHolographicApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("典藏级鎏金塔罗 / 宝可梦 SAR 三层流麻 3D 闪卡生成器")
        self.root.geometry("1220x840")
        self.root.minsize(1040, 740)
        self.root.configure(bg=BG_MAIN)

        self.template_paths = ensure_default_templates()

        # File & numeric state
        self.mode_var = tk.StringVar(value=RENDER_MODE_OPTIONS[0][0])
        self.input_var = tk.StringVar()
        self.bg_var = tk.StringVar()
        self.fg_var = tk.StringVar()
        self.depth_var = tk.StringVar()
        self.studio_file_var = tk.StringVar()
        self.output_var = tk.StringVar()

        self.bg_mode_var = tk.StringVar(value=BG_MODE_OPTIONS[0][0])
        self.line_etch_var = tk.StringVar(value=LINE_ETCH_OPTIONS[0][0])
        self.depth_engine_var = tk.StringVar(value=DEPTH_ENGINE_OPTIONS[0][0])
        self.supersample_var = tk.StringVar(value=SUPERSAMPLE_OPTIONS[0][0])
        self.intensity_key = "medium"

        # Visual-picker selected keys
        self.studio_bg_key = "auto"
        self.bg_tpl_key = "auto"
        self.fg_tpl_key = "auto"
        self.style_key = "auto"
        self.pattern_key = "auto"
        self.metal_key = "auto"

        self.quicksand_var = tk.DoubleVar(value=1.0)
        self.relief_var = tk.DoubleVar(value=1.0)
        self.dof_var = tk.BooleanVar(value=True)
        self.width_var = tk.StringVar(value="460")
        self.frames_var = tk.StringVar(value="48")

        self.status_var = tk.StringVar(value="提示：在「视觉模板库 / 风格与特效」点选任意缩略图即可实时查看右侧 3D 预览")
        self.preview_info_var = tk.StringVar(value="实时 3D 预览准备中...")

        self.output_manually_set = False
        self.is_rendering = False
        self.cancel_event = threading.Event()
        self.last_output_path: Optional[str] = None

        # Live WYSIWYG Preview state
        self.preview_mode = "live3d"   # live3d | source | depth | gif
        self.preview_angle = "right"   # front | right | left
        self._preview_debounce_id: Optional[str] = None
        self._preview_seq = 0
        self.live_photo: Optional[ImageTk.PhotoImage] = None
        self.source_photo: Optional[ImageTk.PhotoImage] = None
        self.depth_photo: Optional[ImageTk.PhotoImage] = None
        self.gif_frames: List[ImageTk.PhotoImage] = []
        self.gif_durations: List[int] = []
        self.gif_frame_idx = 0
        self.gif_after_id: Optional[str] = None

        self._setup_styles()
        self._build_ui()
        # Trigger an immediate initial live 3D preview so the user sees a rich card on launch!
        self.root.after(180, self._schedule_live_preview)

    def _setup_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=BG_MAIN)
        style.configure("Card.TFrame", background=BG_CARD)
        style.configure("TLabel", background=BG_CARD, foreground=TEXT_PRIMARY, font=("Microsoft YaHei UI", 9))
        style.configure("Muted.TLabel", background=BG_CARD, foreground=TEXT_MUTED, font=("Microsoft YaHei UI", 8))
        style.configure("Header.TLabel", background=BG_MAIN, foreground=GOLD_PRIMARY, font=("Microsoft YaHei UI", 14, "bold"))
        style.configure("SubHeader.TLabel", background=BG_MAIN, foreground=TEXT_MUTED, font=("Microsoft YaHei UI", 9))
        style.configure("Section.TLabel", background=BG_CARD, foreground=GOLD_PRIMARY, font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("Dark.TCheckbutton", background=BG_CARD, foreground=TEXT_PRIMARY, font=("Microsoft YaHei UI", 9))
        style.map("Dark.TCheckbutton", background=[("active", BG_CARD)], foreground=[("active", GOLD_PRIMARY)])

        style.configure("Dark.TNotebook", background=BG_CARD, borderwidth=0, tabmargins=[2, 4, 2, 0])
        style.configure(
            "Dark.TNotebook.Tab",
            background=BG_INPUT,
            foreground=TEXT_MUTED,
            padding=[14, 6],
            font=("Microsoft YaHei UI", 9, "bold"),
            borderwidth=0,
        )
        style.map(
            "Dark.TNotebook.Tab",
            background=[("selected", GOLD_PRIMARY)],
            foreground=[("selected", "#15120D")],
        )

        style.configure(
            "Dark.TCombobox",
            fieldbackground=BG_INPUT,
            background=BG_INPUT,
            foreground=TEXT_PRIMARY,
            arrowcolor=GOLD_PRIMARY,
            bordercolor=BORDER_SUBTLE,
            lightcolor=BORDER_SUBTLE,
            darkcolor=BORDER_SUBTLE,
            padding=3,
        )
        style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", BG_INPUT)],
            foreground=[("readonly", TEXT_PRIMARY)],
            selectbackground=[("readonly", BG_INPUT)],
            selectforeground=[("readonly", TEXT_PRIMARY)],
        )

        style.configure(
            "Gold.Horizontal.TProgressbar",
            troughcolor=BG_INPUT,
            background=GOLD_PRIMARY,
            bordercolor=BORDER_SUBTLE,
            lightcolor=GOLD_PRIMARY,
            darkcolor=GOLD_PRIMARY,
            thickness=10,
        )

    # ------------------------------------------------------------------
    # Scrollable Frame Helper for Visual Template Gallery Tabs
    # ------------------------------------------------------------------

    def _make_scrollable_tab(self, notebook: ttk.Notebook, title: str) -> tk.Frame:
        outer = tk.Frame(notebook, bg=BG_CARD)
        notebook.add(outer, text=title)

        canvas = tk.Canvas(outer, bg=BG_CARD, highlightthickness=0, bd=0)
        vbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG_CARD, padx=10, pady=8)

        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")

        def _on_cfg(_e=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_resize(e) -> None:
            canvas.itemconfigure(win_id, width=e.width)

        inner.bind("<Configure>", _on_cfg)
        canvas.bind("<Configure>", _on_canvas_resize)

        # Smooth mousewheel scroll when hovering over this tab
        def _on_wheel(e) -> None:
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _on_wheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))
        return inner

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg=BG_MAIN, padx=18, pady=10)
        header.pack(fill="x")
        ttk.Label(header, text="✦ 典藏级鎏金塔罗 / 宝可梦 SAR 三层流麻 3D 闪卡生成器", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="可视化模板图库 · 非纯黑摄影棚背景 · 实时 3D 角度预览 · AI 单目深度图 + 16步 Relief Mapping + 300颗 3D 法线流麻亮片",
            style="SubHeader.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        body = tk.Frame(self.root, bg=BG_MAIN, padx=18, pady=2)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=6, minsize=620)
        body.columnconfigure(1, weight=5, minsize=400)
        body.rowconfigure(0, weight=1)

        # ---------------- Left Column: Tabbed Controls + Persistent Action Footer ----------------
        left_wrap = tk.Frame(body, bg=BG_CARD, highlightbackground=BORDER_SUBTLE, highlightthickness=1)
        left_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=(0, 14))
        left_wrap.rowconfigure(0, weight=1)
        left_wrap.columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(left_wrap, style="Dark.TNotebook")
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 0))

        tab_files = self._make_scrollable_tab(self.notebook, "  1. 素材与导出  ")
        tab_templates = self._make_scrollable_tab(self.notebook, "  2. 视觉模板库 (图选)  ")
        tab_style = self._make_scrollable_tab(self.notebook, "  3. 质感与 3D 参数  ")

        self._build_tab_files(tab_files)
        self._build_tab_templates(tab_templates)
        self._build_tab_style(tab_style)

        # Persistent Bottom Render Action Bar
        footer = tk.Frame(left_wrap, bg=BG_CARD, padx=14, pady=10)
        footer.grid(row=1, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        tk.Frame(footer, bg=BORDER_SUBTLE, height=1).grid(row=0, column=0, sticky="ew", pady=(0, 10))

        btn_row = tk.Frame(footer, bg=BG_CARD)
        btn_row.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        btn_row.columnconfigure(0, weight=4)
        btn_row.columnconfigure(1, weight=1)
        btn_row.columnconfigure(2, weight=2)
        btn_row.columnconfigure(3, weight=2)

        self.btn_render = self._make_button(btn_row, "✦ 开始生成 48 帧 3D 全息闪卡 GIF", self.start_render, secondary=False)
        self.btn_render.grid(row=0, column=0, sticky="ew", padx=(0, 6), ipady=5)

        self.btn_cancel = self._make_button(btn_row, "取消", self.cancel_render, secondary=True, danger=True)
        self.btn_cancel.grid(row=0, column=1, sticky="ew", padx=(0, 6), ipady=5)
        self.btn_cancel.configure(state="disabled")

        self.btn_open_dir = self._make_button(btn_row, "打开输出", self.open_output_folder, secondary=True)
        self.btn_open_dir.grid(row=0, column=2, sticky="ew", padx=(0, 6), ipady=5)
        self.btn_open_dir.configure(state="disabled")

        self.btn_open_tpl = self._make_button(btn_row, "模板目录", self.open_templates_folder, secondary=True)
        self.btn_open_tpl.grid(row=0, column=3, sticky="ew", ipady=5)

        self.progress = ttk.Progressbar(footer, style="Gold.Horizontal.TProgressbar", mode="determinate", maximum=100, value=0)
        self.progress.grid(row=2, column=0, sticky="ew", pady=(0, 5))

        ttk.Label(footer, textvariable=self.status_var, style="Muted.TLabel", wraplength=590).grid(row=3, column=0, sticky="w")

        # ---------------- Right Column: Live WYSIWYG 3D Preview Stage ----------------
        right_panel = tk.Frame(body, bg=BG_CARD, highlightbackground=BORDER_SUBTLE, highlightthickness=1, padx=14, pady=12)
        right_panel.grid(row=0, column=1, sticky="nsew", pady=(0, 14))
        right_panel.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)

        # Preview Toolbar (Angle & View Switcher)
        bar = tk.Frame(right_panel, bg=BG_CARD)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(bar, text="实时 3D 预览", style="Section.TLabel").pack(side="left", padx=(0, 8))

        self.angle_btns: Dict[str, tk.Button] = {}
        for key, txt in [("front", "0° 正视"), ("right", "+17.5° 右倾"), ("left", "-17.5° 左倾")]:
            b = self._make_button(bar, txt, lambda k=key: self._set_preview_angle(k), secondary=True)
            b.pack(side="left", padx=2)
            self.angle_btns[key] = b
        self._highlight_angle_btn("right")

        self.btn_view_gif = self._make_button(bar, "动图", lambda: self._switch_view("gif"), secondary=True)
        self.btn_view_gif.pack(side="right", padx=(2, 0))
        self.btn_view_gif.configure(state="disabled")

        self.btn_view_depth = self._make_button(bar, "深度", lambda: self._switch_view("depth"), secondary=True)
        self.btn_view_depth.pack(side="right", padx=2)
        self.btn_view_depth.configure(state="disabled")

        self.btn_view_src = self._make_button(bar, "原图", lambda: self._switch_view("source"), secondary=True)
        self.btn_view_src.pack(side="right", padx=2)
        self.btn_view_src.configure(state="disabled")

        self.preview_box = tk.Frame(right_panel, bg=BG_PREVIEW, highlightbackground=BORDER_SUBTLE, highlightthickness=1)
        self.preview_box.grid(row=1, column=0, sticky="nsew")
        self.preview_box.rowconfigure(0, weight=1)
        self.preview_box.columnconfigure(0, weight=1)

        self.preview_label = tk.Label(
            self.preview_box,
            text="正在生成实时 3D 效果预览...",
            bg=BG_PREVIEW,
            fg=TEXT_MUTED,
            font=("Microsoft YaHei UI", 9),
            justify="center",
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        ttk.Label(right_panel, textvariable=self.preview_info_var, style="Muted.TLabel", wraplength=410).grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )

    def _build_tab_files(self, parent: tk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(parent, text="渲染模式", style="Section.TLabel").grid(row=row, column=0, columnspan=4, sticky="w", pady=(2, 6))
        row += 1
        cb_mode = ttk.Combobox(
            parent, textvariable=self.mode_var, values=[x[0] for x in RENDER_MODE_OPTIONS],
            state="readonly", style="Dark.TCombobox",
        )
        cb_mode.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(0, 10))
        cb_mode.bind("<<ComboboxSelected>>", lambda _e: self._schedule_live_preview())
        row += 1

        ttk.Label(parent, text="图层素材（仅「中景主体」必填，其余留空则自动使用模板与 AI 深度）", style="Section.TLabel").grid(
            row=row, column=0, columnspan=4, sticky="w", pady=(2, 6)
        )
        row += 1

        row = self._add_file_row(parent, row, "中景主体 *", self.input_var, self.browse_input, None, "选择图片")
        row = self._add_file_row(parent, row, "卡内后景 (选)", self.bg_var, self.browse_bg, lambda: self._clear_and_preview(self.bg_var), "选择")
        row = self._add_file_row(parent, row, "前景花框 (选)", self.fg_var, self.browse_fg, lambda: self._clear_and_preview(self.fg_var), "选择")
        row = self._add_file_row(parent, row, "外摄影棚 (选)", self.studio_file_var, self.browse_studio_file, lambda: self._clear_and_preview(self.studio_file_var), "选择")

        # Depth map row with One-Click Generate
        ttk.Label(parent, text="深度图 (选)").grid(row=row, column=0, sticky="w", pady=4, padx=(0, 8))
        tk.Entry(
            parent, textvariable=self.depth_var, bg=BG_INPUT, fg=TEXT_PRIMARY,
            insertbackground=GOLD_PRIMARY, relief="flat", font=("Microsoft YaHei UI", 8),
            highlightbackground=BORDER_SUBTLE, highlightthickness=1,
        ).grid(row=row, column=1, sticky="ew", pady=4, ipady=4)
        self.btn_gen_depth = self._make_button(parent, "一键生成", self.generate_depth_now, secondary=False, width=7)
        self.btn_gen_depth.grid(row=row, column=2, padx=(6, 0), pady=4)
        self._make_button(parent, "选择", self.browse_depth, secondary=True, width=5).grid(row=row, column=3, padx=(4, 0), pady=4)
        row += 1

        row = self._add_file_row(parent, row, "输出 GIF", self.output_var, self.browse_output, None, "另存为")

        tk.Frame(parent, bg=BORDER_SUBTLE, height=1).grid(row=row, column=0, columnspan=4, sticky="ew", pady=12)
        row += 1

        ttk.Label(parent, text="导出规格与预处理", style="Section.TLabel").grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 6))
        row += 1

        g = tk.Frame(parent, bg=BG_CARD)
        g.grid(row=row, column=0, columnspan=4, sticky="ew")
        g.columnconfigure(1, weight=1)
        g.columnconfigure(3, weight=1)

        ttk.Label(g, text="背景抠图").grid(row=0, column=0, sticky="w", pady=4, padx=(0, 6))
        cb_bgm = ttk.Combobox(g, textvariable=self.bg_mode_var, values=[x[0] for x in BG_MODE_OPTIONS], state="readonly", style="Dark.TCombobox")
        cb_bgm.grid(row=0, column=1, sticky="ew", pady=4, padx=(0, 12))
        cb_bgm.bind("<<ComboboxSelected>>", lambda _e: self._schedule_live_preview())

        ttk.Label(g, text="线条蚀刻").grid(row=0, column=2, sticky="w", pady=4, padx=(0, 6))
        cb_le = ttk.Combobox(g, textvariable=self.line_etch_var, values=[x[0] for x in LINE_ETCH_OPTIONS], state="readonly", style="Dark.TCombobox")
        cb_le.grid(row=0, column=3, sticky="ew", pady=4)
        cb_le.bind("<<ComboboxSelected>>", lambda _e: self._schedule_live_preview())

        ttk.Label(g, text="深度引擎").grid(row=1, column=0, sticky="w", pady=4, padx=(0, 6))
        ttk.Combobox(g, textvariable=self.depth_engine_var, values=[x[0] for x in DEPTH_ENGINE_OPTIONS], state="readonly", style="Dark.TCombobox").grid(
            row=1, column=1, columnspan=3, sticky="ew", pady=4
        )
        row += 1

        exp = tk.Frame(parent, bg=BG_CARD)
        exp.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(10, 4))
        exp.columnconfigure(1, weight=1)
        exp.columnconfigure(3, weight=1)
        exp.columnconfigure(5, weight=1)

        ttk.Label(exp, text="宽度(px)").grid(row=0, column=0, sticky="w", padx=(0, 4))
        tk.Spinbox(
            exp, from_=240, to=960, increment=20, textvariable=self.width_var,
            bg=BG_INPUT, fg=TEXT_PRIMARY, buttonbackground=BG_INPUT, relief="flat",
            insertbackground=GOLD_PRIMARY, font=("Microsoft YaHei UI", 9),
            highlightbackground=BORDER_SUBTLE, highlightthickness=1, width=6,
        ).grid(row=0, column=1, sticky="ew", ipady=2, padx=(0, 10))

        ttk.Label(exp, text="帧数").grid(row=0, column=2, sticky="w", padx=(0, 4))
        tk.Spinbox(
            exp, from_=24, to=72, increment=4, textvariable=self.frames_var,
            bg=BG_INPUT, fg=TEXT_PRIMARY, buttonbackground=BG_INPUT, relief="flat",
            insertbackground=GOLD_PRIMARY, font=("Microsoft YaHei UI", 9),
            highlightbackground=BORDER_SUBTLE, highlightthickness=1, width=5,
        ).grid(row=0, column=3, sticky="ew", ipady=2, padx=(0, 10))

        ttk.Label(exp, text="超采样").grid(row=0, column=4, sticky="w", padx=(0, 4))
        ttk.Combobox(
            exp, textvariable=self.supersample_var, values=[x[0] for x in SUPERSAMPLE_OPTIONS],
            state="readonly", style="Dark.TCombobox", width=9,
        ).grid(row=0, column=5, sticky="ew", padx=(0, 10))

        cb_dof = ttk.Checkbutton(exp, text="景深 DoF", variable=self.dof_var, style="Dark.TCheckbutton", command=self._schedule_live_preview)
        cb_dof.grid(row=0, column=6, sticky="w")

    def _build_tab_templates(self, parent: tk.Frame) -> None:
        # 1. Outer Studio Backdrop Gallery (Non-Black Atmospheric Backdrops!)
        ttk.Label(parent, text="A. 卡片后外层摄影棚背景（点击缩略图即切换并实时预览）", style="Section.TLabel").pack(anchor="w", pady=(2, 4))
        studio_items: List[Tuple[str, str, Image.Image]] = [
            ("auto", "自动匹配", _make_badge_thumb(64, 86, "Auto", (55, 62, 112), (165, 138, 95)))
        ]
        for key, label in STUDIO_BG_META:
            short = label.replace("棚", "").replace("梦境", "")[:6]
            im = Image.open(self.template_paths["studio"][key]).convert("RGB").resize((64, 86), Image.Resampling.LANCZOS)
            studio_items.append((key, short, im))
        studio_items.append(("obsidian_dark", "曜石暗黑", _make_badge_thumb(64, 86, "Dark", (14, 14, 22), (34, 32, 46))))
        self.grid_studio = VisualSelectorGrid(parent, studio_items, cols=5, initial_key=self.studio_bg_key, on_select=self._on_pick_studio)
        self.grid_studio.pack(fill="x", pady=(0, 10))

        # 2. Inner Card Background Templates (10 Community Styles)
        ttk.Label(parent, text="B. 卡内后景模板（10 套社区/塔罗/SAR 样式）", style="Section.TLabel").pack(anchor="w", pady=(4, 4))
        bg_items: List[Tuple[str, str, Image.Image]] = [
            ("auto", "自动匹配", _make_badge_thumb(64, 96, "Auto", (48, 44, 78), (140, 118, 86)))
        ]
        for key, label in BG_TEMPLATE_META:
            short = label.split("·")[0].strip() if "·" in label else label[:6]
            im = Image.open(self.template_paths["bg"][key]).convert("RGB").resize((64, 96), Image.Resampling.LANCZOS)
            bg_items.append((key, short, im))
        self.grid_bg = VisualSelectorGrid(parent, bg_items, cols=5, initial_key=self.bg_tpl_key, on_select=self._on_pick_bg_tpl)
        self.grid_bg.pack(fill="x", pady=(0, 10))

        # 3. Foreground Transparent Decorative Frames (11 Ornate Frames)
        ttk.Label(parent, text="C. 前景透明装饰花框（11 套双层烫金/冰银花框）", style="Section.TLabel").pack(anchor="w", pady=(4, 4))
        fg_items: List[Tuple[str, str, Image.Image]] = [
            ("auto", "自动匹配", _make_badge_thumb(64, 96, "Auto", (38, 36, 56), (176, 148, 92)))
        ]
        dark_swatch = Image.new("RGBA", (64, 96), (24, 22, 38, 255))
        for key, label in FG_TEMPLATE_META:
            short = label.split("·")[-1].split("(")[0].strip()[:6]
            fg_im = Image.open(self.template_paths["fg"][key]).convert("RGBA").resize((64, 96), Image.Resampling.LANCZOS)
            comp = Image.alpha_composite(dark_swatch, fg_im).convert("RGB")
            fg_items.append((key, short, comp))
        fg_items.append(("none", "无前景框", _make_badge_thumb(64, 96, "None", (26, 24, 36), (40, 38, 52))))
        self.grid_fg = VisualSelectorGrid(parent, fg_items, cols=5, initial_key=self.fg_tpl_key, on_select=self._on_pick_fg_tpl)
        self.grid_fg.pack(fill="x", pady=(0, 8))

    def _build_tab_style(self, parent: tk.Frame) -> None:
        # 1. Overall Style Mood Swatches
        ttk.Label(parent, text="整体气质配色（决定光晕、虹彩相位与自动模板配对）", style="Section.TLabel").pack(anchor="w", pady=(2, 4))
        style_items = [(k, lbl, _make_badge_thumb(78, 44, lbl, c1, c2)) for k, lbl, c1, c2 in STYLE_VISUALS]
        self.grid_style = VisualSelectorGrid(parent, style_items, cols=6, initial_key=self.style_key, on_select=self._on_pick_style)
        self.grid_style.pack(fill="x", pady=(0, 10))

        # 2. Holographic Micro-Etching Pattern Visual Tiles
        ttk.Label(parent, text="全息蚀刻底纹（真实纹样直观预览）", style="Section.TLabel").pack(anchor="w", pady=(2, 4))
        pat_items = [(k, lbl, _make_pattern_thumb(k, 88, 54)) for k, lbl in PATTERN_VISUALS]
        self.grid_pattern = VisualSelectorGrid(parent, pat_items, cols=5, initial_key=self.pattern_key, on_select=self._on_pick_pattern)
        self.grid_pattern.pack(fill="x", pady=(0, 10))

        # 3. Metallic Foil & Card Edge Gradient Swatches
        ttk.Label(parent, text="金属卡边与烫金色调（金属拉丝反光色卡）", style="Section.TLabel").pack(anchor="w", pady=(2, 4))
        metal_items = [(k, lbl, _make_metal_thumb(k, 88, 44)) for k, lbl in METAL_VISUALS]
        self.grid_metal = VisualSelectorGrid(parent, metal_items, cols=5, initial_key=self.metal_key, on_select=self._on_pick_metal)
        self.grid_metal.pack(fill="x", pady=(0, 12))

        # 4. Holographic Intensity Pill Selector
        ttk.Label(parent, text="全息与箔光强度", style="Section.TLabel").pack(anchor="w", pady=(2, 4))
        int_row = tk.Frame(parent, bg=BG_CARD)
        int_row.pack(fill="x", pady=(0, 12))
        self.intensity_btns: Dict[str, tk.Button] = {}
        for idx, (lbl, key) in enumerate(INTENSITY_CHOICES):
            int_row.columnconfigure(idx, weight=1)
            b = self._make_button(int_row, lbl, lambda k=key: self._on_pick_intensity(k), secondary=(key != self.intensity_key))
            b.grid(row=0, column=idx, sticky="ew", padx=3)
            self.intensity_btns[key] = b

        # 5. Interactive Sliders for Quicksand (流麻) & 3D Relief Depth
        sl = tk.Frame(parent, bg=BG_CARD)
        sl.pack(fill="x", pady=(2, 8))
        sl.columnconfigure(1, weight=1)

        ttk.Label(sl, text="背景流麻细闪强度").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=4)
        sc_qs = tk.Scale(
            sl, from_=0.0, to=1.6, resolution=0.1, orient="horizontal", variable=self.quicksand_var,
            bg=BG_CARD, fg=GOLD_PRIMARY, troughcolor=BG_INPUT, highlightthickness=0, bd=0,
        )
        sc_qs.grid(row=0, column=1, sticky="ew", pady=4)
        sc_qs.bind("<ButtonRelease-1>", lambda _e: self._schedule_live_preview())

        ttk.Label(sl, text="3D Relief 立体凸出度").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=4)
        sc_rel = tk.Scale(
            sl, from_=0.0, to=1.6, resolution=0.1, orient="horizontal", variable=self.relief_var,
            bg=BG_CARD, fg=GOLD_PRIMARY, troughcolor=BG_INPUT, highlightthickness=0, bd=0,
        )
        sc_rel.grid(row=1, column=1, sticky="ew", pady=4)
        sc_rel.bind("<ButtonRelease-1>", lambda _e: self._schedule_live_preview())

    def _add_file_row(self, parent, row: int, label: str, var: tk.StringVar, browse_cb, clear_cb, btn_text: str) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 8))
        tk.Entry(
            parent, textvariable=var, bg=BG_INPUT, fg=TEXT_PRIMARY,
            insertbackground=GOLD_PRIMARY, relief="flat", font=("Microsoft YaHei UI", 8),
            highlightbackground=BORDER_SUBTLE, highlightthickness=1,
        ).grid(row=row, column=1, sticky="ew", pady=4, ipady=4)
        self._make_button(parent, btn_text, browse_cb, secondary=True, width=7).grid(row=row, column=2, padx=(6, 0), pady=4)
        if clear_cb is not None:
            self._make_button(parent, "清空", clear_cb, secondary=True, width=5).grid(row=row, column=3, padx=(4, 0), pady=4)
        return row + 1

    def _make_button(self, parent, text: str, command, secondary: bool = False, danger: bool = False, width: Optional[int] = None) -> tk.Button:
        if danger:
            bg, fg, active_bg = "#351E24", "#F2A2A7", "#4A2730"
        elif secondary:
            bg, fg, active_bg = BG_INPUT, TEXT_PRIMARY, "#322D48"
        else:
            bg, fg, active_bg = GOLD_PRIMARY, "#15120C", GOLD_HOVER
        kwargs = {
            "text": text, "command": command, "bg": bg, "fg": fg,
            "activebackground": active_bg, "activeforeground": fg,
            "disabledforeground": "#676175", "relief": "flat", "bd": 0,
            "cursor": "hand2", "font": ("Microsoft YaHei UI", 9, "bold" if not secondary else "normal"),
            "padx": 8, "pady": 4,
        }
        if width is not None:
            kwargs["width"] = width
        return tk.Button(parent, **kwargs)

    # ------------------------------------------------------------------
    # Visual Picker Callbacks (Auto-Refresh Live 3D Preview!)
    # ------------------------------------------------------------------

    def _on_pick_studio(self, key: str) -> None:
        self.studio_bg_key = key
        self.studio_file_var.set("")
        self._schedule_live_preview()

    def _on_pick_bg_tpl(self, key: str) -> None:
        self.bg_tpl_key = key
        self.bg_var.set("")
        self._schedule_live_preview()

    def _on_pick_fg_tpl(self, key: str) -> None:
        self.fg_tpl_key = key
        self.fg_var.set("")
        self._schedule_live_preview()

    def _on_pick_style(self, key: str) -> None:
        self.style_key = key
        self._schedule_live_preview()

    def _on_pick_pattern(self, key: str) -> None:
        self.pattern_key = key
        self._schedule_live_preview()

    def _on_pick_metal(self, key: str) -> None:
        self.metal_key = key
        self._schedule_live_preview()

    def _on_pick_intensity(self, key: str) -> None:
        self.intensity_key = key
        for k, btn in self.intensity_btns.items():
            if k == key:
                btn.configure(bg=GOLD_PRIMARY, fg="#15120C", font=("Microsoft YaHei UI", 9, "bold"))
            else:
                btn.configure(bg=BG_INPUT, fg=TEXT_PRIMARY, font=("Microsoft YaHei UI", 9, "normal"))
        self._schedule_live_preview()

    def _clear_and_preview(self, var: tk.StringVar) -> None:
        var.set("")
        self._schedule_live_preview()

    # ------------------------------------------------------------------
    # File Browsers
    # ------------------------------------------------------------------

    def _pick_image(self, title: str) -> str:
        return filedialog.askopenfilename(
            title=title,
            filetypes=[("图片文件", "*.png *.webp *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
        )

    def browse_input(self) -> None:
        path = self._pick_image("选择中景主体 / 原图")
        if not path:
            return
        self.input_var.set(path)
        if not self.output_manually_set or not self.output_var.get().strip():
            self.output_var.set(f"{os.path.splitext(path)[0]}_holographic_card.gif")
        self._cache_source_photo(path)
        self.btn_view_src.configure(state="normal")
        self.status_var.set("已加载主体原图，正在刷新右侧实时 3D 预览...")
        self._schedule_live_preview(delay_ms=40)

    def browse_bg(self) -> None:
        path = self._pick_image("选择自定义卡内后景图")
        if path:
            self.bg_var.set(path)
            self._schedule_live_preview()

    def browse_fg(self) -> None:
        path = self._pick_image("选择自定义前景透明装饰框 PNG")
        if path:
            self.fg_var.set(path)
            self._schedule_live_preview()

    def browse_studio_file(self) -> None:
        path = self._pick_image("选择自定义外层摄影棚背景图")
        if path:
            self.studio_file_var.set(path)
            self._schedule_live_preview()

    def browse_depth(self) -> None:
        path = self._pick_image("选择单目灰度深度参考图 (白近黑远)")
        if path:
            self.depth_var.set(path)
            self._load_depth_preview(path)

    def browse_output(self) -> None:
        cur = self.output_var.get().strip()
        init_dir = os.path.dirname(cur) if cur else os.getcwd()
        init_file = os.path.basename(cur) if cur else "holographic_card.gif"
        path = filedialog.asksaveasfilename(
            title="选择 GIF 保存路径", initialdir=init_dir, initialfile=init_file,
            defaultextension=".gif", filetypes=[("GIF 动图", "*.gif")],
        )
        if path:
            if not path.lower().endswith(".gif"):
                path += ".gif"
            self.output_var.set(path)
            self.output_manually_set = True

    # ------------------------------------------------------------------
    # Live 3D WYSIWYG Preview Engine (Debounced Background Thread)
    # ------------------------------------------------------------------

    def _highlight_angle_btn(self, active: Optional[str]) -> None:
        for k, b in self.angle_btns.items():
            if k == active:
                b.configure(bg=GOLD_PRIMARY, fg="#15120C", font=("Microsoft YaHei UI", 8, "bold"))
            else:
                b.configure(bg=BG_INPUT, fg=TEXT_PRIMARY, font=("Microsoft YaHei UI", 8, "normal"))

    def _set_preview_angle(self, angle_key: str) -> None:
        self.preview_angle = angle_key
        self.preview_mode = "live3d"
        self._stop_gif_animation()
        self._highlight_angle_btn(angle_key)
        self._schedule_live_preview(delay_ms=20)

    def _schedule_live_preview(self, delay_ms: int = 180) -> None:
        if self.is_rendering:
            return
        if self._preview_debounce_id is not None:
            try:
                self.root.after_cancel(self._preview_debounce_id)
            except tk.TclError:
                pass
        self._preview_debounce_id = self.root.after(delay_ms, self._run_live_preview_async)

    def _run_live_preview_async(self) -> None:
        self._preview_debounce_id = None
        if self.is_rendering:
            return
        self._preview_seq += 1
        seq = self._preview_seq
        self.preview_mode = "live3d"
        self._stop_gif_animation()
        self._highlight_angle_btn(self.preview_angle)

        subj = self.input_var.get().strip() or None
        kwargs = {
            "subject_path": subj if (subj and os.path.isfile(subj)) else None,
            "angle_preset": self.preview_angle,
            "preview_width": 320,
            "bg_path": self.bg_var.get().strip() or None,
            "fg_path": self.fg_var.get().strip() or None,
            "depth_path": self.depth_var.get().strip() or None,
            "studio_bg_path": self.studio_file_var.get().strip() or None,
            "bg_template": self.bg_tpl_key,
            "fg_template": self.fg_tpl_key,
            "studio_bg": self.studio_bg_key,
            "style": self.style_key,
            "pattern": self.pattern_key,
            "metal": self.metal_key,
            "intensity": self.intensity_key,
            "bg_mode": str(self._lookup(BG_MODE_OPTIONS, self.bg_mode_var.get())),
            "line_etch": str(self._lookup(LINE_ETCH_OPTIONS, self.line_etch_var.get())),
            "quicksand_strength": float(self.quicksand_var.get()),
            "relief_strength": float(self.relief_var.get()),
            "enable_dof": bool(self.dof_var.get()),
        }
        self.preview_info_var.set("正在刷新实时 3D 预览...")

        def _worker() -> None:
            try:
                res = render_live_preview_frame(**kwargs)
                self.root.after(0, self._on_live_preview_ready, seq, res)
            except Exception as exc:
                self.root.after(0, lambda: self.preview_info_var.set(f"实时预览提示：{exc}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_live_preview_ready(self, seq: int, res: Dict[str, object]) -> None:
        if seq != self._preview_seq or self.preview_mode != "live3d":
            return
        im: Image.Image = res["preview_image"]  # type: ignore[assignment]
        max_w, max_h = self._get_preview_bounds()
        im_fit = im.copy()
        im_fit.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.live_photo = ImageTk.PhotoImage(im_fit)
        self.preview_label.configure(image=self.live_photo, text="")
        ang_txt = {"front": "0° 正视", "right": "+17.5° 右倾", "left": "-17.5° 左倾"}.get(self.preview_angle, "")
        self.preview_info_var.set(
            f"实时 3D 预览 ({ang_txt}) · 摄影棚:{res['studio_source']} · 后景:{res['bg_source']} · 前景:{res['fg_source']} · {res['style']}/{res['pattern']}/{res['metal']}"
        )

    # ------------------------------------------------------------------
    # One-Click Depth Map Generation & View Switching
    # ------------------------------------------------------------------

    def generate_depth_now(self) -> None:
        input_path = self.input_var.get().strip()
        if not input_path or not os.path.isfile(input_path):
            messagebox.showwarning("未选择主体图", "请先选择「中景主体 *」图片，再点击一键生成深度图。")
            return
        engine = str(self._lookup(DEPTH_ENGINE_OPTIONS, self.depth_engine_var.get()))
        bg_mode = str(self._lookup(BG_MODE_OPTIONS, self.bg_mode_var.get()))
        if engine == "ai":
            ok, msg = is_ai_depth_available()
            if not ok:
                messagebox.showinfo("AI 深度未就绪", f"{msg}\n\n可先选「极速本地算法」或运行 pip install transformers")
                return

        out_depth = f"{os.path.splitext(input_path)[0]}_depth.png"
        self.btn_gen_depth.configure(state="disabled")
        self.status_var.set(f"正在生成单目深度图 ({engine})...")

        def _worker() -> None:
            try:
                _, used = generate_depth_from_image(input_path, output_path=out_depth, engine=engine, bg_mode=bg_mode)
                self.root.after(0, self._on_depth_generated, out_depth, used)
            except Exception as exc:
                self.root.after(0, self._on_depth_error, str(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_depth_generated(self, depth_path: str, used_engine: str) -> None:
        self.btn_gen_depth.configure(state="normal")
        self.depth_var.set(depth_path)
        self._load_depth_preview(depth_path)
        self.status_var.set(f"单目深度图生成完成 ({used_engine})：{os.path.basename(depth_path)}（点击上方角度按钮可查看 3D 实时效果）")

    def _on_depth_error(self, err_msg: str) -> None:
        self.btn_gen_depth.configure(state="normal")
        self.status_var.set("深度图生成失败")
        messagebox.showerror("深度图生成失败", err_msg)

    def _get_preview_bounds(self) -> Tuple[int, int]:
        self.root.update_idletasks()
        bw = max(320, self.preview_box.winfo_width() - 24)
        bh = max(460, self.preview_box.winfo_height() - 24)
        return bw, bh

    def _cache_source_photo(self, path: str) -> None:
        try:
            im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            max_w, max_h = self._get_preview_bounds()
            im.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            self.source_photo = ImageTk.PhotoImage(im)
        except Exception as exc:
            messagebox.showerror("无法读取图片", str(exc))

    def _load_depth_preview(self, path: str) -> None:
        self._stop_gif_animation()
        try:
            im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            orig_w, orig_h = im.size
            max_w, max_h = self._get_preview_bounds()
            im.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            self.depth_photo = ImageTk.PhotoImage(im)
            self.btn_view_depth.configure(state="normal")
            self._switch_view("depth")
            self.preview_info_var.set(f"单目深度图：{os.path.basename(path)} ({orig_w}×{orig_h}，白近黑远)")
        except Exception as exc:
            messagebox.showerror("无法读取深度图", str(exc))

    def _load_gif_preview(self, gif_path: str) -> None:
        self._stop_gif_animation()
        self.gif_frames.clear()
        self.gif_durations.clear()
        try:
            im = Image.open(gif_path)
            max_w, max_h = self._get_preview_bounds()
            n_frames = getattr(im, "n_frames", 1)
            for idx in range(n_frames):
                im.seek(idx)
                frame_rgb = im.convert("RGB")
                if frame_rgb.width > max_w or frame_rgb.height > max_h:
                    frame_rgb.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                self.gif_frames.append(ImageTk.PhotoImage(frame_rgb))
                self.gif_durations.append(int(im.info.get("duration", 45)))
            if self.gif_frames:
                self.btn_view_gif.configure(state="normal")
                self._switch_view("gif")
        except Exception as exc:
            self.preview_info_var.set(f"GIF 预览加载失败：{exc}")

    def _stop_gif_animation(self) -> None:
        if self.gif_after_id is not None:
            try:
                self.root.after_cancel(self.gif_after_id)
            except tk.TclError:
                pass
            self.gif_after_id = None

    def _play_next_gif_frame(self) -> None:
        if self.preview_mode != "gif" or not self.gif_frames:
            return
        self.preview_label.configure(image=self.gif_frames[self.gif_frame_idx], text="")
        delay = max(20, self.gif_durations[self.gif_frame_idx])
        self.gif_frame_idx = (self.gif_frame_idx + 1) % len(self.gif_frames)
        self.gif_after_id = self.root.after(delay, self._play_next_gif_frame)

    def _switch_view(self, mode: str) -> None:
        self._stop_gif_animation()
        self._highlight_angle_btn(None)
        if mode == "source" and self.source_photo is not None:
            self.preview_mode = "source"
            self.preview_label.configure(image=self.source_photo, text="")
            self.preview_info_var.set("当前显示：主体原图")
        elif mode == "depth" and self.depth_photo is not None:
            self.preview_mode = "depth"
            self.preview_label.configure(image=self.depth_photo, text="")
            self.preview_info_var.set("当前显示：单目灰度深度图 (白近黑远)")
        elif mode == "gif" and self.gif_frames:
            self.preview_mode = "gif"
            self.gif_frame_idx = 0
            self._play_next_gif_frame()
            if self.last_output_path:
                self.preview_info_var.set(f"当前播放：48 帧 3D 全息循环动图 ({os.path.basename(self.last_output_path)})")

    # ------------------------------------------------------------------
    # Full 48-Frame GIF Render Pipeline
    # ------------------------------------------------------------------

    def _lookup(self, options: List[Tuple[str, object]], label: str):
        for k, v in options:
            if k == label:
                return v
        return options[0][1]

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.btn_render.configure(state="normal" if enabled else "disabled")
        self.btn_cancel.configure(state="disabled" if enabled else "normal")

    def start_render(self) -> None:
        if self.is_rendering:
            return
        input_path = self.input_var.get().strip()
        output_path = self.output_var.get().strip()
        if not input_path or not os.path.isfile(input_path):
            messagebox.showwarning("未选择图片", "请先在「1. 素材与导出」中选择一张有效的「中景主体 *」图片。")
            self.notebook.select(0)
            return
        if not output_path:
            output_path = f"{os.path.splitext(input_path)[0]}_holographic_card.gif"
            self.output_var.set(output_path)

        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        try:
            final_width = int(self.width_var.get())
            n_frames = int(self.frames_var.get())
            if not (160 <= final_width <= 1200):
                raise ValueError("输出宽度需在 160 ~ 1200 之间")
            if not (12 <= n_frames <= 120):
                raise ValueError("循环帧数需在 12 ~ 120 之间")
        except ValueError as exc:
            messagebox.showwarning("参数错误", str(exc))
            return

        render_mode = str(self._lookup(RENDER_MODE_OPTIONS, self.mode_var.get()))
        common = {
            "output_path": output_path,
            "studio_bg": self.studio_bg_key,
            "studio_bg_path": self.studio_file_var.get().strip() or None,
            "style": self.style_key,
            "pattern": self.pattern_key,
            "metal": self.metal_key,
            "intensity": self.intensity_key,
            "bg_mode": str(self._lookup(BG_MODE_OPTIONS, self.bg_mode_var.get())),
            "line_etch": str(self._lookup(LINE_ETCH_OPTIONS, self.line_etch_var.get())),
            "final_width": final_width,
            "n_frames": n_frames,
            "supersample": int(self._lookup(SUPERSAMPLE_OPTIONS, self.supersample_var.get())),
        }

        if render_mode == "3layer_sar":
            job = {
                "mode": "3layer_sar",
                "kwargs": {
                    "subject_path": input_path,
                    "bg_path": self.bg_var.get().strip() or None,
                    "fg_path": self.fg_var.get().strip() or None,
                    "depth_path": self.depth_var.get().strip() or None,
                    "bg_template": self.bg_tpl_key,
                    "fg_template": self.fg_tpl_key,
                    "depth_engine": str(self._lookup(DEPTH_ENGINE_OPTIONS, self.depth_engine_var.get())),
                    "quicksand_strength": float(self.quicksand_var.get()),
                    "relief_strength": float(self.relief_var.get()),
                    "enable_dof": bool(self.dof_var.get()),
                    **common,
                },
            }
        else:
            job = {
                "mode": "classic_tarot",
                "kwargs": {"input_path": input_path, **common},
            }

        self.is_rendering = True
        self.cancel_event.clear()
        self.progress.configure(value=0)
        self.status_var.set("正在初始化渲染管线...")
        self._set_controls_enabled(False)

        threading.Thread(target=self._render_worker, args=(job,), daemon=True).start()

    def cancel_render(self) -> None:
        if self.is_rendering:
            self.cancel_event.set()
            self.status_var.set("正在取消渲染，请稍候...")
            self.btn_cancel.configure(state="disabled")

    def _on_progress(self, step: int, total: int, msg: str) -> None:
        pct = (step / max(1, total)) * 100.0
        self.root.after(0, self._update_progress_ui, pct, msg)

    def _update_progress_ui(self, pct: float, msg: str) -> None:
        self.progress.configure(value=pct)
        self.status_var.set(msg)

    def _render_worker(self, job: Dict[str, object]) -> None:
        try:
            kwargs = dict(job["kwargs"])  # type: ignore[arg-type]
            kwargs["progress_callback"] = self._on_progress
            kwargs["cancel_event"] = self.cancel_event
            if job["mode"] == "3layer_sar":
                info = render_3layer_sar_gif(**kwargs)
            else:
                info = render_tarot_gif(**kwargs)
            self.root.after(0, self._on_render_success, info)
        except RenderCancelledError:
            self.root.after(0, self._on_render_cancelled)
        except Exception as exc:
            self.root.after(0, self._on_render_error, str(exc))

    def _on_render_success(self, info: Dict[str, object]) -> None:
        self.is_rendering = False
        self._set_controls_enabled(True)
        self.last_output_path = str(info["output_path"])
        self.btn_open_dir.configure(state="normal")
        self.progress.configure(value=100)

        w, h = info["output_size"]
        extra = ""
        if info.get("mode") == "3layer_sar":
            extra = f" · 摄影棚:{info.get('studio_source')} · 深度:{info.get('depth_source')}"
        self.status_var.set(
            f"完成！{w}×{h} · {info['frames']} 帧 ({info['total_duration_s']}s/{info['avg_fps']}fps) · "
            f"{info['file_size_kb']} KB · {info['style']}/{info['pattern']}/{info['metal']}{extra}"
        )
        self._load_gif_preview(self.last_output_path)

    def _on_render_cancelled(self) -> None:
        self.is_rendering = False
        self._set_controls_enabled(True)
        self.progress.configure(value=0)
        self.status_var.set("渲染已取消")

    def _on_render_error(self, err_msg: str) -> None:
        self.is_rendering = False
        self._set_controls_enabled(True)
        self.progress.configure(value=0)
        self.status_var.set("渲染失败，请查看弹窗提示")
        messagebox.showerror("渲染失败", f"生成 GIF 时发生错误：\n{err_msg}")

    def _reveal_path(self, path: str) -> None:
        abs_path = os.path.abspath(path)
        try:
            if sys.platform.startswith("win"):
                if os.path.isfile(abs_path):
                    subprocess.Popen(["explorer", "/select,", abs_path])
                elif os.path.isdir(abs_path):
                    os.startfile(abs_path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", abs_path])
            else:
                subprocess.Popen(["xdg-open", abs_path if os.path.isdir(abs_path) else os.path.dirname(abs_path)])
        except Exception as exc:
            messagebox.showerror("打开目录失败", str(exc))

    def open_output_folder(self) -> None:
        target = self.last_output_path or self.output_var.get().strip()
        if target:
            self._reveal_path(target)

    def open_templates_folder(self) -> None:
        tdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        ensure_default_templates(tdir)
        self._reveal_path(tdir)


def _relaunch_without_console() -> bool:
    """
    On Windows, transparently hand off from `python.exe` to `pythonw.exe`
    (detached with CREATE_NO_WINDOW) so no black console window stays open.
    Pass `--console` to keep console output for debugging.
    """
    if not sys.platform.startswith("win"):
        return False
    if "--console" in sys.argv or "--detached" in sys.argv:
        return False

    exe_name = os.path.basename(sys.executable).lower()
    if exe_name == "pythonw.exe":
        return False

    pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    if os.path.isfile(pythonw):
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
        subprocess.Popen(
            [pythonw, os.path.abspath(__file__), "--detached", *sys.argv[1:]],
            creationflags=creationflags,
            close_fds=True,
        )
        return True

    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32
        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            pids = (ctypes.c_uint32 * 4)()
            if kernel32.GetConsoleProcessList(pids, 4) <= 1:
                user32.ShowWindow(hwnd, 0)
    except Exception:
        pass
    return False


def main() -> None:
    if _relaunch_without_console():
        return

    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    TarotHolographicApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
