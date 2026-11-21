const tauriCore = window.__TAURI__?.core ?? null;
const tauriEvent = window.__TAURI__?.event ?? null;

async function invokeTauri(cmd, args = {}) {
  if (!tauriCore) return null;
  return await tauriCore.invoke(cmd, args);
}

const IMAGE_FILTERS = [
  { name: "图片文件", extensions: ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"] },
];

const GIF_FILTERS = [
  { name: "GIF 动图", extensions: ["gif"] },
];

const state = {
  backendUrl: "http://127.0.0.1:18765",
  templatesDir: "",
  defaultOutputDir: "",

  render_mode: "3layer_sar",
  subject_path: "",
  depth_path: "",
  studio_bg_path: "",
  bg_path: "",
  fg_path: "",
  output_path: "",

  studio_bg: "auto",
  bg_template: "auto",
  fg_template: "auto",
  style: "auto",
  pattern: "auto",
  metal: "auto",

  bg_mode: "auto",
  card_aspect: "auto",
  subject_scale: 1.0,
  subject_offset_x: 0.0,
  subject_offset_y: 0.0,
  depth_engine: "auto",
  intensity: "medium",
  line_etch: "medium",
  relief_strength: 1.0,
  quicksand_strength: 1.0,
  enable_dof: true,
  quality_tier: "hd",
  final_width: 480,
  n_frames: 48,
  supersample: 2,

  angle_preset: "right",
  preview_view: "live3d",
  last_live3d_data_url: "",
  last_live3d_meta: "",
  last_rendered_gif: "",
};

let previewTimer = null;
let previewSeq = 0;
let pollTimer = null;
let isRendering = false;
let isGeneratingDepth = false;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function fileBaseName(path) {
  if (!path) return "";
  return path.replace(/\\/g, "/").split("/").pop() || path;
}

function buildLocalFileUrl(filePath, extraQuery = "") {
  const suffix = extraQuery ? `&${extraQuery}` : "";
  return `${state.backendUrl}/api/file?path=${encodeURIComponent(filePath)}${suffix}&t=${Date.now()}`;
}

function setStatus(text, percent = null) {
  const label = $("#progress-label");
  if (label) label.textContent = text;
  if (percent !== null && Number.isFinite(percent)) {
    const clamped = Math.max(0, Math.min(100, percent));
    const bar = $("#render-progress");
    const pct = $("#progress-pct");
    if (bar) bar.value = clamped;
    if (pct) pct.textContent = `${Math.round(clamped)}%`;
  }
}

async function apiGet(path) {
  const res = await fetch(`${state.backendUrl}${path}`, { cache: "no-store" });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

async function apiPost(path, body) {
  const res = await fetch(`${state.backendUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

async function openFileDialog(title, filters = IMAGE_FILTERS) {
  if (tauriCore) {
    try {
      const selected = await invokeTauri("plugin:dialog|open", {
        options: {
          title,
          multiple: false,
          directory: false,
          filters,
        },
      });
      if (typeof selected === "string" && selected.length > 0) {
        return selected;
      }
      if (Array.isArray(selected) && selected.length > 0) {
        return selected[0];
      }
      return null;
    } catch (err) {
      console.error("Tauri open dialog failed:", err);
    }
  }
  return null;
}

async function saveFileDialog(title, defaultPath = "") {
  if (tauriCore) {
    try {
      const selected = await invokeTauri("plugin:dialog|save", {
        options: {
          title,
          defaultPath: defaultPath || undefined,
          filters: GIF_FILTERS,
        },
      });
      if (typeof selected === "string" && selected.length > 0) {
        return selected.toLowerCase().endsWith(".gif") ? selected : `${selected}.gif`;
      }
      return null;
    } catch (err) {
      console.error("Tauri save dialog failed:", err);
    }
  }
  return null;
}

async function uploadBrowserFile(file) {
  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("读取拖拽文件失败"));
    reader.readAsDataURL(file);
  });
  const res = await apiPost("/api/upload", {
    filename: file.name || "subject.png",
    data_base64: dataUrl,
  });
  return res.path;
}

function updateSubjectDisplay() {
  const titleEl = $("#subject-filename");
  const hintEl = $("#subject-path-hint");
  if (state.subject_path) {
    titleEl.textContent = fileBaseName(state.subject_path);
    hintEl.textContent = state.subject_path;
    if (!state.output_path) {
      const stem = state.subject_path.replace(/\.[^.\\/]+$/, "");
      $("#input-output-path").placeholder = `${stem}_holographic_card.gif`;
    }
  } else {
    titleEl.textContent = "点击选择或拖拽人物 / 塔罗主体图片到此处";
    hintEl.textContent = "支持 PNG / JPG / WebP · 留空时右侧自动展示默认 3D 示例卡";
    $("#input-output-path").placeholder = "默认保存至原图同目录 *_holographic_card.gif";
  }
}

function setSubjectPath(newPath) {
  state.subject_path = newPath || "";
  updateSubjectDisplay();
  if (state.preview_view === "subject") {
    refreshCurrentView();
  } else {
    schedulePreview(80);
  }
}

/* ---------------------------------------------------------------------------
 * Visual Thumbnail Gallery Renderer
 * -------------------------------------------------------------------------*/
function renderGallery(containerId, items, stateKey, tagId) {
  const container = $(containerId);
  const tagEl = $(tagId);
  if (!container) return;

  container.innerHTML = "";
  const labelMap = new Map();

  for (const item of items) {
    labelMap.set(item.key, item.label);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "thumb-card" + (state[stateKey] === item.key ? " is-selected" : "");
    btn.dataset.key = item.key;
    btn.title = item.label;

    const img = document.createElement("img");
    img.loading = "lazy";
    img.alt = item.label;
    img.src = item.thumb.startsWith("data:") ? item.thumb : `${state.backendUrl}${item.thumb}`;

    const span = document.createElement("span");
    span.textContent = item.label;

    btn.append(img, span);
    btn.addEventListener("click", () => {
      state[stateKey] = item.key;
      for (const el of container.querySelectorAll(".thumb-card")) {
        el.classList.toggle("is-selected", el.dataset.key === item.key);
      }
      if (tagEl) tagEl.textContent = item.label;
      if (state.preview_view !== "live3d") {
        setPreviewView("live3d", false);
      }
      schedulePreview(60);
    });

    container.appendChild(btn);
  }

  if (tagEl) {
    tagEl.textContent = labelMap.get(state[stateKey]) || state[stateKey];
  }
}

/* ---------------------------------------------------------------------------
 * Live 3D WYSIWYG Preview Stage
 * -------------------------------------------------------------------------*/
function showPreviewImage(src, emptyMessage = "") {
  const img = $("#preview-image");
  const empty = $("#preview-empty");
  if (src) {
    img.src = src;
    img.hidden = false;
    empty.hidden = true;
  } else {
    img.hidden = true;
    img.removeAttribute("src");
    empty.hidden = false;
    empty.querySelector("p").textContent = emptyMessage || "暂无对应图像";
  }
}

function setPreviewView(viewName, triggerLoad = true) {
  state.preview_view = viewName;
  for (const btn of $$(".seg-btn")) {
    btn.classList.toggle("is-active", btn.dataset.view === viewName);
  }
  const angleBar = $(".angle-bar");
  if (angleBar) {
    angleBar.style.opacity = viewName === "live3d" ? "1" : "0.45";
  }
  if (triggerLoad) {
    refreshCurrentView();
  }
}

function refreshCurrentView() {
  const spinner = $("#preview-spinner");
  const meta = $("#preview-meta");

  if (state.preview_view === "live3d") {
    requestLive3DPreview();
    return;
  }

  spinner.hidden = true;

  if (state.preview_view === "subject") {
    if (state.subject_path) {
      showPreviewImage(buildLocalFileUrl(state.subject_path));
      meta.textContent = `主体原图：${state.subject_path}`;
    } else {
      showPreviewImage("", "尚未选择中景主体图片（拖拽图片或点击“浏览图片”即可加载）");
      meta.textContent = "未选择主体原图";
    }
    return;
  }

  if (state.preview_view === "depth") {
    if (state.depth_path) {
      showPreviewImage(buildLocalFileUrl(state.depth_path, "as_depth=1"));
      meta.textContent = `单目深度图（支持 8/16/32 位，白近黑远）：${state.depth_path}`;
    } else {
      showPreviewImage("", "尚未生成或指定深度图，可在左侧点击“一键生成并预览深度图”");
      meta.textContent = "未指定深度图";
    }
    return;
  }

  if (state.preview_view === "gif") {
    if (state.last_rendered_gif) {
      showPreviewImage(buildLocalFileUrl(state.last_rendered_gif));
      meta.textContent = `导出动图预览：${state.last_rendered_gif}`;
    } else {
      showPreviewImage("", "尚未导出 GIF 动图，点击左下角“开始生成 3D 全息 GIF”后自动播放");
      meta.textContent = "尚未导出 GIF";
    }
  }
}

function schedulePreview(delayMs = 150) {
  if (previewTimer) clearTimeout(previewTimer);
  previewTimer = setTimeout(() => {
    if (state.preview_view === "live3d") {
      requestLive3DPreview();
    }
  }, delayMs);
}

async function requestLive3DPreview() {
  const mySeq = ++previewSeq;
  const spinner = $("#preview-spinner");
  const meta = $("#preview-meta");
  spinner.hidden = false;

  try {
    const payload = {
      subject_path: state.subject_path || null,
      angle_preset: state.angle_preset,
      preview_width: 360,
      depth_engine: state.depth_engine,
      bg_path: state.bg_path || null,
      fg_path: state.fg_path || null,
      depth_path: state.depth_path || null,
      studio_bg_path: state.studio_bg_path || null,
      bg_template: state.bg_template,
      fg_template: state.fg_template,
      studio_bg: state.studio_bg,
      style: state.style,
      pattern: state.pattern,
      metal: state.metal,
      intensity: state.intensity,
      bg_mode: state.bg_mode,
      card_aspect: state.card_aspect,
      subject_scale: state.subject_scale,
      subject_offset_x: state.subject_offset_x,
      subject_offset_y: state.subject_offset_y,
      line_etch: state.line_etch,
      quicksand_strength: state.quicksand_strength,
      relief_strength: state.relief_strength,
      enable_dof: state.enable_dof,
    };

    const res = await apiPost("/api/preview", payload);
    if (mySeq !== previewSeq) return;

    state.last_live3d_data_url = res.image_data_url;
    const szTag = Array.isArray(res.subject_size) && res.subject_size[0] > 0 && state.subject_path
      ? `原图:${res.subject_size[0]}×${res.subject_size[1]} | 缩放:${Math.round(state.subject_scale * 100)}% | `
      : "";
    state.last_live3d_meta = `${szTag}卡外:${res.studio_source} | 后景:${res.bg_source} | 前景:${res.fg_source} | 风格:${res.style} / ${res.pattern} / ${res.metal}`;
    if (state.preview_view === "live3d") {
      showPreviewImage(res.image_data_url);
      meta.textContent = state.last_live3d_meta;
    }
  } catch (err) {
    if (mySeq !== previewSeq) return;
    meta.textContent = `预览刷新失败：${err.message}`;
  } finally {
    if (mySeq === previewSeq) {
      spinner.hidden = true;
    }
  }
}

/* ---------------------------------------------------------------------------
 * AI / Fast Depth Map Generation
 * -------------------------------------------------------------------------*/
async function handleGenerateDepth() {
  if (isGeneratingDepth) return;
  if (!state.subject_path) {
    setStatus("请先选择或拖拽一张中景主体图片，再生成单目深度图。");
    return;
  }

  isGeneratingDepth = true;
  const btn = $("#btn-gen-depth");
  btn.disabled = true;
  btn.textContent = "深度计算中…";
  setStatus("正在运行单目深度估计模型（本地 GPU/离线加速）…", 35);

  try {
    const res = await apiPost("/api/depth", {
      subject_path: state.subject_path,
      depth_engine: state.depth_engine,
      bg_mode: state.bg_mode,
    });
    state.depth_path = res.depth_path;
    $("#input-depth-path").value = res.depth_path;
    setStatus(`深度图已生成（引擎：${res.engine}）→ ${fileBaseName(res.depth_path)}`, 100);
    setPreviewView("depth", true);
  } catch (err) {
    setStatus(`深度图生成失败：${err.message}`, 0);
  } finally {
    isGeneratingDepth = false;
    btn.disabled = false;
    btn.textContent = "一键生成并预览深度图";
  }
}

/* ---------------------------------------------------------------------------
 * Full 48-Frame GIF Render Pipeline
 * -------------------------------------------------------------------------*/
function setRenderButtons(running) {
  isRendering = running;
  $("#btn-render").disabled = running;
  $("#btn-cancel").disabled = !running;
  $("#btn-reveal-output").disabled = !state.last_rendered_gif;
}

async function handleStartRender() {
  if (isRendering) return;
  if (!state.subject_path) {
    setStatus("请先在「1. 素材与导出」中选择或拖拽一张中景主体图片。");
    return;
  }

  setRenderButtons(true);
  setStatus("正在提交 3D 渲染任务…", 2);

  try {
    const payload = {
      render_mode: state.render_mode,
      subject_path: state.subject_path,
      bg_path: state.bg_path || null,
      fg_path: state.fg_path || null,
      depth_path: state.depth_path || null,
      studio_bg_path: state.studio_bg_path || null,
      output_path: state.output_path || "",
      studio_bg: state.studio_bg,
      bg_template: state.bg_template,
      fg_template: state.fg_template,
      depth_engine: state.depth_engine,
      style: state.style,
      pattern: state.pattern,
      metal: state.metal,
      intensity: state.intensity,
      bg_mode: state.bg_mode,
      card_aspect: state.card_aspect,
      subject_scale: state.subject_scale,
      subject_offset_x: state.subject_offset_x,
      subject_offset_y: state.subject_offset_y,
      line_etch: state.line_etch,
      quicksand_strength: state.quicksand_strength,
      relief_strength: state.relief_strength,
      enable_dof: state.enable_dof,
      quality_tier: state.quality_tier,
      final_width: state.final_width,
      n_frames: state.n_frames,
      supersample: state.supersample,
    };

    const started = await apiPost("/api/render/start", payload);
    if (started.output_path && !state.output_path) {
      $("#input-output-path").placeholder = started.output_path;
    }

    startPollingRenderStatus();
  } catch (err) {
    setRenderButtons(false);
    setStatus(`启动渲染失败：${err.message}`, 0);
  }
}

function startPollingRenderStatus() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const st = await apiGet("/api/render/status");
      setStatus(st.message || "渲染中…", st.percent ?? 0);

      if (!st.running) {
        clearInterval(pollTimer);
        pollTimer = null;

        if (st.error) {
          setRenderButtons(false);
          setStatus(`渲染失败：${st.error}`, 0);
        } else if (st.result && (st.result.output_path || st.result.output)) {
          const outFile = st.result.output_path || st.result.output;
          const kbVal = st.result.file_size_kb ?? st.result.size_kb;
          state.last_rendered_gif = outFile;
          setRenderButtons(false);
          const kb = kbVal ? ` (${Number(kbVal).toFixed(0)} KB)` : "";
          setStatus(`渲染完成${kb}：${fileBaseName(outFile)}`, 100);
          setPreviewView("gif", true);
        } else {
          setRenderButtons(false);
        }
      }
    } catch (err) {
      clearInterval(pollTimer);
      pollTimer = null;
      setRenderButtons(false);
      setStatus(`状态轮询中断：${err.message}`);
    }
  }, 350);
}

async function handleCancelRender() {
  if (!isRendering) return;
  try {
    await apiPost("/api/render/cancel", {});
    setStatus("正在中止渲染任务…");
  } catch (err) {
    console.error(err);
  }
}

/* ---------------------------------------------------------------------------
 * Event Wiring & Native Tauri Drag-Drop
 * -------------------------------------------------------------------------*/
function bindUIEvents() {
  // Tabs
  for (const btn of $$(".tab-btn")) {
    btn.addEventListener("click", () => {
      const targetId = btn.dataset.tab;
      for (const b of $$(".tab-btn")) {
        const active = b === btn;
        b.classList.toggle("is-active", active);
        b.setAttribute("aria-selected", String(active));
      }
      for (const panel of $$(".tab-panel")) {
        panel.hidden = panel.id !== targetId;
        panel.classList.toggle("is-active", panel.id === targetId);
      }
    });
  }

  // Render mode radios
  for (const radio of $$('input[name="render_mode"]')) {
    radio.addEventListener("change", () => {
      if (radio.checked) {
        state.render_mode = radio.value;
      }
    });
  }

  // Subject pick / clear
  const triggerSubjectPick = async () => {
    const picked = await openFileDialog("选择中景主体图片");
    if (picked) {
      setSubjectPath(picked);
    } else if (!tauriCore) {
      $("#fallback-file-subject").click();
    }
  };

  $("#btn-pick-subject").addEventListener("click", (e) => {
    e.stopPropagation();
    triggerSubjectPick();
  });

  $("#dropzone-subject").addEventListener("click", (e) => {
    if (e.target.closest("button")) return;
    triggerSubjectPick();
  });

  $("#dropzone-subject").addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      triggerSubjectPick();
    }
  });

  $("#fallback-file-subject").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      setStatus("正在导入图片…");
      const uploaded = await uploadBrowserFile(file);
      setSubjectPath(uploaded);
      setStatus(`已载入主体图片：${file.name}`);
    } catch (err) {
      setStatus(`导入失败：${err.message}`);
    }
  });

  $("#btn-clear-subject").addEventListener("click", () => {
    setSubjectPath("");
  });

  // Depth pick / clear / generate
  const onDepthPathChanged = () => {
    if (state.preview_view === "depth") {
      refreshCurrentView();
    } else {
      schedulePreview(80);
    }
  };

  $("#btn-pick-depth").addEventListener("click", async () => {
    const picked = await openFileDialog("选择单目灰度深度图");
    if (picked) {
      state.depth_path = picked;
      $("#input-depth-path").value = picked;
      onDepthPathChanged();
    }
  });

  $("#btn-clear-depth").addEventListener("click", () => {
    state.depth_path = "";
    $("#input-depth-path").value = "";
    onDepthPathChanged();
  });

  $("#input-depth-path").addEventListener("change", (e) => {
    state.depth_path = e.target.value.trim();
    onDepthPathChanged();
  });

  $("#btn-gen-depth").addEventListener("click", handleGenerateDepth);

  // Optional custom layer paths
  const layerMap = {
    studio: { input: "#input-studio-path", key: "studio_bg_path", title: "选择卡外自定义摄影棚背景图" },
    bg: { input: "#input-bg-path", key: "bg_path", title: "选择卡内自定义后景图" },
    fg: { input: "#input-fg-path", key: "fg_path", title: "选择自定义透明前景装饰框 (RGBA PNG)" },
  };

  for (const btn of $$("[data-pick]")) {
    btn.addEventListener("click", async () => {
      const cfg = layerMap[btn.dataset.pick];
      if (!cfg) return;
      const picked = await openFileDialog(cfg.title);
      if (picked) {
        state[cfg.key] = picked;
        $(cfg.input).value = picked;
        schedulePreview(80);
      }
    });
  }

  for (const btn of $$("[data-clear]")) {
    btn.addEventListener("click", () => {
      const cfg = layerMap[btn.dataset.clear];
      if (!cfg) return;
      state[cfg.key] = "";
      $(cfg.input).value = "";
      schedulePreview(80);
    });
  }

  for (const cfg of Object.values(layerMap)) {
    $(cfg.input).addEventListener("change", (e) => {
      state[cfg.key] = e.target.value.trim();
      schedulePreview(80);
    });
  }

  // Output path
  $("#btn-pick-output").addEventListener("click", async () => {
    const def = state.output_path || (state.defaultOutputDir ? `${state.defaultOutputDir}\\holographic_card.gif` : "holographic_card.gif");
    const picked = await saveFileDialog("导出 3D 全息 GIF 动图", def);
    if (picked) {
      state.output_path = picked;
      $("#input-output-path").value = picked;
    }
  });

  $("#input-output-path").addEventListener("change", (e) => {
    state.output_path = e.target.value.trim();
  });

  // Selects & numeric inputs
  $("#select-bg-mode").addEventListener("change", (e) => {
    state.bg_mode = e.target.value;
    schedulePreview(80);
  });

  const syncTransformUI = () => {
    $("#range-subject-scale").value = String(state.subject_scale);
    $("#out-subject-scale").textContent = `${Math.round(state.subject_scale * 100)}%`;
    $("#range-offset-x").value = String(state.subject_offset_x);
    $("#out-offset-x").textContent = `${Math.round(state.subject_offset_x * 100)}%`;
    $("#range-offset-y").value = String(state.subject_offset_y);
    $("#out-offset-y").textContent = `${Math.round(state.subject_offset_y * 100)}%`;
    $("#select-card-aspect").value = state.card_aspect;
  };

  $("#select-card-aspect").addEventListener("change", (e) => {
    state.card_aspect = e.target.value;
    if (state.preview_view !== "live3d") setPreviewView("live3d", false);
    schedulePreview(80);
  });

  $("#range-subject-scale").addEventListener("input", (e) => {
    state.subject_scale = parseFloat(e.target.value);
    $("#out-subject-scale").textContent = `${Math.round(state.subject_scale * 100)}%`;
    if (state.preview_view !== "live3d") setPreviewView("live3d", false);
    schedulePreview(130);
  });

  $("#range-offset-x").addEventListener("input", (e) => {
    state.subject_offset_x = parseFloat(e.target.value);
    $("#out-offset-x").textContent = `${Math.round(state.subject_offset_x * 100)}%`;
    if (state.preview_view !== "live3d") setPreviewView("live3d", false);
    schedulePreview(130);
  });

  $("#range-offset-y").addEventListener("input", (e) => {
    state.subject_offset_y = parseFloat(e.target.value);
    $("#out-offset-y").textContent = `${Math.round(state.subject_offset_y * 100)}%`;
    if (state.preview_view !== "live3d") setPreviewView("live3d", false);
    schedulePreview(130);
  });

  for (const btn of $$("[data-scale]")) {
    btn.addEventListener("click", () => {
      state.subject_scale = parseFloat(btn.dataset.scale);
      syncTransformUI();
      if (state.preview_view !== "live3d") setPreviewView("live3d", false);
      schedulePreview(60);
    });
  }

  $("#btn-reset-transform").addEventListener("click", () => {
    state.subject_scale = 1.0;
    state.subject_offset_x = 0.0;
    state.subject_offset_y = 0.0;
    state.card_aspect = "auto";
    syncTransformUI();
    if (state.preview_view !== "live3d") setPreviewView("live3d", false);
    schedulePreview(60);
  });

  // Mouse-wheel zoom on live 3D preview stage
  $("#preview-stage").addEventListener("wheel", (e) => {
    if (state.preview_view !== "live3d" || !state.subject_path) return;
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.06 : -0.06;
    state.subject_scale = Math.max(0.4, Math.min(1.8, Math.round((state.subject_scale + delta) * 50) / 50));
    syncTransformUI();
    schedulePreview(140);
  }, { passive: false });

  $("#select-depth-engine").addEventListener("change", (e) => {
    state.depth_engine = e.target.value;
    schedulePreview(80);
  });

  // 3-tier GIF clarity presets
  const QUALITY_PRESETS = {
    standard: { width: 360, frames: 36, supersample: 2 },
    hd: { width: 480, frames: 48, supersample: 2 },
    ultra: { width: 680, frames: 48, supersample: 2 },
  };
  for (const radio of $$('input[name="quality_tier"]')) {
    radio.addEventListener("change", () => {
      if (!radio.checked) return;
      state.quality_tier = radio.value;
      const preset = QUALITY_PRESETS[radio.value];
      if (preset) {
        state.final_width = preset.width;
        state.n_frames = preset.frames;
        state.supersample = preset.supersample;
        $("#input-width").value = String(preset.width);
        $("#input-frames").value = String(preset.frames);
        $("#select-supersample").value = String(preset.supersample);
      }
    });
  }

  $("#input-width").addEventListener("change", (e) => {
    state.final_width = Math.max(280, Math.min(960, parseInt(e.target.value, 10) || 480));
    e.target.value = String(state.final_width);
  });

  $("#input-frames").addEventListener("change", (e) => {
    state.n_frames = Math.max(24, Math.min(72, parseInt(e.target.value, 10) || 48));
    e.target.value = String(state.n_frames);
  });

  $("#select-supersample").addEventListener("change", (e) => {
    state.supersample = parseInt(e.target.value, 10) || 2;
  });

  $("#select-intensity").addEventListener("change", (e) => {
    state.intensity = e.target.value;
    schedulePreview(80);
  });

  $("#select-line-etch").addEventListener("change", (e) => {
    state.line_etch = e.target.value;
    schedulePreview(80);
  });

  $("#check-dof").addEventListener("change", (e) => {
    state.enable_dof = e.target.checked;
    schedulePreview(80);
  });

  // Sliders
  $("#range-relief").addEventListener("input", (e) => {
    const v = parseFloat(e.target.value);
    state.relief_strength = v;
    $("#out-relief").textContent = `${v.toFixed(2)}x`;
    schedulePreview(180);
  });

  $("#range-quicksand").addEventListener("input", (e) => {
    const v = parseFloat(e.target.value);
    state.quicksand_strength = v;
    $("#out-quicksand").textContent = `${v.toFixed(2)}x`;
    schedulePreview(180);
  });

  // Preview views & angles
  for (const btn of $$(".seg-btn")) {
    btn.addEventListener("click", () => setPreviewView(btn.dataset.view, true));
  }

  for (const btn of $$(".angle-btn")) {
    btn.addEventListener("click", () => {
      state.angle_preset = btn.dataset.angle;
      for (const b of $$(".angle-btn")) {
        b.classList.toggle("is-active", b === btn);
      }
      if (state.preview_view !== "live3d") {
        setPreviewView("live3d", false);
      }
      requestLive3DPreview();
    });
  }

  $("#btn-refresh-preview").addEventListener("click", refreshCurrentView);

  $("#preview-image").addEventListener("dblclick", async () => {
    if (!tauriCore) return;
    let targetPath = "";
    if (state.preview_view === "gif") targetPath = state.last_rendered_gif;
    else if (state.preview_view === "depth") targetPath = state.depth_path;
    else if (state.preview_view === "subject") targetPath = state.subject_path;
    if (targetPath) {
      await invokeTauri("open_path", { path: targetPath });
    }
  });

  // Render dock & header utilities
  $("#btn-render").addEventListener("click", handleStartRender);
  $("#btn-cancel").addEventListener("click", handleCancelRender);

  $("#btn-reveal-output").addEventListener("click", async () => {
    if (!state.last_rendered_gif) return;
    if (tauriCore) {
      await invokeTauri("reveal_path", { path: state.last_rendered_gif });
    }
  });

  $("#btn-open-templates").addEventListener("click", async () => {
    if (!state.templatesDir) return;
    if (tauriCore) {
      await invokeTauri("reveal_path", { path: state.templatesDir });
    }
  });

  // HTML5 Drag & Drop fallback
  const dropTargets = [$("#dropzone-subject"), $("#preview-stage")];
  for (const el of dropTargets) {
    el.addEventListener("dragover", (e) => {
      e.preventDefault();
      el.classList.add("is-dragover");
    });
    el.addEventListener("dragleave", () => {
      el.classList.remove("is-dragover");
    });
    el.addEventListener("drop", async (e) => {
      e.preventDefault();
      el.classList.remove("is-dragover");
      const file = e.dataTransfer?.files?.[0];
      if (!file) return;
      try {
        setStatus(`正在接收拖入图片：${file.name}…`);
        const uploaded = await uploadBrowserFile(file);
        setSubjectPath(uploaded);
        setStatus(`已载入主体图片：${file.name}`);
      } catch (err) {
        setStatus(`拖入图片失败：${err.message}`);
      }
    });
  }

  // Native Tauri OS Drag & Drop (zero-copy disk path!)
  if (tauriEvent) {
    tauriEvent.listen("tauri://drag-enter", () => {
      $("#dropzone-subject")?.classList.add("is-dragover");
      $("#preview-stage")?.classList.add("is-dragover");
    });
    tauriEvent.listen("tauri://drag-leave", () => {
      $("#dropzone-subject")?.classList.remove("is-dragover");
      $("#preview-stage")?.classList.remove("is-dragover");
    });
    tauriEvent.listen("tauri://drag-drop", (event) => {
      $("#dropzone-subject")?.classList.remove("is-dragover");
      $("#preview-stage")?.classList.remove("is-dragover");
      const paths = event?.payload?.paths;
      if (Array.isArray(paths) && paths.length > 0) {
        const firstImg = paths.find((p) => /\.(png|jpe?g|webp|bmp|tiff?)$/i.test(p));
        if (firstImg) {
          setSubjectPath(firstImg);
          setStatus(`已载入主体图片：${fileBaseName(firstImg)}`);
        }
      }
    });
  }
}

/* ---------------------------------------------------------------------------
 * Startup Bootstrap: Wait for Silent Local Python Sidecar Backend
 * -------------------------------------------------------------------------*/
async function waitForBackendReady(maxAttempts = 45) {
  const bootStatus = $("#boot-status");
  for (let i = 1; i <= maxAttempts; i++) {
    try {
      const health = await apiGet("/api/health");
      if (health && health.ok) {
        return health;
      }
    } catch {
      // Backend still importing OpenCV / PyTorch
    }
    if (bootStatus && i > 4) {
      bootStatus.textContent = `正在预热 3D 全息着色器与本地深度模块… (${i}/${maxAttempts})`;
    }
    await new Promise((r) => setTimeout(r, 350));
  }
  throw new Error("连接本地 Python 渲染后端超时，请检查 Python 环境或 %TEMP%\\tarot_holographic_backend.log");
}

async function bootstrap() {
  bindUIEvents();
  const bootOverlay = $("#boot-overlay");
  const bootStatus = $("#boot-status");

  try {
    if (tauriCore) {
      const info = await invokeTauri("get_backend_info");
      if (info?.error) {
        throw new Error(info.error);
      }
      if (info?.url) {
        state.backendUrl = info.url;
      }
    }

    const health = await waitForBackendReady();
    const badge = $("#badge-engine");
    if (badge) {
      badge.textContent = health.ai_depth
        ? `GPU AI 深度就绪 · ${health.ai_desc}`
        : `极速离线引擎 · ${health.ai_desc}`;
      badge.classList.toggle("is-gpu", Boolean(health.ai_depth));
    }

    const catalog = await apiGet("/api/catalog");
    state.templatesDir = catalog.templates_dir || "";
    state.defaultOutputDir = catalog.default_output_dir || "";

    renderGallery("#gallery-studios", catalog.studios, "studio_bg", "#selected-studio-label");
    renderGallery("#gallery-bgs", catalog.bgs, "bg_template", "#selected-bg-label");
    renderGallery("#gallery-fgs", catalog.fgs, "fg_template", "#selected-fg-label");
    renderGallery("#gallery-styles", catalog.styles, "style", "#selected-style-label");
    renderGallery("#gallery-patterns", catalog.patterns, "pattern", "#selected-pattern-label");
    renderGallery("#gallery-metals", catalog.metals, "metal", "#selected-metal-label");

    bootOverlay.classList.add("is-hidden");
    setTimeout(() => {
      bootOverlay.hidden = true;
    }, 280);

    requestLive3DPreview();
  } catch (err) {
    if (bootStatus) {
      bootStatus.textContent = `启动失败：${err.message}`;
    }
    setStatus(`初始化失败：${err.message}`);
  }
}

bootstrap();
