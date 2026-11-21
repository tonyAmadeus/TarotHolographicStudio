# 3D 全息浮雕塔罗 & SAR 流麻闪卡工作室 (Tarot Holographic Studio)

基于 **Tauri v2 + Python (OpenCV / NumPy / Pillow / Depth-Anything-V2 ONNX & CUDA)** 打造的 3D 全息悬浮塔罗 / 宝可梦 SAR 流麻闪卡桌面生成器。感谢 X 上的 @yygqdkqk 和 @aiiii498 的提示词。

## 核心特性

- **强烈 3D 景深、全息镭射与实体卡边**：8 层金属拉丝卡边 + 随倾角流动的倒角亮边，卡内三层视差窗口（0° 正视像素级对齐、倾摆无穿帮缝隙；满幅实拍自动满铺至圆角卡边），并在凹陷卡底、蚀刻光栅、立体内腔壁及前景框棱边叠合周期锁定全息镭射衍射波。
- **单目深度平滑高度场 + 混合 3D 视差引擎**：
  - **CIE Lab 三通道彩色导向滤波 + 16-bit 无损 AI 深度链路**：同时锁紧亮度边与同明度异色赛璐珞轮廓，AI 深度保存为无损 16-bit PNG（实测保留 5.8 万级连续深度），消除外轮廓飞点与二次平滑模糊。
  - **深度分布自适应分层（Adaptive Quantile MPI）+ 2 步连续 Relief Mapping / Depth Warping + 局部侧壁 + 净边遮挡补全**：自动把更多立体层数分配到主体聚集区间，兼具平滑曲面形变、自遮挡、细腻 2–2.8px 局部倒角侧壁及去黑线遮挡补全。
  - **3 档深度分层近实远虚动态投影**：近底部位紧实贴合阴影，前景凸出肢体投出大偏移柔和悬浮半影，同步作用于卡底与流麻层。
- **Premultiplied Alpha + 纯净赛璐珞光照 & Pillow RGBA 合成**：
  - 全链路预乘 Alpha + 平滑法线 + Half-Lambert 柔和漫反射 + $C^1$ 连续高光 Roll-off + 暗色轮廓高光隔离，保留纯粹赛璐珞画风、清晰线条与低噪点暗部。
  - **OpenCV / NumPy** 负责逐帧 3D 透视变换与光效运算，**Pillow** 负责 `Image.alpha_composite` RGBA 分层合成、Lanczos 降采样与 GIF 编码。
- **全比特深度图与多引擎 AI 深度支持**：
  - 原生无损支持 **8-bit / 16-bit (`I;16` / `uint16`) / 32-bit Float** 外部深度图加载与归一化预览，CLI/API 支持深度反转（`--invert-depth`）与曲线调节（`--depth-gamma`）。
  - 支持内置轻量 **Depth-Anything-V2-Small (ONNX Runtime)**（无需 Python / 无需 PyTorch / 纯 CPU ~0.4–0.7s）、本机 **NVIDIA CUDA GPU（`Small` 与旗舰 `Large` FP16 加速）** 及极速离线导向滤波算法。
  - 实时预览内置 **AI 深度内存缓存**，换视角/调模板单帧低至 ~0.23s。
- **主体构图微调 & GIF 清晰度 3 档调节**：
  - **主体缩放（`40%`–`180%`，支持预览区鼠标滚轮缩放）+ 水平/垂直偏移 + 5 种卡片画幅比例**（跟随原图 / 标准塔罗 5:8 / 宝可梦 5:7 / 经典竖卡 3:4 / 方形流麻砖 1:1），预乘 Alpha 深度图同步对齐。
  - **3 档 GIF 清晰度预设**：**轻量流畅（`360px` · 36帧）** / **高清标准（`480px` · 48帧，推荐）** / **超清典藏（`680px` · 48帧 · 精细锐化 + 200px 高精调色板采样）**，也支持手动自定义宽度（最高 `960px`）。
- **共享 256 色三子调色板 + Floyd–Steinberg 抖动**：
  - 为**暗部渐变（≤54色）**、**赛璐珞纯白/肤色（≤36色）**与**全息虹彩卡片主体（≥166色）**独立分配精确色槽后合并为全帧共享 256 色调色板，配合 Floyd–Steinberg 抖动显著减少虹彩色带、消除暗部颗粒噪点与帧间闪色。
- **丰富可视化默认模板库**：
  - **7 套**非纯黑卡外摄影棚氛围背景 + **10 套**卡内后景模板 + **11 套**透明烫金前景框（自动程序化生成）。
  - **11 套**整体氛围气质 + **11 款**程序化光栅微蚀刻纹理（含碎钻棱镜 SAR、指纹等高 UR、赛博蜂巢、国风祥云、玫瑰花窗、樱花花雨、十字星闪格等）+ **11 款**烫金/金属拉丝色系。

---

## 快速开始

### 方式一：直接下载便携版（Windows 免装 Python，推荐）
1. 前往本仓库右侧 **[Releases](../../releases)** 页面，下载最新版 **`TarotHolographicStudio_AI_Portable.zip`**（约 182 MB）。
2. **完整解压**整个压缩包目录。
3. 双击 **`TarotHolographicStudio.exe`** 即可使用（已内置 Depth-Anything-V2 ONNX 轻量 AI 模型与全部默认模板，无需安装 Python 或独立显卡，无黑框）。

### 方式二：从源码运行（支持本机 CUDA GPU 与 `Depth-Anything-V2-Large`）

#### 1. 安装 Python 依赖
```bash
pip install -r requirements.txt
# 如需本机 NVIDIA CUDA GPU 加速或 Depth-Anything-V2-Large：
# pip install torch torchvision transformers
```
> 若本机已将 `depth-anything/Depth-Anything-V2-Large-hf` 下载至 通过环境变量 `TAROT_LARGE_DEPTH_DIR` 指定路径），程序会自动检测并启用 FP16 CUDA GPU 加速。

#### 2. 运行桌面端
- **Tauri v2 开发模式**（需安装 Node.js / pnpm 与 Rust 工具链）：
  ```bash
  cd tauri-app
  pnpm install
  pnpm tauri dev
  ```
- **编译 Tauri Release 独立可执行文件**：
  ```bash
  cd tauri-app
  pnpm run build:exe
  ```
- **或直接运行备用 Tkinter 界面**：
  ```bash
  python tarot_gui.py
  ```

#### 3. 命令行直接渲染
```bash
python render_sar_card.py \
  --subject path/to/character.png \
  --output holographic_card.gif \
  --style prism --pattern prism_shard --metal aurora_prism \
  --card-aspect tarot --subject-scale 1.05 --width 480
```

---

## 更新日志

### v1.2.0
- **混合 3D 景深与全息镭射引擎升级**：提升 3D 倾角（`±19.5°`）、卡体厚度与三层视差深度，结合平滑高度场、2 步连续 Depth Warping / Relief Mapping、自遮挡、细腻局部倒角侧壁及去黑线遮挡补全；新增周期锁定的三维全息镭射衍射光波。
- **单目深度深度融合（方案 A）**：升级 **CIE Lab 三通道彩色导向滤波**（同明度异色边界锐度提升 6.7×）、无损 **16-bit AI 深度图保存**与去二次平滑、**深度分布自适应分层（Adaptive Quantile MPI）** + 动态深度支点、`--invert-depth` / `--depth-gamma` 深度反转与曲线支持，以及 **3 档深度分层近实远虚动态接触投影**。
- **纯粹赛璐珞画风 & Pillow RGBA 合成**：全链路 Premultiplied Alpha、$C^1$ 平滑法线、Half-Lambert 光照、$C^1$ 高光 Roll-off 与暗色轮廓保护；升级为 OpenCV/NumPy 逐帧透视运算 + Pillow `Image.alpha_composite` RGBA 分层合成。
- **共享 256 色三子调色板 + Floyd–Steinberg 抖动**：独立分配暗部（≤54色）、赛璐珞纯白/肤色（≤36色）与卡片全息虹彩（≥166色），暗部噪点大幅降低且赛璐珞白裙/肤色更纯净平滑。

### v1.1.0
- 重构 3D 视差核心为 **14 层遮挡补全多平面立体视差（MPI）+ RGB 引导深度吸附 + 雕塑法线 & SSAO**，大幅提升立体贴合度与自然感。
- 新增 **上传主体尺寸缩放（40%~180% / 滚轮缩放）、X/Y 位置微调与 5 种卡片画幅比例**。
- 新增 **GIF 清晰度 3 档调节**（轻量流畅 360px / 高清标准 480px / 超清典藏 680px）。
- 完整支持 **16 位 (`I;16`) / 32 位深度图** 无损加载与预览归一化，修复 16 位深度图变全白问题。
- 彻底修复渲染与导出 GIF 画面发灰泛白问题，升级为乘法聚光灯 + 主体 2× 加权 `FASTOCTREE` 共享 256 色调色板。
- 新增 **`Depth-Anything-V2-Large` 本机 GPU 加速支持** 与实时预览 AI 深度缓存。

### v1.0.0
- 首个公开版本：Tauri v2 暗黑鎏金桌面工作室、3 层 SAR 流麻引擎、28 套前/后/卡外模板、12×3 视觉配置墙与 ONNX 轻量便携版。
