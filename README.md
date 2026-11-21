# 3D 全息浮雕塔罗 & SAR 流麻闪卡工作室 (Tarot Holographic Studio)

基于 **Tauri v2 + Python (OpenCV / NumPy / Pillow / Depth-Anything-V2 ONNX & CUDA)** 打造的 3D 全息悬浮塔罗 / 宝可梦 SAR 流麻闪卡桌面生成器。 感谢 X上的@yygqdkqk 和@aiiii498 的提示词

## 核心特性

- **三层卡内视差窗口 & 实体卡边**：8 层金属拉丝卡边 + 随倾角流动的倒角亮边，0° 正视像素级对齐，倾摆无穿帮缝隙。
- **单目 AI 深度驱动 3D 立体主体**：
  - 支持内置轻量 **Depth-Anything-V2-Small (ONNX Runtime)**（无需 Python / 无需 PyTorch / 纯 CPU ~0.4–0.7s）、本机 **CUDA GPU（Small / Large）** 以及极速离线导向滤波算法。
  - ECC 仿射对齐 + 轮廓外扩 + 保边双边滤波 + 16 步透视光线步进（Relief Mapping）与自遮挡投影。
- **卡内背景流麻（流沙细闪）+ SAR 分区全息着色器**：
  - 双层周期流动珠光流沙 + 300 颗带独立随机 3D 法线的微闪亮片。
  - 预乘 Alpha 相机空间景深（杜绝灰边与串色）、4 芒十字星闪与空间漂浮粒子。
  - **48 帧严格无缝循环 GIF**（~22 fps，共享 256 色 Median Cut 调色板 + Floyd–Steinberg 抖动）。
- **丰富可视化默认模板库**：
  - **7 套**非纯黑卡外摄影棚氛围背景 + **10 套**卡内后景模板 + **11 套**透明烫金前景框（自动程序化生成）。
  - **11 套**整体氛围气质 + **11 款**程序化光栅微蚀刻纹理（含碎钻棱镜 SAR、指纹等高 UR、赛博蜂巢、国风祥云、玫瑰花窗、樱花花雨等）+ **11 款**烫金/金属拉丝色系。
- **现代暗黑鎏金 Tauri v2 桌面端**：原生文件选择与拖拽导入、零黑框静默后端、右侧多角度实时 3D 预览舞台。

---

## 快速开始

### 方式一：直接下载便携版（Windows 免装 Python，推荐）
1. 前往本仓库右侧 **[Releases](../../releases)** 页面，下载 **`TarotHolographicStudio_AI_Portable.zip`**（约 165 MB）。
2. 完整解压整个文件夹。
3. 双击 **`TarotHolographicStudio.exe`** 即可使用（已内置 Depth-Anything-V2 ONNX 轻量 AI 模型与全部默认模板，无需配置 Python 或显卡环境）。

### 方式二：从源码运行（支持本机 CUDA GPU 加速）

#### 1. 安装 Python 依赖
```bash
pip install -r requirements.txt
# 如需本机 NVIDIA CUDA GPU 加速或 Depth-Anything-V2-Large：
# pip install torch torchvision transformers
```

#### 2. 运行桌面端
- **Tauri v2 开发模式**（需安装 Node.js / pnpm 与 Rust 工具链）：
  ```bash
  cd tauri-app
  pnpm install
  pnpm tauri dev
  ```
- **或编译 Tauri Release 独立可执行文件**：
  ```bash
  cd tauri-app
  pnpm run build:exe
  ```
- **或直接运行轻量 Tkinter 界面**：
  ```bash
  python tarot_gui.py
  ```

#### 3. 命令行直接渲染
```bash
python render_sar_card.py --subject path/to/character.png --output holographic_card.gif --style prism --pattern prism_shard --metal aurora_prism
```

---


