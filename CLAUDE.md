# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

VocoType CLI 是一款离线语音输入法的命令行版本，核心功能：麦克风录音 → 语音识别（FunASR 本地或 Volcengine 云端）→ 将文本注入当前活动窗口。仅支持 Windows 平台。

## 常用命令

```bash
# 环境搭建
pip install uv
uv venv --python 3.12
.\.venv\Scripts\activate
uv pip install -r requirements.txt

# 运行（首次会自动下载 ONNX 模型）
python main.py                          # 正常模式：F2 麦克风 / F3 系统声音 / F4 文件识别
python main.py --once                   # 单次转录调试模式
python main.py --save-dataset           # 保存音频/文本对到 dataset/
python main.py --config config.json     # 使用自定义配置（如切换后端/引擎）

# 独立转写 CLI（测试用）
python -m app.funasr_server --audio test.wav --pretty
```

无测试套件、无 linter 配置。

## 架构

### 数据流

```
音源（麦克风/系统声音/文件） → 缓冲区 → 分段定时器(5s) / stop()
  → 提交到异步队列 → 后台线程调用 ASR 后端
  → TranscriptionResult → type_text() 注入活动窗口 + 追加写入 txt
```

### 热键

- **F2**：麦克风录音 toggle（开始/停止）
- **F3**：系统声音识别 toggle（WASAPI loopback）
- **F4**：音频文件识别（输入文件路径，支持 wav/mp3/flac/ogg）

### 核心模块 (`app/`)

- **`transcribe.py`** — `TranscriptionWorker`：核心协调器。管理录音会话、异步转录队列（`queue.Queue`，maxsize=10）、后台工作线程。支持分段自动提交（每 5 秒）、指定设备录音（`start_with_device`）、文件转录（`transcribe_file`）。单次录音有 20MB 大小限制。
- **`audio_capture.py`** — `AudioCapture`：基于 `sounddevice` 的麦克风采集，使用回调 + 队列模式。支持设备回退。提供 `list_devices()` 和 `find_loopback_device()` 用于设备枚举。
- **`funasr_server.py`** — `FunASRServer`：本地离线 ASR 后端。支持两种引擎：`sensevoice`（SenseVoice 多语言：中英日粤韩）和 `paraformer`（中文专用）。SenseVoice 加载单个模型；Paraformer 加载 ASR+VAD+PUNC 三个模型。模型缓存在 `~/.cache/modelscope/hub/models/iic/`。
- **`volcengine_asr.py`** — `VolcengineASRClient`：火山引擎 BigASR 流式识别后端，通过 WebSocket + 自定义二进制协议通信。同步接口内部包装 asyncio。
- **`output.py`** — `type_text()`：Windows 文本注入，三种策略依次尝试：`keyboard.write` → 剪贴板+Ctrl+V → Unicode SendInput。
- **`config.py`** — 配置管理。`DEFAULT_CONFIG` 定义所有默认值，`load_config()` 深度合并用户 JSON 覆盖。
- **`hotkeys.py`** — 全局热键注册，基于 `keyboard` 库。
- **`plugins/dataset_recorder.py`** — AOP 风格包装器，用 `--save-dataset` 启用时将每次转录的音频和文本保存为 JSONL + WAV。

### 后端与引擎切换

通过 `config.json` 配置：
- `"backend": "funasr"`（默认）：本地 ONNX 推理，完全离线
- `"backend": "volcengine"`：火山引擎云端，需配置 `app_key` 和 `access_key`
- `"asr.engine": "sensevoice"`（默认）：SenseVoice 多语言模型（中英日粤韩，自动检测语言）
- `"asr.engine": "paraformer"`：Paraformer 中文专用模型

## 关键环境变量

- `FUNASR_DEVICE` — 推理设备，默认 `cpu`，可设 `cuda:0`
- `OMP_NUM_THREADS` — ONNX 并行线程数，默认 `8`
- `FUNASR_MODEL_REVISION` — 模型版本，默认 `v2.0.5`
- `FUNASR_ASR_MODEL` / `FUNASR_VAD_MODEL` / `FUNASR_PUNC_MODEL` — 覆盖 Paraformer 模型名称
- `FUNASR_SENSEVOICE_MODEL` — 覆盖 SenseVoice 模型名称

## 代码风格

- 类型注解 + `from __future__ import annotations`
- `dataclass` 用于数据结构
- 日志使用 `logging` 模块，通过 `app/logging_config.py` 统一配置
- 线程同步使用 `threading.Event` / `threading.Lock` / `threading.RLock`
- 资源清理通过 `cleanup()` 方法 + `__del__` 兜底
