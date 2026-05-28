# VocoType CLI — 离线语音转文字工具

<h2 align="center">本地离线，多语言语音转文字</h2>

**VocoType CLI** 是一款完全离线的语音转文字命令行工具，支持麦克风录音、系统声音捕获、音频文件转写三种输入方式。所有识别均在本地完成，不上传任何数据。

基于 [FunASR](https://github.com/modelscope/FunASR) 的 SenseVoice 多语言模型，支持中文、英文、日文、粤语、韩语自动检测。

---

## 功能特性

- **三种音源输入**：麦克风（F2）、系统声音（F3）、音频文件（F4）
- **多语言识别**：中文 / 英文 / 日文 / 粤语 / 韩语，自动检测语言
- **边识别边输出**：每 5 秒自动提交转录，实时查看结果
- **自动保存**：转录结果自动追加写入 `logs/transcription.txt`
- **100% 离线**：所有识别在本地完成，无需联网，保护隐私
- **可选云端后端**：支持接入火山引擎 BigASR 流式识别

## 快捷键

| 快捷键 | 功能 |
|:--|:--|
| **F2** | 麦克风录音 toggle（开始/停止） |
| **F3** | 系统声音识别 toggle（WASAPI loopback） |
| **F4** | 音频文件识别（输入文件路径，支持 wav/mp3/flac/ogg） |
| **Ctrl+C** | 退出程序 |

## 环境要求

- Windows 10/11
- Python 3.12
- 麦克风（F2/F3 需要音频输入设备）

## 安装与运行

```bash
# 1. 克隆仓库
git clone https://github.com/doubletao/vocotype-cli-local-file.git
cd vocotype-cli-local-file

# 2. 创建并激活虚拟环境（推荐）
pip install uv
uv venv --python 3.12
.\.venv\Scripts\activate

# 3. 安装依赖
uv pip install -r requirements.txt

# 4. 运行
python main.py
```

> **首次运行**会自动下载 SenseVoice 模型（约 1GB），请确保网络连接稳定。模型缓存在 `~/.cache/modelscope/hub/models/iic/`。

## 使用说明

### 麦克风识别（F2）

按 F2 开始录音，再次按 F2 停止。录音过程中每 5 秒自动转录一次并输出结果。

### 系统声音识别（F3）

按 F3 开始捕获系统声音（扬声器/耳机播放的音频），再次按 F3 停止。适用于会议录音、视频字幕等场景。

> 需要系统存在 WASAPI loopback 设备。程序启动时会自动检测并打印可用设备列表。

### 音频文件识别（F4）

按 F4 后输入音频文件路径，支持 wav、mp3、flac、ogg 等格式。大文件会自动分片转录，边识别边输出。

### 输出

- 转录结果实时打印到终端日志
- 自动追加保存到 `logs/transcription.txt`
- 可通过 `--save-dataset` 参数保存音频/文本对到 `dataset/` 目录

## 配置

创建 `config.json` 可自定义配置，使用 `--config` 参数加载：

```bash
python main.py --config config.json
```

配置示例：

```json
{
  "asr": {
    "engine": "sensevoice",
    "language": "auto"
  },
  "audio": {
    "segment_seconds": 5,
    "loopback_device": null
  },
  "output": {
    "log_file": "logs/transcription.txt"
  }
}
```

### 引擎切换

| 引擎 | 配置值 | 说明 |
|:--|:--|:--|
| **SenseVoice**（默认） | `"engine": "sensevoice"` | 多语言：中英日粤韩，自动检测 |
| **Paraformer** | `"engine": "paraformer"` | 中文专用，精度更高 |

### 后端切换

| 后端 | 配置值 | 说明 |
|:--|:--|:--|
| **FunASR**（默认） | `"backend": "funasr"` | 本地离线推理 |
| **Volcengine** | `"backend": "volcengine"` | 火山引擎云端识别，需配置 `app_key` 和 `access_key` |

## 独立转写 CLI

可单独使用 `funasr_server.py` 进行音频文件转写测试：

```bash
python -m app.funasr_server --audio test.wav --pretty
```

## 常见问题

**Q: F3 系统声音识别不可用？**

A: 程序启动时会打印可用音频设备列表。如果没有检测到 loopback 设备，可能需要在系统音频设置中启用"立体声混音"（Stereo Mix）。

**Q: SenseVoice 模型加载失败？**

A: 确保已安装所有依赖（`pip install -r requirements.txt`）。首次加载会自动导出 ONNX 模型，可能需要较长时间。

**Q: 数据安全吗？**

A: 默认使用本地 FunASR 后端，所有识别在本地完成，音频数据不会上传。仅当使用 Volcengine 后端时，音频会发送至火山引擎服务器。

## 项目结构

```
├── main.py                  # 程序入口
├── app/
│   ├── transcribe.py        # 核心协调器：录音、分段、异步转录
│   ├── audio_capture.py     # 音频采集与设备枚举
│   ├── funasr_server.py     # 本地 ASR 后端（SenseVoice / Paraformer）
│   ├── volcengine_asr.py    # 火山引擎云端 ASR 后端
│   ├── config.py            # 配置管理
│   ├── hotkeys.py           # 全局热键注册
│   └── output.py            # 文本输出
├── requirements.txt
└── logs/
    └── transcription.txt    # 转录结果自动保存
```

## 致谢

- **[FunASR](https://github.com/modelscope/FunASR)** — 阿里巴巴达摩院开源的语音识别框架
- **[SenseVoice](https://github.com/FunAudioLLM/SenseVoice)** — 多语言语音理解模型
