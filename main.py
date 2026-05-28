"""Command-line entry for the speak-keyboard prototype."""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time

import keyboard

from app import HotkeyManager, TranscriptionResult, TranscriptionWorker, load_config, type_text
from app.audio_capture import find_loopback_device, list_devices
from app.plugins.dataset_recorder import wrap_result_handler
from app.logging_config import setup_logging


logger = logging.getLogger(__name__)


_TOGGLE_DEBOUNCE_SECONDS = 0.2
_toggle_lock = threading.Lock()
_last_toggle_time = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speak Keyboard prototype")
    parser.add_argument("--config", help="Path to config JSON")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single transcription cycle for debugging",
    )
    parser.add_argument("--save-dataset", action="store_true", help="Persist audio/text pairs")
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    # 配置日志系统（统一配置）
    from app.config import ensure_logging_dir
    log_dir_abs = ensure_logging_dir(config)
    setup_logging(
        level=config["logging"].get("level", "INFO"),
        log_dir=log_dir_abs
    )

    output_cfg = config.get("output", {})
    output_method = output_cfg.get("method", "auto")
    append_newline = output_cfg.get("append_newline", False)
    log_file = output_cfg.get("log_file", "logs/transcription.txt")
    log_file_abs = log_file if os.path.isabs(log_file) else os.path.join(log_dir_abs, os.path.basename(log_file))

    # 先创建worker（没有回调）
    worker = TranscriptionWorker(
        config_path=args.config,
        on_result=None,  # 稍后设置
    )
    
    # 创建result handler（需要worker引用）
    worker.on_result = _make_result_handler(output_method, append_newline, worker, log_file_abs)
    if args.save_dataset:
        worker.on_result = wrap_result_handler(worker.on_result, worker, args.dataset_dir)
    
    hotkeys = HotkeyManager()

    toggle_combo = config["hotkeys"].get("toggle", "f2")
    hotkeys.register(toggle_combo, lambda: _toggle(worker))

    # F3：系统声音识别
    loopback_device = config["audio"].get("loopback_device")
    if loopback_device is None:
        loopback_device = find_loopback_device()
    if loopback_device is not None:
        hotkeys.register("f3", lambda: _toggle_system(worker, loopback_device))
        logger.info("F3 已注册：系统声音识别（设备 #%s）", loopback_device)
    else:
        logger.warning("未找到 loopback 设备，F3 系统声音识别不可用")

    # F4：文件识别
    hotkeys.register("f4", lambda: _transcribe_from_file(worker))
    logger.info("F4 已注册：音频文件识别")

    # 打印音频设备列表
    devices = list_devices()
    logger.info("可用音频设备:")
    for d in devices:
        marker = " <-- loopback" if d["index"] == loopback_device else ""
        logger.info("  #%s: %s (输入:%s, 输出:%s)%s",
                     d["index"], d["name"], d["input_channels"], d["output_channels"], marker)

    try:
        logger.info("Speak Keyboard 启动完成，按 %s 录音/F3 系统声音/F4 文件识别，Ctrl+C 退出", toggle_combo)
        if args.once:
            _toggle(worker)
            input("按 Enter 停止并退出...")
            _toggle(worker)
        else:
            keyboard.wait()
    except KeyboardInterrupt:
        logger.info("用户中断，正在退出...")
    finally:
        # 清理所有资源
        try:
            worker.stop()
        except Exception as exc:
            logger.debug("停止 worker 时出错: %s", exc)
        
        try:
            worker.cleanup()
        except Exception as exc:
            logger.debug("清理 worker 时出错: %s", exc)
        
        try:
            hotkeys.cleanup()
        except Exception as exc:
            logger.debug("清理热键时出错: %s", exc)
        
        logger.info("所有资源已清理，正常退出")
        import sys
        sys.exit(0)


def _make_result_handler(output_method: str, append_newline: bool, worker: TranscriptionWorker, log_file: str):
    def _handle_result(result: TranscriptionResult) -> None:
        if result.error:
            logger.error("转写失败: %s", result.error)
            return

        # 获取转录统计信息
        stats = worker.transcription_stats

        logger.info(
            "转写成功: %s (推理 %.2fs) [已完成 %d/%d，队列剩余 %d]",
            result.text,
            result.inference_latency,
            stats["completed"],
            stats["submitted"],
            stats["pending"],
        )
        type_text(
            result.text,
            append_newline=append_newline,
            method=output_method,
        )

        # 追加写入 txt 文件
        if result.text and log_file:
            try:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(result.text + "\n")
            except Exception as exc:
                logger.warning("写入转录日志失败: %s", exc)

    return _handle_result


def _toggle(worker: TranscriptionWorker) -> None:
    global _last_toggle_time
    now = time.monotonic()
    with _toggle_lock:
        if now - _last_toggle_time < _TOGGLE_DEBOUNCE_SECONDS:
            logger.debug("忽略快速重复的录音切换请求 (%.3fs)", now - _last_toggle_time)
            return
        _last_toggle_time = now

    if worker.is_running:
        worker.stop()
        stats = worker.transcription_stats
        if stats["pending"] > 0:
            logger.info(
                "录音已停止并提交转录，队列中还有 %d 个任务等待处理",
                stats["pending"]
            )
    else:
        stats = worker.transcription_stats
        if stats["pending"] > 0:
            logger.info(
                "开始录音（后台还有 %d 个转录任务正在处理）",
                stats["pending"]
            )
        worker.start()


def _toggle_system(worker: TranscriptionWorker, device) -> None:
    """F3 系统声音 toggle，与 _toggle 逻辑相同但使用 loopback 设备。"""
    global _last_toggle_time
    now = time.monotonic()
    with _toggle_lock:
        if now - _last_toggle_time < _TOGGLE_DEBOUNCE_SECONDS:
            logger.debug("忽略快速重复的系统声音切换请求 (%.3fs)", now - _last_toggle_time)
            return
        _last_toggle_time = now

    if worker.is_running:
        worker.stop()
        stats = worker.transcription_stats
        if stats["pending"] > 0:
            logger.info(
                "系统声音采集已停止并提交转录，队列中还有 %d 个任务等待处理",
                stats["pending"]
            )
    else:
        stats = worker.transcription_stats
        if stats["pending"] > 0:
            logger.info(
                "开始采集系统声音（后台还有 %d 个转录任务正在处理）",
                stats["pending"]
            )
        worker.start_with_device(device)


def _transcribe_from_file(worker: TranscriptionWorker) -> None:
    """F4 文件识别：提示用户输入文件路径并提交转录。"""
    if worker.is_running:
        logger.warning("正在录音中，请先按 F2/F3 停止录音后再使用文件识别")
        return

    try:
        file_path = input("请输入音频文件路径（支持 wav/mp3/flac/ogg 等）: ").strip()
    except EOFError:
        return

    if not file_path:
        return

    # 去除引号
    file_path = file_path.strip('"').strip("'")

    logger.info("提交文件转录: %s", file_path)
    # 在独立线程中执行，避免阻塞 keyboard 回调线程
    t = threading.Thread(target=worker.transcribe_file, args=(file_path,), daemon=True)
    t.start()


if __name__ == "__main__":
    main()

