"""Session-based transcription worker — supports FunASR (local) and Volcengine (cloud) backends."""

from __future__ import annotations

import logging
import os
import queue
import tempfile
import threading
import time
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .audio_capture import AudioCapture
from .config import ensure_logging_dir, load_config


logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    text: str
    raw_text: str
    duration: float
    inference_latency: float
    confidence: float
    error: Optional[str] = None


class TranscriptionWorker:
    """Capture full session audio and transcribe once when stopped."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        on_result: Optional[Callable[[TranscriptionResult], None]] = None,
    ) -> None:
        self.config = load_config(config_path)
        self.on_result = on_result
        self.log_dir = ensure_logging_dir(self.config)
        self.last_segment_path: Optional[Path] = None
        self._session_id_counter = itertools.count(1)
        self._current_session_id: Optional[int] = None

        audio_cfg = self.config["audio"]
        self.audio = AudioCapture(
            sample_rate=audio_cfg["sample_rate"],
            block_ms=audio_cfg["block_ms"],
            device=audio_cfg.get("device"),
        )

        backend = self.config.get("backend", "funasr").lower()
        self._backend = backend

        if backend == "volcengine":
            from app.volcengine_asr import VolcengineASRClient
            self._volcengine_client = VolcengineASRClient(
                self.config.get("volcengine", {})
            )
            self.fun_server = None  # 不使用本地 FunASR 模型
            logger.info("使用 Volcengine BigASR 流式识别后端")
        else:
            from app.funasr_server import FunASRServer
            self._volcengine_client = None
            asr_cfg = self.config.get("asr", {})
            engine = asr_cfg.get("engine", "sensevoice")
            self.fun_server = FunASRServer(engine=engine)
            init_result = self.fun_server.initialize()
            if not init_result.get("success"):
                raise RuntimeError(f"FunASR 初始化失败: {init_result}")
            logger.info("使用 FunASR 本地离线识别后端")

        self._running = threading.Event()
        self._recording = threading.Event()
        self._stop_requested = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._state_lock = threading.RLock()
        self._audio_cfg = audio_cfg
        self._buffer: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()
        # 单次会话大小限制（字节）与计数器（配置健壮性：转换为正整型，非法回退至20MB）
        try:
            raw_limit = audio_cfg.get("max_session_bytes", 20 * 1024 * 1024)
            self._max_session_bytes: int = int(raw_limit)
            if self._max_session_bytes <= 0:
                raise ValueError
        except Exception:
            self._max_session_bytes = 20 * 1024 * 1024
            logger.warning("max_session_bytes 配置非法，已回退至 20MB")
        self._session_bytes: int = 0

        # 分段配置
        self._segment_seconds: int = int(audio_cfg.get("segment_seconds", 5))
        self._segment_timer: Optional[threading.Timer] = None
        self._original_device = audio_cfg.get("device")  # 保存原始设备，供 start_with_device 恢复

        # 异步转录队列和工作线程
        self._transcription_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=10)
        self._transcription_thread: Optional[threading.Thread] = None
        self._transcription_running = threading.Event()
        self._transcription_task_count = 0  # 已提交的任务计数
        self._transcription_completed_count = 0  # 已完成的任务计数
        
        # 启动转录工作线程
        self._start_transcription_worker()

    def __del__(self) -> None:
        """析构函数，确保资源被清理"""
        try:
            self.cleanup()
        except Exception as exc:
            logger.debug("析构函数清理时出错: %s", exc)

    def cleanup(self) -> None:
        """清理所有资源，包括缓冲区、音频设备和 ASR 后端。"""
        logger.debug("开始清理 TranscriptionWorker 资源")
        try:
            # 停止录音（_running 可能在 __init__ 失败时未创建）
            if hasattr(self, '_running') and self._running.is_set():
                self.stop()

            # 取消分段定时器
            if hasattr(self, '_segment_timer') and self._segment_timer is not None:
                self._segment_timer.cancel()
                self._segment_timer = None

            # 停止转录工作线程
            if hasattr(self, '_transcription_running'):
                self._stop_transcription_worker()

            # 清理缓冲区
            if hasattr(self, '_buffer_lock'):
                with self._buffer_lock:
                    self._buffer.clear()

            # 停止音频捕获
            if hasattr(self, 'audio'):
                self.audio.stop()

            # 清理 ASR 后端资源
            if hasattr(self, 'fun_server') and self.fun_server is not None:
                self.fun_server.cleanup()
            elif hasattr(self, '_volcengine_client') and self._volcengine_client is not None:
                self._volcengine_client.cleanup()

            logger.debug("TranscriptionWorker 资源清理完成")
        except Exception as exc:
            logger.error("清理资源时出错: %s", exc)

    def _start_transcription_worker(self) -> None:
        """启动转录工作线程"""
        if self._transcription_running.is_set():
            logger.debug("转录工作线程已在运行")
            return
        
        self._transcription_running.set()
        self._transcription_thread = threading.Thread(
            target=self._transcription_worker_loop,
            daemon=True,
            name="TranscriptionWorker"
        )
        self._transcription_thread.start()
        logger.info("转录工作线程已启动")

    def _stop_transcription_worker(self, timeout: float = 3.0) -> None:
        """停止转录工作线程，等待队列清空
        
        Args:
            timeout: 等待队列清空的超时时间（秒），默认3秒
        """
        if not self._transcription_running.is_set():
            logger.debug("转录工作线程未运行")
            return
        
        pending = self._transcription_queue.qsize()
        if pending > 0:
            logger.info(f"正在停止转录工作线程，队列中还有 {pending} 个任务，最多等待 {timeout} 秒...")
        else:
            logger.info("正在停止转录工作线程...")
        
        # 等待队列中的任务完成（最多等待timeout秒）
        start_time = time.time()
        while not self._transcription_queue.empty():
            elapsed = time.time() - start_time
            if elapsed > timeout:
                remaining = self._transcription_queue.qsize()
                logger.warning(f"等待超时（{timeout}秒），强制退出，丢弃 {remaining} 个未完成任务")
                break
            time.sleep(0.1)
        
        # 发送停止信号（None表示停止）
        self._transcription_running.clear()
        try:
            self._transcription_queue.put(None, timeout=0.5)
        except queue.Full:
            logger.warning("转录队列已满，无法发送停止信号")
        
        # 等待线程结束
        if self._transcription_thread and self._transcription_thread.is_alive():
            self._transcription_thread.join(timeout=2.0)
            if self._transcription_thread.is_alive():
                logger.warning("转录工作线程未能在2秒内结束，强制继续退出")
        
        self._transcription_thread = None
        logger.info(f"转录工作线程已停止，共完成 {self._transcription_completed_count}/{self._transcription_task_count} 个任务")

    def _transcription_worker_loop(self) -> None:
        """转录工作线程的主循环，从队列中获取音频并转录"""
        logger.info("转录工作线程开始运行")
        
        while self._transcription_running.is_set():
            try:
                # 从队列获取音频数据（阻塞等待，超时1秒）
                samples = self._transcription_queue.get(timeout=1.0)
                
                # None是停止信号
                if samples is None:
                    logger.debug("收到停止信号，转录工作线程退出")
                    break
                
                # 执行转录
                logger.info(f"开始处理转录任务 #{self._transcription_completed_count + 1}，队列剩余: {self._transcription_queue.qsize()}")
                self._transcribe_once(samples)
                self._transcription_completed_count += 1
                
                # 标记任务完成
                self._transcription_queue.task_done()
                
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as exc:
                logger.error(f"转录工作线程出错: {exc}", exc_info=True)
                # 继续运行，不因单个任务失败而退出
        
        logger.info("转录工作线程已退出")

    def start(self) -> None:
        with self._state_lock:
            if self._running.is_set():
                logger.debug("Transcription worker 已在运行，忽略重复启动")
                return

            session_id = next(self._session_id_counter)
            logger.info("Transcription worker starting (session_id=%s)", session_id)
            self._running.set()
            self._stop_requested.clear()
            with self._buffer_lock:
                self._buffer.clear()
                self._session_bytes = 0
            self.audio.start()
            self._recording.set()
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            self._current_session_id = session_id

            # 启动分段定时器
            self._start_segment_timer()

    def start_with_device(self, device) -> None:
        """用指定设备开始录音（F3 系统声音用）。临时切换设备后调用 start()。"""
        self.audio.device = device
        self.start()

    def _start_segment_timer(self) -> None:
        """启动分段定时器，每隔 segment_seconds 自动提交当前缓冲区。"""
        if self._segment_seconds <= 0:
            return
        self._segment_timer = threading.Timer(self._segment_seconds, self._segment_timer_callback)
        self._segment_timer.daemon = True
        self._segment_timer.start()

    def _segment_timer_callback(self) -> None:
        """分段定时器回调：提交当前缓冲区并重启定时器。"""
        if not self._running.is_set():
            return
        self._submit_current_buffer()
        # 重启定时器
        self._start_segment_timer()

    def _submit_current_buffer(self) -> None:
        """取出当前缓冲区并提交到转录队列（不改变录音状态）。"""
        with self._buffer_lock:
            if not self._buffer:
                return
            try:
                combined = np.concatenate(self._buffer, axis=0)
                self._buffer.clear()
                self._session_bytes = 0
            except Exception as exc:
                logger.error("分段合并音频缓冲区时出错: %s", exc)
                self._buffer.clear()
                return

        if combined.size == 0:
            return

        try:
            self._transcription_queue.put_nowait(combined)
            with self._state_lock:
                self._transcription_task_count += 1
                task_count = self._transcription_task_count
            logger.info(
                "分段自动提交转录（任务 #%s），队列中有 %s 个待处理任务",
                task_count,
                self._transcription_queue.qsize(),
            )
        except queue.Full:
            logger.warning("转录队列已满，分段提交跳过")

    def stop(self, _from_capture_thread: bool = False) -> None:
        """停止录音并提交转录任务

        Args:
            _from_capture_thread: 内部参数，标识是否从capture线程调用（避免死锁）
        """
        # 取消分段定时器
        if self._segment_timer is not None:
            self._segment_timer.cancel()
            self._segment_timer = None

        # 恢复原始设备（start_with_device 可能切换了设备）
        if self.audio.device != self._original_device:
            self.audio.device = self._original_device

        # 第一阶段：在锁内快速更新状态并保存资源引用
        with self._state_lock:
            if not self._running.is_set():
                logger.debug("Transcription worker 未运行，忽略 stop")
                return

            session_id = self._current_session_id
            reason = "size_limit" if self._session_bytes >= self._max_session_bytes else "user"
            logger.info("Transcription worker stopping (session_id=%s, reason=%s)", session_id, reason)
            self._stop_requested.set()
            self._running.clear()
            self._recording.clear()
            
            # 保存当前会话的线程引用，避免操作到新会话的线程
            capture_thread_to_join = self._capture_thread
            # 清空线程引用，允许新会话创建新线程
            self._capture_thread = None
        
        # 第二阶段：在锁外执行耗时操作
        self.audio.stop()
        
        # 只有从外部调用时才join capture线程，避免自己join自己
        # 使用保存的线程引用，而不是self._capture_thread
        if not _from_capture_thread:
            if capture_thread_to_join and capture_thread_to_join.is_alive():
                capture_thread_to_join.join(timeout=5)

        combined = self._combine_buffer()
        self.audio.flush()

        if combined is None or combined.size == 0:
            logger.warning("未捕获到任何音频样本，跳过转写 (session_id=%s)", session_id)
            with self._state_lock:
                self._current_session_id = None
            return

        # 将音频数据提交到转录队列，立即返回（异步处理）
        try:
            self._transcription_queue.put_nowait(combined)
            # 更新计数器时需要锁保护
            with self._state_lock:
                self._transcription_task_count += 1
                task_count = self._transcription_task_count
            logger.info(
                "录音已提交到转录队列（session_id=%s，任务 #%s），队列中有 %s 个待处理任务",
                session_id,
                task_count,
                self._transcription_queue.qsize(),
            )
        except queue.Full:
            logger.error("转录队列已满，无法提交新任务 (session_id=%s)！请等待当前转录完成。", session_id)
            # 即使队列满了，也不阻塞用户，只是记录错误
        
        # 最后清理session_id
        with self._state_lock:
            self._current_session_id = None

    def _capture_loop(self) -> None:
        queue_obj = self.audio.queue
        while self._recording.is_set():
            try:
                frame = queue_obj.get(timeout=0.2)
            except Exception:
                if not self._recording.is_set():
                    break
                continue

            try:
                with self._buffer_lock:
                    if isinstance(frame, np.ndarray):
                        self._buffer.append(frame)
                        bytes_added = frame.nbytes
                    else:
                        arr = np.frombuffer(frame, dtype=np.int16)
                        self._buffer.append(arr)
                        bytes_added = arr.nbytes
                    self._session_bytes += bytes_added
            except Exception as exc:
                logger.error("处理音频帧时出错: %s", exc)

            # 达到单次会话大小上限后，自动停止录音
            if self._session_bytes >= self._max_session_bytes and not self._stop_requested.is_set():
                logger.warning(
                    "单次录音大小达到上限，自动停止（%s/%s 字节，%.2f/%.2f MB）",
                    self._session_bytes,
                    self._max_session_bytes,
                    self._session_bytes / (1024 * 1024),
                    self._max_session_bytes / (1024 * 1024),
                )
                # 从capture线程调用stop，传入标志避免死锁
                self.stop(_from_capture_thread=True)
                break  # 停止后立即退出循环

        with self._buffer_lock:
            frame_count = len(self._buffer)
        logger.debug("capture loop exiting, collected %s frames", frame_count)

    def _combine_buffer(self) -> Optional[np.ndarray]:
        with self._buffer_lock:
            if not self._buffer:
                return None
            try:
                combined = np.concatenate(self._buffer, axis=0)
                logger.info("会话录音合并完成，总样本数=%s", combined.size)
                self._buffer.clear()
                return combined
            except Exception as exc:
                logger.error("合并音频缓冲区时出错: %s", exc)
                self._buffer.clear()  # 即使出错也清理缓冲区
                return None

    def _write_temp_wav(self, samples: np.ndarray) -> str:
        import wave

        sample_rate = self._audio_cfg["sample_rate"]
        self._write_recent_wav(samples)

        fd, path = tempfile.mkstemp(prefix="asr_session_", suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(samples.tobytes())

        return path

    def _write_recent_wav(self, samples: np.ndarray) -> None:
        """将最近一次录音保存为 recent.wav（供调试用），更新 last_segment_path。"""
        import wave

        sample_rate = self._audio_cfg["sample_rate"]
        recent_path = Path(self.log_dir) / "recent.wav"
        os.makedirs(recent_path.parent, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix="recent_", suffix=".wav", dir=recent_path.parent
        )
        os.close(tmp_fd)
        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(samples.tobytes())
        os.replace(tmp_path, recent_path)
        self.last_segment_path = recent_path

    def _transcribe_once(self, samples: np.ndarray) -> None:
        if self._backend == "volcengine":
            self._transcribe_once_volcengine(samples)
        else:
            self._transcribe_once_funasr(samples)

    def _transcribe_once_funasr(self, samples: np.ndarray) -> None:
        """使用本地 FunASR 进行转录。"""
        tmp_path = self._write_temp_wav(samples)
        start = time.time()
        try:
            asr_result = self.fun_server.transcribe_audio(
                tmp_path,
                options=self.config.get("asr"),
            )

        finally:
            inference_latency = time.time() - start
            try:
                os.remove(tmp_path)
            except OSError:
                logger.debug("删除临时文件失败: %s", tmp_path)

        self._dispatch_result(asr_result, inference_latency)

    def _transcribe_once_volcengine(self, samples: np.ndarray) -> None:
        """使用 Volcengine BigASR 流式识别进行转录。"""
        # 保存 recent.wav 供调试（与 FunASR 路径保持一致）
        self._write_recent_wav(samples)

        sample_rate = self._audio_cfg["sample_rate"]
        # 仅传递影响识别行为的选项，不包含凭据字段
        volcengine_cfg = self.config.get("volcengine", {})
        transcribe_options = {
            k: volcengine_cfg[k]
            for k in ("enable_punc", "enable_itn")
            if k in volcengine_cfg
        }
        start = time.time()
        asr_result = self._volcengine_client.transcribe(
            samples,
            sample_rate=sample_rate,
            options=transcribe_options,
        )
        inference_latency = time.time() - start

        self._dispatch_result(asr_result, asr_result.get("inference_latency", inference_latency))

    def transcribe_file(self, file_path: str) -> None:
        """加载音频文件，分片提交到转录队列（F4 文件识别用）。"""
        import os
        if not os.path.isfile(file_path):
            logger.error("音频文件不存在: %s", file_path)
            return

        sample_rate = self._audio_cfg["sample_rate"]
        segment_samples = self._segment_seconds * sample_rate

        try:
            import librosa
            samples, _ = librosa.load(file_path, sr=sample_rate, mono=True)
        except Exception as exc:
            logger.error("加载音频文件失败: %s — %s", file_path, exc)
            return

        # float32 → int16
        samples_int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        total_samples = len(samples_int16)
        logger.info("音频文件加载完成: %s（%d 样本，%.1f 秒）", file_path, total_samples, total_samples / sample_rate)

        # 分片提交（阻塞等待队列有空位，避免丢弃分片）
        offset = 0
        chunk_index = 0
        while offset < total_samples:
            end = min(offset + segment_samples, total_samples)
            chunk = samples_int16[offset:end]
            chunk_index += 1
            try:
                self._transcription_queue.put(chunk, timeout=300)
                with self._state_lock:
                    self._transcription_task_count += 1
                if chunk_index % 50 == 0 or end >= total_samples:
                    logger.info("文件分片进度: %d/%d（样本 %d~%d），队列中有 %s 个待处理任务",
                                chunk_index, (total_samples + segment_samples - 1) // segment_samples,
                                offset, end, self._transcription_queue.qsize())
            except queue.Full:
                logger.error("转录队列超时（300秒），文件分片提交失败（样本 %d~%d）", offset, end)
                break
            offset = end

    def _dispatch_result(self, asr_result: dict, inference_latency: float) -> None:
        """将 ASR 结果包装成 TranscriptionResult 并回调。"""
        if not asr_result.get("success"):
            result = TranscriptionResult(
                text="",
                raw_text="",
                duration=0.0,
                inference_latency=inference_latency,
                confidence=0.0,
                error=asr_result.get("error", "unknown"),
            )
        else:
            result = TranscriptionResult(
                text=asr_result.get("text", ""),
                raw_text=asr_result.get("raw_text", ""),
                duration=asr_result.get("duration", 0.0),
                inference_latency=inference_latency,
                confidence=asr_result.get("confidence", 0.0),
            )

        if self.on_result:
            try:
                self.on_result(result)
            except Exception as exc:  # noqa: BLE001
                logger.error("处理转写结果时出错: %s", exc)

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def is_transcribing(self) -> bool:
        """是否有转录任务正在进行或等待中"""
        return not self._transcription_queue.empty()

    @property
    def pending_transcriptions(self) -> int:
        """返回队列中等待转录的任务数"""
        return self._transcription_queue.qsize()

    @property
    def transcription_stats(self) -> dict:
        """返回转录统计信息"""
        return {
            "submitted": self._transcription_task_count,
            "completed": self._transcription_completed_count,
            "pending": self.pending_transcriptions,
            "is_recording": self._running.is_set(),
            "is_transcribing": self.is_transcribing,
        }


