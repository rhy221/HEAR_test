import logging
import os
import subprocess
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_WARN_SHOWN = False


class ModelManager:
    """Manages sequential load/unload of VLM and text models for Mode B.

    Mode B warning: loading models sequentially on A100 40GB incurs significant
    overhead (model load time ~2-5 min each swap). Use Mode A (shared_vlm) unless
    you specifically need separate text model reasoning.
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._vlm_proc: Optional[subprocess.Popen] = None
        self._text_proc: Optional[subprocess.Popen] = None
        self._vlm_port = int(cfg.model.qwen_vl.vllm_url.split(":")[-1].split("/")[0])
        self._text_port = int(cfg.model.qwen_text.vllm_url.split(":")[-1].split("/")[0])
        _warn_mode_b()

    def load_vlm(self) -> None:
        if self._vlm_proc is not None:
            return
        logger.info("ModelManager: loading VLM server on port %d", self._vlm_port)
        vlm_cfg = self.cfg.model.qwen_vl
        cmd = _build_vllm_cmd(str(vlm_cfg.path), self._vlm_port, vlm_cfg)
        self._vlm_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _wait_for_server(self._vlm_port, timeout=300)
        logger.info("VLM server ready on port %d", self._vlm_port)

    def unload_vlm(self) -> None:
        if self._vlm_proc is None:
            return
        logger.info("ModelManager: unloading VLM server")
        self._vlm_proc.terminate()
        try:
            self._vlm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._vlm_proc.kill()
        self._vlm_proc = None
        time.sleep(2)  # Allow GPU memory to be freed

    def load_text_model(self) -> None:
        if self._text_proc is not None:
            return
        logger.info("ModelManager: loading text model server on port %d", self._text_port)
        text_cfg = self.cfg.model.qwen_text
        cmd = _build_vllm_cmd(str(text_cfg.path), self._text_port, text_cfg)
        self._text_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _wait_for_server(self._text_port, timeout=300)
        logger.info("Text model server ready on port %d", self._text_port)

    def unload_text_model(self) -> None:
        if self._text_proc is None:
            return
        logger.info("ModelManager: unloading text model server")
        self._text_proc.terminate()
        try:
            self._text_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._text_proc.kill()
        self._text_proc = None
        time.sleep(2)

    def cleanup(self) -> None:
        self.unload_vlm()
        self.unload_text_model()


def _build_vllm_cmd(model_path: str, port: int, model_cfg: Any) -> list:
    cmd = [
        "vllm", "serve", model_path,
        "--port", str(port),
        "--dtype", str(model_cfg.dtype),
        "--max-model-len", str(model_cfg.max_model_len),
        "--gpu-memory-utilization", str(model_cfg.gpu_memory_utilization),
    ]
    if model_cfg.quantization:
        cmd += ["--quantization", model_cfg.quantization]
    if getattr(model_cfg, "enforce_eager", False):
        cmd.append("--enforce-eager")
    return cmd


def _wait_for_server(port: int, timeout: int = 300) -> None:
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            return
        except Exception:
            time.sleep(3)
    raise TimeoutError(f"vLLM server on port {port} did not start within {timeout}s")


def _warn_mode_b() -> None:
    global _WARN_SHOWN
    if not _WARN_SHOWN:
        logger.warning(
            "\n" + "!" * 60 +
            "\n  WARNING: agents.mode=separate_text_model selected." +
            "\n  This requires sequential model loading/unloading on A100." +
            "\n  Each swap takes ~2-5 minutes — significant overhead." +
            "\n  Recommend: use agents.mode=shared_vlm for A100 40GB." +
            "\n" + "!" * 60
        )
        _WARN_SHOWN = True
