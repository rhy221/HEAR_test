import base64
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class VLLMClient:
    def __init__(self, base_url: str, api_key: str, model_name: str, cfg: Any):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.request_timeout_s = getattr(cfg, "request_timeout_s", 120)
        self.max_retries = getattr(cfg, "max_retries", 2)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.request_timeout_s,
            )
        return self._client

    def chat(
        self,
        messages: List[dict],
        images: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: Optional[str] = None,
    ) -> ChatResult:
        client = self._get_client()

        prepared = list(messages)

        # Inject images into last user message if provided
        if images:
            for i in range(len(prepared) - 1, -1, -1):
                if prepared[i]["role"] == "user":
                    text_content = prepared[i]["content"]
                    content_parts = [{"type": "text", "text": text_content}]
                    for b64 in images:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        })
                    prepared[i] = {"role": "user", "content": content_parts}
                    break

        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                t0 = time.monotonic()
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=prepared,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                latency = time.monotonic() - t0

                content = response.choices[0].message.content or ""
                usage = response.usage
                result = ChatResult(
                    content=content,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    latency_s=latency,
                )
                logger.debug(
                    "VLLMClient chat: prompt=%d comp=%d latency=%.2fs",
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.latency_s,
                )
                return result
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    logger.warning("VLLMClient attempt %d failed: %s — retrying", attempt + 1, e)
                    time.sleep(1.0 * (attempt + 1))

        raise RuntimeError(f"VLLMClient failed after {self.max_retries + 1} attempts: {last_exc}")

    @classmethod
    def from_cfg(cls, model_cfg: Any) -> "VLLMClient":
        return cls(
            base_url=model_cfg.vllm_url,
            api_key=model_cfg.vllm_api_key,
            model_name=str(model_cfg.path).split("/")[-1],
            cfg=model_cfg,
        )
