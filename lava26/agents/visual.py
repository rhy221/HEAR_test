import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Agent, AgentResult

logger = logging.getLogger(__name__)


class VisualAgent(Agent):
    name = "visual"

    def _load_prompt(self) -> str:
        path = Path(self.cfg.prompts.dir) / self.cfg.prompts.visual_agent
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def run(self, context: Dict[str, Any]) -> AgentResult:
        if getattr(self.cfg.ablation, "disable_image_agent", False):
            return AgentResult(claim="", evidence_pages=[], status="Uncertain")

        question = context["question"]
        lang = context["lang"]
        image_paths = context.get("image_paths", [])
        retrieved_pages = context.get("retrieved_pages", [])

        images_b64 = _load_images_b64(image_paths)

        prompt_template = self._load_prompt()
        prompt = prompt_template.format(question=question) if "{question}" in prompt_template else (
            f"{prompt_template}\n\nQuestion: {question}"
        )

        system_msg = self.cfg.prompts.system_message.get(lang, "")
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        agent_cfg = self.cfg.agents.visual
        try:
            result = self.llm_client.chat(
                messages=messages,
                images=images_b64 if images_b64 else None,
                max_tokens=agent_cfg.max_tokens,
                temperature=agent_cfg.temperature,
            )
            from .textual import _parse_agent_output
            parsed = _parse_agent_output(result.content, retrieved_pages)
            return AgentResult(
                claim=parsed["claim"],
                evidence_pages=parsed["pages"],
                status="Uncertain",
                raw_response=result.content,
                token_usage={"prompt": result.prompt_tokens, "completion": result.completion_tokens},
                latency_s=result.latency_s,
            )
        except Exception as e:
            logger.warning("VisualAgent failed: %s", e)
            return AgentResult(claim="", evidence_pages=retrieved_pages, status="Uncertain")


def _load_images_b64(image_paths: List[Optional[Path]]) -> List[str]:
    b64_list = []
    for p in image_paths:
        if p is None:
            continue
        path = Path(p)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    b64_list.append(base64.b64encode(f.read()).decode())
            except Exception as e:
                logger.warning("Failed to load image %s: %s", path, e)
    return b64_list
