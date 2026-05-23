import logging
from pathlib import Path
from typing import Any, Dict, List

from .base import Agent, AgentResult

logger = logging.getLogger(__name__)


class CoordinatorAgent(Agent):
    name = "coordinator"

    def _load_prompt(self) -> str:
        path = Path(self.cfg.prompts.dir) / self.cfg.prompts.coordinator_triage
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def run(self, context: Dict[str, Any]) -> AgentResult:
        if getattr(self.cfg.ablation, "disable_coordinator_agent", False):
            return AgentResult(
                claim="",
                evidence_pages=context.get("retrieved_pages", []),
                status="Uncertain",
            )

        question = context["question"]
        lang = context["lang"]
        pages_summary = context.get("pages_summary", "")
        retrieved_pages = context.get("retrieved_pages", [])

        prompt_template = self._load_prompt()
        prompt = prompt_template.format(
            question=question,
            pages_summary=pages_summary,
        ) if "{question}" in prompt_template else (
            f"{prompt_template}\n\nQuestion: {question}\n\nEvidence pages summary:\n{pages_summary}"
        )

        system_msg = self.cfg.prompts.system_message.get(lang, "")
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        agent_cfg = self.cfg.agents.coordinator
        try:
            result = self.llm_client.chat(
                messages=messages,
                max_tokens=agent_cfg.max_tokens,
                temperature=agent_cfg.temperature,
            )
            triage = _parse_triage(result.content, retrieved_pages)
            return AgentResult(
                claim=result.content,
                evidence_pages=triage.get("assigned_pages", retrieved_pages),
                status="Uncertain",
                raw_response=result.content,
                token_usage={"prompt": result.prompt_tokens, "completion": result.completion_tokens},
                latency_s=result.latency_s,
            )
        except Exception as e:
            logger.warning("Coordinator failed: %s", e)
            return AgentResult(claim="", evidence_pages=retrieved_pages, status="Uncertain")


def _parse_triage(content: str, fallback_pages: List[int]) -> Dict:
    import re
    pages_match = re.search(r"pages?\s*[:=]\s*\[?([\d,\s]+)\]?", content, re.IGNORECASE)
    if pages_match:
        try:
            pages = [int(x.strip()) for x in pages_match.group(1).split(",") if x.strip().isdigit()]
            return {"assigned_pages": pages or fallback_pages}
        except Exception:
            pass
    return {"assigned_pages": fallback_pages}
