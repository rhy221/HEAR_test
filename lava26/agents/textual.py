import logging
from pathlib import Path
from typing import Any, Dict, List

from .base import Agent, AgentResult

logger = logging.getLogger(__name__)


class TextualAgent(Agent):
    name = "textual"

    def _load_prompt(self) -> str:
        path = Path(self.cfg.prompts.dir) / self.cfg.prompts.textual_agent
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def run(self, context: Dict[str, Any]) -> AgentResult:
        if getattr(self.cfg.ablation, "disable_text_agent", False):
            return AgentResult(claim="", evidence_pages=[], status="Uncertain")

        question = context["question"]
        lang = context["lang"]
        text_evidence = context.get("text_evidence", "")
        retrieved_pages = context.get("retrieved_pages", [])

        prompt_template = self._load_prompt()
        prompt = prompt_template.format(
            question=question,
            text_evidence=text_evidence,
        ) if "{question}" in prompt_template else (
            f"{prompt_template}\n\nQuestion: {question}\n\nText evidence:\n{text_evidence}"
        )

        system_msg = self.cfg.prompts.system_message.get(lang, "")
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        agent_cfg = self.cfg.agents.textual
        try:
            result = self.llm_client.chat(
                messages=messages,
                max_tokens=agent_cfg.max_tokens,
                temperature=agent_cfg.temperature,
            )
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
            logger.warning("TextualAgent failed: %s", e)
            return AgentResult(claim="", evidence_pages=retrieved_pages, status="Uncertain")


def _parse_agent_output(content: str, fallback_pages: List[int]) -> Dict:
    import re
    claim_match = re.search(r"(?:claim|answer)[:\s]+(.+?)(?:\n|evidence|page)", content, re.IGNORECASE | re.DOTALL)
    claim = claim_match.group(1).strip() if claim_match else content.strip()[:500]

    pages_match = re.search(r"(?:evidence[_\s]?pages?|pages?)[:\s]*\[?([\d,\s]+)\]?", content, re.IGNORECASE)
    if pages_match:
        try:
            pages = [int(x.strip()) for x in pages_match.group(1).split(",") if x.strip().isdigit()]
            return {"claim": claim, "pages": pages or fallback_pages}
        except Exception:
            pass
    return {"claim": claim, "pages": fallback_pages}
