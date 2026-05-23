import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .base import AgentResult

logger = logging.getLogger(__name__)

CONSISTENT = "Consistent"
CONTRADICTORY = "Contradictory"
UNCERTAIN = "Uncertain"


@dataclass
class VerificationResult:
    status: str           # Consistent | Contradictory | Uncertain
    reasoning: str = ""
    merged_claim: str = ""
    merged_pages: list = None

    def __post_init__(self):
        if self.merged_pages is None:
            self.merged_pages = []


def cross_modal_verify(
    text_result: AgentResult,
    visual_result: AgentResult,
    llm_client: Any,
    lang: str,
    cfg: Any,
) -> VerificationResult:
    if getattr(cfg.ablation, "disable_cross_modal_verification", False) or \
       not cfg.agents.enable_cross_verification:
        merged_pages = sorted(set(text_result.evidence_pages + visual_result.evidence_pages))
        return VerificationResult(
            status=UNCERTAIN,
            merged_claim=text_result.claim or visual_result.claim,
            merged_pages=merged_pages,
        )

    prompt_template = _load_prompt(cfg)
    prompt = prompt_template.format(
        text_claim=text_result.claim,
        visual_claim=visual_result.claim,
    ) if "{text_claim}" in prompt_template else (
        f"{prompt_template}\n\n"
        f"Text agent claim: {text_result.claim}\n"
        f"Visual agent claim: {visual_result.claim}"
    )

    system_msg = cfg.prompts.system_message.get(lang, "")
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    try:
        result = llm_client.chat(messages=messages, max_tokens=512, temperature=0.0)
        status = _parse_status(result.content)
        merged_pages = sorted(set(text_result.evidence_pages + visual_result.evidence_pages))
        return VerificationResult(
            status=status,
            reasoning=result.content,
            merged_claim=text_result.claim if status != CONTRADICTORY else "",
            merged_pages=merged_pages,
        )
    except Exception as e:
        logger.warning("cross_modal_verify failed: %s", e)
        merged_pages = sorted(set(text_result.evidence_pages + visual_result.evidence_pages))
        return VerificationResult(status=UNCERTAIN, merged_pages=merged_pages)


def _load_prompt(cfg: Any) -> str:
    path = Path(cfg.prompts.dir) / cfg.prompts.cross_verification
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _parse_status(content: str) -> str:
    content_lower = content.lower()
    if "consistent" in content_lower:
        return CONSISTENT
    if "contradict" in content_lower or "conflict" in content_lower:
        return CONTRADICTORY
    return UNCERTAIN
