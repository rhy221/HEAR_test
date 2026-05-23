import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import AgentResult
from .coordinator import CoordinatorAgent
from .textual import TextualAgent
from .visual import VisualAgent
from .verification import cross_modal_verify, CONTRADICTORY, CONSISTENT, UNCERTAIN

logger = logging.getLogger(__name__)


class QuestionResult:
    def __init__(self):
        self.answer: str = ""
        self.evidence_pages: List[int] = [0]  # 0-indexed
        self.verification_status: str = UNCERTAIN
        self.agent_claims: Dict = {}
        self.reeval_rounds: int = 0
        self.error: Optional[str] = None


def _load_prompt(prompts_dir: str, filename: str) -> str:
    path = Path(prompts_dir) / filename
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _synthesize_answer(
    text_result: AgentResult,
    visual_result: AgentResult,
    verification_status: str,
    question: str,
    lang: str,
    llm_client: Any,
    cfg: Any,
) -> str:
    prompt_template = _load_prompt(cfg.prompts.dir, cfg.prompts.synthesis)

    branch = (
        "FULL_AGREEMENT" if verification_status == CONSISTENT else
        "EXPLICIT_CONFLICT" if verification_status == CONTRADICTORY else
        "PARTIAL_AGREEMENT"
    )

    prompt = prompt_template.format(
        question=question,
        text_claim=text_result.claim,
        visual_claim=visual_result.claim,
        branch=branch,
    ) if "{question}" in prompt_template else (
        f"{prompt_template}\n\n"
        f"Branch: {branch}\n"
        f"Question: {question}\n"
        f"Text claim: {text_result.claim}\n"
        f"Visual claim: {visual_result.claim}"
    )

    system_msg = cfg.prompts.system_message.get(lang, "")
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    agent_cfg = cfg.agents.synthesis
    try:
        result = llm_client.chat(
            messages=messages,
            max_tokens=agent_cfg.max_tokens,
            temperature=agent_cfg.temperature,
        )
        return _extract_final_answer(result.content)
    except Exception as e:
        logger.warning("Synthesis failed: %s", e)
        return text_result.claim or visual_result.claim or ""


def _extract_final_answer(content: str) -> str:
    import re
    match = re.search(r"(?:final\s+answer|answer)[:\s]+(.+?)(?:\n|$)", content, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
    return lines[-1] if lines else content.strip()


def _generate_subquery(
    question: str,
    text_claim: str,
    visual_claim: str,
    lang: str,
    llm_client: Any,
    cfg: Any,
) -> str:
    prompt_template = _load_prompt(cfg.prompts.dir, cfg.prompts.subquery_generation)
    prompt = prompt_template.format(
        question=question,
        text_claim=text_claim,
        visual_claim=visual_claim,
    ) if "{question}" in prompt_template else (
        f"{prompt_template}\n\nOriginal question: {question}\n"
        f"Text claim: {text_claim}\nVisual claim: {visual_claim}"
    )

    system_msg = cfg.prompts.system_message.get(lang, "")
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    try:
        result = llm_client.chat(messages=messages, max_tokens=256, temperature=0.0)
        return result.content.strip()
    except Exception as e:
        logger.warning("Subquery generation failed: %s", e)
        return question


class Orchestrator:
    def __init__(self, cfg: Any, llm_client_factory: Callable):
        self.cfg = cfg
        self.mode = cfg.agents.mode
        self._vlm_client = llm_client_factory("vlm")
        self._text_client = (
            llm_client_factory("text")
            if self.mode == "separate_text_model"
            else self._vlm_client
        )

        self.coordinator = CoordinatorAgent(self._vlm_client, cfg)
        self.textual_agent = TextualAgent(self._text_client, cfg)
        self.visual_agent = VisualAgent(self._vlm_client, cfg)

    def process_question(
        self,
        question_meta: dict,
        pages: List[Any],         # List[ReconciledPage]
        retrieved_page_indices: List[int],  # 0-indexed
        retrieval_scores: List[float],
        retrieve_fn: Optional[Callable] = None,
    ) -> QuestionResult:
        qr = QuestionResult()
        question = question_meta["question"]
        lang = question_meta.get("_lang", "ja")

        # Build context for agents
        def build_context(page_indices):
            text_evidence = "\n\n".join(
                f"[Page {pages[i].page_num_1}]\n" + "\n".join(pages[i].text_chunks)
                for i in page_indices if i < len(pages)
            )
            image_paths = [
                pages[i].image_path for i in page_indices if i < len(pages)
            ]
            pages_summary = "\n".join(
                f"Page {pages[i].page_num_1}: {pages[i].raw_text[:200]}"
                for i in page_indices if i < len(pages)
            )
            return {
                "question": question,
                "lang": lang,
                "retrieved_pages": page_indices,
                "text_evidence": text_evidence,
                "image_paths": image_paths,
                "pages_summary": pages_summary,
            }

        context = build_context(retrieved_page_indices)

        # Step 1: Coordinator triage
        coord_result = self.coordinator.run(context)
        assigned = coord_result.evidence_pages or retrieved_page_indices
        context = build_context(assigned)

        def run_round(page_indices, reeval=False) -> tuple:
            ctx = build_context(page_indices)
            text_r = self.textual_agent.run(ctx)
            visual_r = self.visual_agent.run(ctx)
            verif = cross_modal_verify(text_r, visual_r, self._vlm_client, lang, self.cfg)
            return text_r, visual_r, verif

        text_result, visual_result, verification = run_round(assigned)

        # Step 2: Conflict-driven re-evaluation
        reeval_rounds = 0
        if (
            verification.status == CONTRADICTORY
            and self.cfg.agents.enable_conflict_reeval
            and self.cfg.agents.max_reeval_rounds > 0
            and retrieve_fn is not None
        ):
            subquery = _generate_subquery(
                question, text_result.claim, visual_result.claim, lang,
                self._vlm_client, self.cfg
            )
            new_indices = retrieve_fn(subquery)
            if new_indices and new_indices != assigned:
                text_result, visual_result, verification = run_round(new_indices)
                reeval_rounds = 1
                logger.info("Re-evaluation round completed for question: %s", question_meta.get("id"))

        # Step 3: Synthesis
        answer = _synthesize_answer(
            text_result, visual_result, verification.status,
            question, lang, self._vlm_client, self.cfg
        )

        # Evidence pages: consensus 0-indexed from verification
        evidence_0idx = verification.merged_pages or assigned
        evidence_0idx = sorted(set(evidence_0idx))

        qr.answer = answer
        qr.evidence_pages = evidence_0idx
        qr.verification_status = verification.status
        qr.agent_claims = {
            "coordinator": coord_result.claim,
            "textual": text_result.claim,
            "visual": visual_result.claim,
            "verification_reasoning": verification.reasoning,
        }
        qr.reeval_rounds = reeval_rounds
        return qr
