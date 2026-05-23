import logging
import re
from pathlib import Path
from typing import Any, List

logger = logging.getLogger(__name__)

_HTML_TABLE_OPEN = re.compile(r"<table", re.IGNORECASE)
_HEADER_LESS_START = re.compile(r"^\s*<tr", re.IGNORECASE)


def _looks_like_headerless_table(text: str) -> bool:
    stripped = text.strip()
    return bool(_HEADER_LESS_START.match(stripped)) or (
        stripped.startswith("<table") and "<th" not in stripped.lower()
    )


def _load_prompt(prompts_dir: str, filename: str) -> str:
    path = Path(prompts_dir) / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    logger.warning("Prompt file not found: %s", path)
    return ""


def _vlm_should_merge(prev_table: str, next_content: str, llm_client: Any, cfg: Any) -> bool:
    prompt_template = _load_prompt(cfg.prompts.dir, cfg.prompts.table_merge)
    prompt = prompt_template.format(
        prev_table=prev_table[:2000],
        next_content=next_content[:1000],
    ) if "{prev_table}" in prompt_template else (
        f"{prompt_template}\n\nPrevious table:\n{prev_table[:2000]}\n\nNext page start:\n{next_content[:1000]}"
    )

    messages = [{"role": "user", "content": prompt}]
    try:
        result = llm_client.chat(messages=messages, max_tokens=64, temperature=0.0)
        answer = result.content.strip().lower()
        return "yes" in answer or "merge" in answer or "true" in answer
    except Exception as e:
        logger.warning("Table merge VLM call failed: %s", e)
        return False


def merge_cross_page_tables(
    pages: List[Any],  # List[ReconciledPage]
    llm_client: Any,
    cfg: Any,
) -> List[Any]:
    if not getattr(cfg.parsing, "cross_page_table", True) or getattr(cfg.ablation, "disable_cross_page_table", False):
        return pages

    max_span = getattr(cfg.parsing, "cross_page_table_max_span", 5)
    pages = list(pages)

    i = 0
    while i < len(pages):
        page = pages[i]
        if not page.tables_html:
            i += 1
            continue

        last_table = page.tables_html[-1]
        merged_count = 0

        j = i + 1
        while j < len(pages) and merged_count < max_span:
            next_page = pages[j]
            # Check if next page starts with continuation table content
            next_text = "\n".join(next_page.text_chunks[:2]) if next_page.text_chunks else ""
            if not _looks_like_headerless_table(next_text) and not any(
                _looks_like_headerless_table(t) for t in next_page.tables_html
            ):
                break

            continuation = next_page.tables_html[0] if next_page.tables_html else next_text
            if _vlm_should_merge(last_table, continuation, llm_client, cfg):
                # Merge: append continuation rows into last_table
                merged_table = last_table.rstrip().rstrip("</table>").rstrip() + "\n" + continuation
                if not merged_table.strip().endswith("</table>"):
                    merged_table += "\n</table>"
                page.tables_html[-1] = merged_table
                last_table = merged_table

                # Remove consumed content from next page
                if next_page.tables_html:
                    next_page.tables_html.pop(0)
                elif next_page.text_chunks:
                    next_page.text_chunks.pop(0)

                merged_count += 1
                logger.debug("Merged table from page %d into page %d", j, i)
            else:
                break
            j += 1

        i += 1

    return pages
