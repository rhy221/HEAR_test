import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReconciledPage:
    page_num: int        # 0-indexed
    page_num_1: int      # 1-indexed
    text_chunks: List[str] = field(default_factory=list)
    tables_html: List[str] = field(default_factory=list)
    figure_captions: List[str] = field(default_factory=list)
    raw_text: str = ""
    image_path: Optional[Path] = None


def _load_prompt(prompts_dir: str, filename: str) -> str:
    path = Path(prompts_dir) / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    logger.warning("Prompt file not found: %s", path)
    return ""


def _image_to_b64(image_path: Path) -> Optional[str]:
    if image_path is None or not image_path.exists():
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def reconcile_page(
    page_data: Any,  # PageData
    llm_client: Any,
    lang: str,
    cfg: Any,
) -> ReconciledPage:
    if not getattr(cfg.parsing, "vlm_reconcile", True) or getattr(cfg.ablation, "disable_vlm_reconciliation", False):
        return ReconciledPage(
            page_num=page_data.page_num,
            page_num_1=page_data.page_num_1,
            text_chunks=[page_data.text] if page_data.text else [],
            raw_text=page_data.text,
            image_path=page_data.image_path,
        )

    prompt_file = cfg.prompts.reconcile.get(lang, cfg.prompts.reconcile.get("ja"))
    prompt_template = _load_prompt(cfg.prompts.dir, prompt_file)

    prompt = prompt_template.format(
        page_text=page_data.text or "(no text)",
    ) if "{page_text}" in prompt_template else prompt_template + f"\n\nPage text:\n{page_data.text}"

    images = None
    if page_data.image_path and page_data.image_path.exists():
        b64 = _image_to_b64(page_data.image_path)
        if b64:
            images = [b64]

    system_msg = cfg.prompts.system_message.get(lang, "")
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    try:
        result = llm_client.chat(
            messages=messages,
            images=images,
            max_tokens=getattr(cfg.parsing, "reconcile_max_tokens", 4096),
            temperature=getattr(cfg.parsing, "reconcile_temperature", 0.0),
        )
        content = result.content
    except Exception as e:
        logger.warning("VLM reconciliation failed for page %d: %s", page_data.page_num, e)
        content = page_data.text or ""

    # Parse VLM output: extract tables (```html blocks) and figure captions
    tables_html = _extract_html_tables(content)
    figure_captions = _extract_figure_captions(content)
    text_chunks = _extract_text_chunks(content, tables_html)

    return ReconciledPage(
        page_num=page_data.page_num,
        page_num_1=page_data.page_num_1,
        text_chunks=text_chunks,
        tables_html=tables_html,
        figure_captions=figure_captions,
        raw_text=content,
        image_path=page_data.image_path,
    )


def _extract_html_tables(content: str) -> List[str]:
    import re
    pattern = re.compile(r"```html\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    tables = []
    for m in pattern.finditer(content):
        t = m.group(1).strip()
        if "<table" in t.lower():
            tables.append(t)
    return tables


def _extract_figure_captions(content: str) -> List[str]:
    import re
    # Match lines starting with "Figure", "図", "Hình", "Bảng" etc.
    pattern = re.compile(r"^(?:Figure|図|Hình|Bảng|Table|Caption)[^\n]*", re.MULTILINE | re.IGNORECASE)
    return [m.group(0).strip() for m in pattern.finditer(content)]


def _extract_text_chunks(content: str, tables_html: List[str]) -> List[str]:
    import re
    # Remove HTML table blocks from content
    clean = re.sub(r"```html.*?```", "", content, flags=re.DOTALL | re.IGNORECASE)
    # Split into paragraphs
    chunks = [c.strip() for c in re.split(r"\n{2,}", clean) if c.strip()]
    return chunks or [content.strip()]
