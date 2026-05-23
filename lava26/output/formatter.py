import ast
import re
from typing import Any, List, Optional, Union


_LIST_INDICATORS = re.compile(
    r"\b(list|multiple|all|each|enumerate|いくつ|すべて|tất cả|các|danh sách)\b",
    re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")
_ORDERED_INDICATORS = re.compile(
    r"\b(order|rank|sequence|step|first|second|third|順|順番|thứ tự|bước)\b",
    re.IGNORECASE,
)


def detect_answer_type(question_meta: dict, question_text: str) -> str:
    explicit = question_meta.get("answer_type")
    if explicit and explicit in ("string", "number", "unordered_list", "ordered_list"):
        return explicit

    q = question_text.lower()
    if _LIST_INDICATORS.search(q):
        if _ORDERED_INDICATORS.search(q):
            return "ordered_list"
        return "unordered_list"
    return "string"


def format_answer(raw_answer: Any, answer_type: str) -> str:
    if answer_type in ("unordered_list", "ordered_list"):
        items = _to_list(raw_answer)
        return _python_list_repr(items)

    if answer_type == "number":
        s = str(raw_answer).strip()
        if _NUMBER_PATTERN.match(s):
            try:
                v = int(s) if "." not in s else float(s)
                return str(v)
            except ValueError:
                pass
        return s

    return str(raw_answer).strip()


def format_evidence(page_indices_0: List[int]) -> str:
    """Convert 0-indexed page list to 1-indexed Python list repr string."""
    pages_1 = sorted(set(p + 1 for p in page_indices_0))
    return _python_list_repr(pages_1)


def _to_list(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("["):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        # Newline-separated or comma-separated
        if "\n" in stripped:
            return [l.strip(" -•*") for l in stripped.splitlines() if l.strip()]
        if "," in stripped:
            return [p.strip() for p in stripped.split(",") if p.strip()]
        return [stripped]
    return [str(raw)]


def _python_list_repr(items: List) -> str:
    parts = ", ".join(f"'{str(x)}'" for x in items)
    return f"[{parts}]"
