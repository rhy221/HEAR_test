import json
from pathlib import Path
from typing import Any, List, Optional


def load_questions(path: str, cfg_fields: Any) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Handle {questions: [...]} wrapper
        data = data.get("questions", data.get("data", list(data.values())[0]))

    questions = []
    for item in data:
        q = {
            "id": item.get(cfg_fields.id, item.get("id")),
            "question": item.get(cfg_fields.question, item.get("question", "")),
            "pdf_filename": item.get(cfg_fields.pdf_filename, item.get("pdf", "")),
            "answer_type": item.get(cfg_fields.answer_type, item.get("answer_type", None)),
            "language": item.get(cfg_fields.language, item.get("language", None)),
        }
        questions.append(q)
    return questions


def get_pdf_path(pdf_dir: str, pdf_filename: str) -> Path:
    pdf_dir = Path(pdf_dir)
    candidate = pdf_dir / pdf_filename
    if candidate.exists():
        return candidate
    # Try without directory component in filename
    candidate2 = pdf_dir / Path(pdf_filename).name
    if candidate2.exists():
        return candidate2
    return candidate  # Return even if not found; caller handles missing


def apply_sample_limit(questions: List[dict], n: Optional[int]) -> List[dict]:
    if n is None or n <= 0:
        return questions
    return questions[:n]
