import csv
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ResultStore:
    def __init__(self):
        self._data: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def store(self, question_id: str, result: dict) -> None:
        with self._lock:
            self._data[question_id] = result

    def get(self, question_id: str) -> Optional[dict]:
        with self._lock:
            return self._data.get(question_id)

    def all_results(self) -> Dict[str, dict]:
        with self._lock:
            return dict(self._data)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def dump_submission_csv(
    results: Dict[str, dict],
    questions_order: List[dict],
    cfg: Any,
) -> Path:
    output_path = Path(str(cfg.output.submission_csv))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fallback_answer = cfg.output.submission.fallback_answer
    fallback_evidence = cfg.output.submission.fallback_evidence

    rows = []
    for q in questions_order:
        qid = q["id"]
        r = results.get(qid)
        if r is None:
            rows.append({
                "id": qid,
                "answer": fallback_answer,
                "evidence_page_number": _format_evidence_list(fallback_evidence),
            })
        else:
            answer = r.get("predicted_answer", r.get("answer", fallback_answer))
            predicted_pages = r.get("predicted_pages")
            if predicted_pages is not None and isinstance(predicted_pages, list):
                evidence_str = _format_evidence_list(predicted_pages)
            else:
                evidence_str = r.get("evidence_page_number", _format_evidence_list(fallback_evidence))
            rows.append({"id": qid, "answer": answer, "evidence_page_number": evidence_str})

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer", "evidence_page_number"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote submission CSV: %s (%d rows)", output_path, len(rows))
    return output_path


def dump_results_json(
    results: Dict[str, dict],
    questions_order: List[dict],
    cfg: Any,
) -> Path:
    output_path = Path(str(cfg.output.results_json))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [results[q["id"]] for q in questions_order if q["id"] in results]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=str)

    logger.info("Wrote results JSON: %s (%d entries)", output_path, len(rows))
    return output_path


def _format_evidence_list(pages: List[int]) -> str:
    parts = ", ".join(f"'{p}'" for p in pages)
    return f"[{parts}]"
