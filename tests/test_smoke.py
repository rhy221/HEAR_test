"""End-to-end smoke test using synthetic fixtures and mocked LLM."""
import ast
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURES_DIR = Path(__file__).parent / "fixtures"
PDF_DIR = FIXTURES_DIR / "pdfs"
QUESTIONS_FILE = FIXTURES_DIR / "questions.json"


@pytest.fixture(scope="session", autouse=True)
def create_test_pdfs():
    """Create synthetic PDFs if they don't exist."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf_ja = PDF_DIR / "test_doc_ja.pdf"
    pdf_vi = PDF_DIR / "test_doc_vi.pdf"

    if not pdf_ja.exists() or not pdf_vi.exists():
        create_script = FIXTURES_DIR / "create_test_pdfs.py"
        if create_script.exists():
            import subprocess
            subprocess.run([sys.executable, str(create_script)], check=False)

    # Fallback: create minimal PDFs if still missing
    for path, content in [(pdf_ja, "Test JA PDF 2026"), (pdf_vi, "Test VI PDF 2026")]:
        if not path.exists():
            _write_minimal_pdf(path, content)


def _write_minimal_pdf(path: Path, text: str) -> None:
    content_bytes = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode()
    pdf = (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Contents 4 0 R/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>  >>endobj\n"
        b"4 0 obj<</Length " + str(len(content_bytes)).encode() + b">>\nstream\n"
        + content_bytes + b"\nendstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f \n"
        b"trailer<</Size 5/Root 1 0 R>>\nstartxref\n500\n%%EOF"
    )
    path.write_bytes(pdf)


def _make_mock_chat_result(content="Test answer"):
    mock = MagicMock()
    mock.content = content
    mock.prompt_tokens = 100
    mock.completion_tokens = 50
    mock.latency_s = 0.1
    return mock


class TestOutputFormatter:
    def test_format_string_answer(self):
        from lava26.output.formatter import format_answer
        assert format_answer("Hello", "string") == "Hello"

    def test_format_number_answer(self):
        from lava26.output.formatter import format_answer
        assert format_answer("42", "number") == "42"
        assert format_answer(42, "number") == "42"

    def test_format_list_answer(self):
        from lava26.output.formatter import format_answer
        result = format_answer("item1\nitem2\nitem3", "unordered_list")
        parsed = ast.literal_eval(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 3

    def test_format_evidence_1indexed(self):
        from lava26.output.formatter import format_evidence
        result = format_evidence([0, 2])  # 0-indexed input
        parsed = ast.literal_eval(result)
        assert parsed == ['1', '3']  # 1-indexed output

    def test_format_evidence_sorted_unique(self):
        from lava26.output.formatter import format_evidence
        result = format_evidence([2, 0, 2, 1])
        parsed = ast.literal_eval(result)
        assert parsed == ['1', '2', '3']


class TestResultStore:
    def test_store_and_retrieve(self):
        from lava26.output.submission import ResultStore
        store = ResultStore()
        store.store("q_001", {"answer": "test", "evidence_page_number": "[1]"})
        assert store.get("q_001")["answer"] == "test"
        assert len(store) == 1

    def test_thread_safety(self):
        import threading
        from lava26.output.submission import ResultStore
        store = ResultStore()
        errors = []

        def write(i):
            try:
                store.store(f"q_{i:04d}", {"answer": str(i)})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(store) == 50


class TestDumpSubmissionCSV:
    def test_output_format(self, tmp_path):
        from omegaconf import OmegaConf
        from lava26.output.submission import dump_submission_csv

        cfg = OmegaConf.create({
            "output": {
                "submission_csv": str(tmp_path / "submission.csv"),
                "results_json": str(tmp_path / "results.json"),
                "submission": {
                    "fallback_answer": "",
                    "fallback_evidence": [1],
                    "preserve_order": True,
                }
            }
        })

        results = {
            "q_0001": {"answer": "2026-01-15", "evidence_page_number": "['1']"},
            "q_0003": {"answer": "Câu trả lời", "evidence_page_number": "['1']"},
        }
        questions = [
            {"id": "q_0001"}, {"id": "q_0002"}, {"id": "q_0003"}
        ]

        csv_path = dump_submission_csv(results, questions, cfg)

        assert csv_path.exists()
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0]["id"] == "q_0001"
        assert rows[1]["id"] == "q_0002"
        assert rows[1]["answer"] == ""  # fallback
        assert rows[2]["id"] == "q_0003"


_torch_available = False
try:
    import torch  # noqa: F401
    _torch_available = True
except ImportError:
    pass


class TestPreprocessPhase:
    @pytest.mark.skipif(not QUESTIONS_FILE.exists(), reason="Fixtures not found")
    @pytest.mark.skipif(not _torch_available, reason="torch not installed")
    def test_preprocess_runs_without_crash(self, tmp_path):
        """Test that preprocess phase runs end-to-end without crashing."""
        from omegaconf import OmegaConf

        cfg_dict = {
            "data": {
                "test_questions": str(QUESTIONS_FILE),
                "pdf_dir": str(PDF_DIR),
                "sample": 2,
                "fields": {
                    "id": "id", "question": "question",
                    "pdf_filename": "pdf", "answer_type": "answer_type",
                    "language": "language",
                },
            },
            "parsing": {
                "dpi": 72, "max_image_long_side": 800,
                "text_extractor": "pymupdf",
                "use_mineru": False, "mineru_fallback_on_error": True,
                "vlm_reconcile": False, "cross_page_table": False,
                "cache_dir": str(tmp_path / "cache/parsed"),
                "force_reparse": False,
                "reconcile_max_tokens": 512, "reconcile_temperature": 0.0,
                "cross_page_table_max_span": 5,
            },
            "retriever": {
                "dense": {"enabled": False, "model": "dummy", "max_length": 512,
                           "batch_size": 1, "device": "cpu", "normalize": True,
                           "cache_dir": str(tmp_path / "cache/dense")},
                "sparse": {"enabled": False, "algorithm": "bm25", "bm25_k1": 1.5,
                            "bm25_b": 0.75, "tokenizer": "auto",
                            "cache_dir": str(tmp_path / "cache/bm25")},
                "visual": {"enabled": False, "retriever_model": "dummy",
                            "batch_size": 1, "device": "cpu",
                            "cache_dir": str(tmp_path / "cache/visual"),
                            "query_cache": str(tmp_path / "cache/query_embs.pt")},
                "fusion": {"method": "rrf", "rrf_k": 60,
                            "adaptive_threshold_ratio": 0.8, "top_k_cap": 2, "min_k": 1},
            },
            "budget": {"total_inference_seconds": 7200, "preprocess_seconds": 1500, "reserve_seconds": 300, "fast_path_threshold_s": 120},
            "language": {"source": "metadata_or_detect", "detector": "langid", "default": "ja"},
            "logging": {"level": "WARNING", "log_file": str(tmp_path / "test.log"),
                         "log_llm_calls": False, "log_retrieval_top_n": 3, "progress_bar": False},
            "ablation": {k: False for k in [
                "disable_vlm_reconciliation", "disable_cross_page_table",
                "disable_visual_retrieval", "disable_sparse_retrieval",
                "disable_dynamic_pruning", "disable_coordinator_agent",
                "disable_text_agent", "disable_image_agent", "disable_cross_modal_verification"
            ]},
        }
        cfg = OmegaConf.create(cfg_dict)

        import logging
        logging.disable(logging.CRITICAL)
        try:
            from run import run_preprocess
            run_preprocess(cfg, logging.getLogger("test"))
        finally:
            logging.disable(logging.NOTSET)
