#!/usr/bin/env python3
"""LAVA 2026 pipeline entrypoint.

Usage:
    python run.py [--config CONFIG] [--output OUTPUT] [--preprocess] [--clear-cache]
                  [key=value ...]

Examples:
    python run.py --help
    python run.py --config config.yaml --preprocess data.sample=5
    python run.py --config config.yaml data.sample=5 agents.mode=shared_vlm
"""

import argparse
import logging
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).parent))


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed
        set_seed(seed)
    except ImportError:
        pass


def _parse_args():
    parser = argparse.ArgumentParser(
        description="LAVA 2026 Document VQA Pipeline (HEAR-inspired)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="config.yaml", help="OmegaConf config file")
    parser.add_argument("--output", default=None, help="Override output.results_json path")
    parser.add_argument("--preprocess", action="store_true",
                        help="Phase 1 only: parse PDFs + encode embeddings, skip inference")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear all caches before running")
    parser.add_argument("overrides", nargs="*", metavar="key=value",
                        help="OmegaConf CLI overrides (e.g. data.sample=5 agents.mode=shared_vlm)")
    return parser.parse_args()


def _load_config(config_path: str, overrides: list):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


def _make_llm_client_factory(cfg):
    from lava26.llm.vllm_client import VLLMClient

    def factory(mode: str = "vlm"):
        if mode == "text" and cfg.agents.mode == "separate_text_model":
            model_cfg = cfg.model.qwen_text
        else:
            model_cfg = cfg.model.qwen_vl
        return VLLMClient(
            base_url=str(model_cfg.vllm_url),
            api_key=str(model_cfg.vllm_api_key),
            model_name=str(model_cfg.path).rstrip("/").split("/")[-1],
            cfg=model_cfg,
        )
    return factory


def run_preprocess(cfg, logger):
    """Phase 1: Parse PDFs + encode dense/visual embeddings."""
    from lava26.data.loader import load_questions, get_pdf_path, apply_sample_limit
    from lava26.data.lang_detect import resolve_language
    from lava26.parsing.extractor import extract_all_pages
    from lava26.utils.cache import DiskCache
    from lava26.utils.timing import BudgetTracker
    from lava26.retrieval.dense import DenseRetriever
    from lava26.retrieval.visual import VisualRetriever

    budget = BudgetTracker(
        cfg.budget.total_inference_seconds,
        cfg.budget.preprocess_seconds,
        cfg.budget.reserve_seconds,
    )
    budget.start_phase("preprocess")

    questions = load_questions(cfg.data.test_questions, cfg.data.fields)
    questions = apply_sample_limit(questions, cfg.data.sample)
    logger.info("Loaded %d questions", len(questions))

    parse_cache = DiskCache(cfg.parsing.cache_dir)

    # Group by PDF to avoid re-parsing
    pdf_to_questions: dict = {}
    for q in questions:
        pdf = q["pdf_filename"]
        pdf_to_questions.setdefault(pdf, []).append(q)

    dense_retriever = DenseRetriever(cfg) if cfg.retriever.dense.enabled else None
    visual_retriever = VisualRetriever(cfg) if cfg.retriever.visual.enabled else None

    from tqdm import tqdm
    for pdf_filename, pdf_questions in tqdm(pdf_to_questions.items(), desc="Parsing PDFs"):
        pdf_path = get_pdf_path(cfg.data.pdf_dir, pdf_filename)
        doc_id = Path(pdf_filename).stem
        cache_key = f"pages_{doc_id}"

        if not cfg.parsing.force_reparse and parse_cache.exists(cache_key):
            logger.debug("Cache hit for %s", doc_id)
            pages = parse_cache.load(cache_key)
        else:
            image_cache_dir = Path(cfg.parsing.cache_dir) / "images" / doc_id
            if pdf_path.exists():
                pages = extract_all_pages(pdf_path, cfg, image_cache_dir)
            else:
                logger.warning("PDF not found: %s", pdf_path)
                pages = []
            parse_cache.save(cache_key, pages)

        # Dense encode
        if dense_retriever and pages:
            dense_cache = DiskCache(cfg.retriever.dense.cache_dir)
            dense_key = f"dense_{doc_id}"
            if not dense_cache.exists(dense_key):
                embs = dense_retriever.encode_corpus(pages)
                dense_cache.save(dense_key, embs)

        # Visual encode
        if visual_retriever and pages:
            images = [
                _load_pil_image(p.image_path) for p in pages
            ]
            visual_retriever.encode_pages(images, doc_id)

        # Pre-encode query embeddings for visual retrieval
        if visual_retriever:
            for q in pdf_questions:
                lang = resolve_language(q, q["question"], cfg.language)
                q["_lang"] = lang
                visual_retriever.encode_query(q["question"], q["id"])

    if visual_retriever:
        visual_retriever.unload()
    if dense_retriever:
        dense_retriever.unload()

    budget.end_phase("preprocess")
    budget.print_summary()
    logger.info("Preprocessing complete.")


def _load_pil_image(image_path):
    if image_path is None:
        return None
    try:
        from PIL import Image
        return Image.open(image_path).convert("RGB")
    except Exception:
        return None


def run_inference(cfg, logger, result_store=None):
    """Phase 2: Retrieval + Multi-agent reasoning + Output."""
    from lava26.data.loader import load_questions, get_pdf_path, apply_sample_limit
    from lava26.data.lang_detect import resolve_language
    from lava26.utils.cache import DiskCache
    from lava26.utils.timing import BudgetTracker
    from lava26.utils.logging_setup import log_llm_call, log_retrieval
    from lava26.retrieval.dense import DenseRetriever
    from lava26.retrieval.sparse import SparseRetriever
    from lava26.retrieval.visual import VisualRetriever
    from lava26.retrieval.fusion import fuse_retrievers
    from lava26.agents.orchestrator import Orchestrator
    from lava26.output.formatter import detect_answer_type, format_answer, format_evidence
    from lava26.output.submission import ResultStore, dump_submission_csv, dump_results_json

    if result_store is None:
        result_store = ResultStore()

    budget = BudgetTracker(
        cfg.budget.total_inference_seconds,
        cfg.budget.preprocess_seconds,
        cfg.budget.reserve_seconds,
    )
    budget.start_phase("inference")

    questions = load_questions(cfg.data.test_questions, cfg.data.fields)
    questions = apply_sample_limit(questions, cfg.data.sample)

    parse_cache = DiskCache(cfg.parsing.cache_dir)
    dense_cache = DiskCache(cfg.retriever.dense.cache_dir)

    dense_retriever = DenseRetriever(cfg) if cfg.retriever.dense.enabled else None
    sparse_retriever = SparseRetriever(cfg) if cfg.retriever.sparse.enabled else None
    visual_retriever = VisualRetriever(cfg) if cfg.retriever.visual.enabled else None

    llm_factory = _make_llm_client_factory(cfg)
    orchestrator = Orchestrator(cfg, llm_factory)

    fast_path_threshold = cfg.budget.fast_path_threshold_s

    from tqdm import tqdm
    for q in tqdm(questions, desc="Processing questions"):
        if budget.is_budget_exhausted():
            logger.warning("Budget exhausted — stopping at question %s", q["id"])
            break

        qid = q["id"]
        lang = resolve_language(q, q["question"], cfg.language)
        q["_lang"] = lang

        # Skip re-evaluation if low budget
        use_fast_path = budget.is_fast_path(fast_path_threshold)

        try:
            with budget.question_timer(cfg.agents.per_question_timeout_s):
                # Load pages from cache
                doc_id = Path(q["pdf_filename"]).stem
                cache_key = f"pages_{doc_id}"
                pages = parse_cache.load(cache_key) or []

                if not pages:
                    logger.warning("No parsed pages for %s — using fallback", qid)
                    _write_fallback(result_store, q, cfg)
                    continue

                # Dense retrieval
                dense_res = None
                if dense_retriever and cfg.retriever.dense.enabled:
                    dense_key = f"dense_{doc_id}"
                    corpus_embs = dense_cache.load(dense_key)
                    if corpus_embs is not None:
                        query_emb = dense_retriever.encode_query(q["question"])
                        dense_res = dense_retriever.retrieve(query_emb, corpus_embs)

                # Sparse retrieval
                sparse_res = None
                if sparse_retriever and cfg.retriever.sparse.enabled:
                    sparse_retriever.build_index(pages, lang)
                    sparse_res = sparse_retriever.retrieve(q["question"], lang)

                # Visual retrieval (from cache — no model load in Phase 2)
                visual_res = None
                if visual_retriever and cfg.retriever.visual.enabled:
                    visual_res = visual_retriever.retrieve_from_cache(qid, doc_id)

                # Fusion
                fused = fuse_retrievers(dense_res, sparse_res, visual_res, cfg)
                retrieved_page_indices = [idx for idx, _ in fused]
                retrieval_scores = [s for _, s in fused]

                if not retrieved_page_indices:
                    retrieved_page_indices = [0]
                    retrieval_scores = [0.0]

                log_retrieval(logger, qid, [
                    {"page_num": pages[i].page_num_1 if i < len(pages) else i + 1, "score": s}
                    for i, s in zip(retrieved_page_indices, retrieval_scores)
                ], n=cfg.logging.log_retrieval_top_n)

                # Re-retrieve function for conflict re-evaluation
                def retrieve_fn(subquery: str):
                    if dense_retriever and cfg.retriever.dense.enabled:
                        dense_key2 = f"dense_{doc_id}"
                        ce = dense_cache.load(dense_key2)
                        if ce is not None:
                            qe = dense_retriever.encode_query(subquery)
                            res = dense_retriever.retrieve(qe, ce)
                            fused2 = fuse_retrievers(res, None, None, cfg)
                            return [i for i, _ in fused2]
                    return retrieved_page_indices

                # Multi-agent reasoning
                qr = orchestrator.process_question(
                    question_meta=q,
                    pages=pages,
                    retrieved_page_indices=retrieved_page_indices,
                    retrieval_scores=retrieval_scores,
                    retrieve_fn=None if use_fast_path else retrieve_fn,
                )

                # Format output
                answer_type = detect_answer_type(q, q["question"])
                formatted_answer = format_answer(qr.answer, answer_type)
                formatted_evidence = format_evidence(qr.evidence_pages)

                result = {
                    "id": qid,
                    "answer": formatted_answer,
                    "evidence_page_number": formatted_evidence,
                    "answer_type": answer_type,
                    "verification_status": qr.verification_status,
                    "agent_claims": qr.agent_claims,
                    "reeval_rounds": qr.reeval_rounds,
                    "retrieval_scores": list(zip(
                        [pages[i].page_num_1 if i < len(pages) else i + 1 for i in retrieved_page_indices],
                        [round(s, 4) for s in retrieval_scores],
                    )),
                }
                result_store.store(qid, result)

        except TimeoutError:
            logger.warning("Question %s timed out — using fallback", qid)
            _write_fallback(result_store, q, cfg)
        except Exception as e:
            logger.error("Question %s failed: %s", qid, traceback.format_exc())
            _write_fallback(result_store, q, cfg)

    budget.end_phase("inference")

    # Write output
    all_results = result_store.all_results()
    dump_results_json(all_results, cfg)
    csv_path = dump_submission_csv(all_results, questions, cfg)

    budget.print_summary()
    logger.info("Done. Submission: %s", csv_path)
    return result_store


def _write_fallback(result_store, q, cfg):
    from lava26.output.submission import _format_evidence_list
    fallback = {
        "id": q["id"],
        "answer": cfg.output.submission.fallback_answer,
        "evidence_page_number": _format_evidence_list(cfg.output.submission.fallback_evidence),
    }
    result_store.store(q["id"], fallback)


def main():
    args = _parse_args()

    cfg = _load_config(args.config, args.overrides)

    if args.output:
        from omegaconf import OmegaConf
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"output.results_json={args.output}"]))

    from lava26.utils.logging_setup import setup_logging
    logger = setup_logging(cfg)

    logger.info("LAVA 2026 pipeline starting")
    logger.info("Config: %s | Mode: %s", args.config, cfg.agents.mode)

    _set_seeds(cfg.seed)

    if args.clear_cache:
        from lava26.utils.cache import DiskCache
        for cache_dir in [
            cfg.parsing.cache_dir,
            cfg.retriever.dense.cache_dir,
            cfg.retriever.visual.cache_dir,
        ]:
            DiskCache(cache_dir).clear()
        logger.info("Caches cleared.")

    # Ensure output dirs exist
    Path(str(cfg.output.results_json)).parent.mkdir(parents=True, exist_ok=True)
    Path(str(cfg.output.submission_csv)).parent.mkdir(parents=True, exist_ok=True)

    result_store = None
    from lava26.output.submission import ResultStore, dump_submission_csv, dump_results_json
    result_store = ResultStore()

    def _emergency_dump(signum=None, frame=None):
        logger.warning("Signal received — writing partial submission CSV")
        try:
            from lava26.data.loader import load_questions, apply_sample_limit
            questions = load_questions(cfg.data.test_questions, cfg.data.fields)
            questions = apply_sample_limit(questions, cfg.data.sample)
            dump_submission_csv(result_store.all_results(), questions, cfg)
            dump_results_json(result_store.all_results(), cfg)
        except Exception as e:
            logger.error("Emergency dump failed: %s", e)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _emergency_dump)
    signal.signal(signal.SIGINT, _emergency_dump)

    try:
        if args.preprocess:
            run_preprocess(cfg, logger)
        else:
            run_inference(cfg, logger, result_store)
    except Exception:
        logger.error("Pipeline crashed:\n%s", traceback.format_exc())
        _emergency_dump()
        sys.exit(1)


if __name__ == "__main__":
    main()
