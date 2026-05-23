import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def setup_logging(cfg: Any) -> logging.Logger:
    level = getattr(logging, cfg.logging.level, logging.INFO)
    log_file = Path(str(cfg.logging.log_file))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    logger = logging.getLogger("lava26")
    logger.setLevel(level)
    return logger


def log_llm_call(
    logger: logging.Logger,
    agent_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_s: float,
) -> None:
    logger.info(
        "LLM call | agent=%-20s | prompt_tok=%5d | completion_tok=%5d | latency=%.2fs",
        agent_name,
        prompt_tokens,
        completion_tokens,
        latency_s,
    )


def log_retrieval(
    logger: logging.Logger,
    question_id: str,
    top_n: List[Dict],
    n: int = 5,
) -> None:
    logger.info("Retrieval | id=%-10s | top-%d pages:", question_id, n)
    for i, item in enumerate(top_n[:n]):
        logger.info(
            "  [%d] page=%-4s score=%.4f",
            i + 1,
            item.get("page_num", "?"),
            item.get("score", 0.0),
        )
