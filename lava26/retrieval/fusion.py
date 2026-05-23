import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

Ranking = List[Tuple[int, float]]


def rrf_fusion(rankings: List[Ranking], k: int = 60) -> Ranking:
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, (page_idx, _) in enumerate(ranking):
            scores[page_idx] = scores.get(page_idx, 0.0) + 1.0 / (k + rank + 1)
    result = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(idx, score) for idx, score in result]


def adaptive_threshold(ranked: Ranking, ratio: float = 0.8) -> float:
    if not ranked:
        return 0.0
    max_score = ranked[0][1]
    return ratio * max_score


def dynamic_cap(
    ranked: Ranking,
    threshold: float,
    top_k_cap: int = 2,
    min_k: int = 1,
) -> Ranking:
    selected = [(idx, s) for idx, s in ranked if s >= threshold]
    if len(selected) < min_k:
        selected = ranked[:min_k]
    if len(selected) > top_k_cap:
        selected = selected[:top_k_cap]
    return selected


def fuse_retrievers(
    dense_res: Optional[Ranking],
    sparse_res: Optional[Ranking],
    visual_res: Optional[Ranking],
    cfg: Any,
) -> Ranking:
    if getattr(cfg.ablation, "disable_dynamic_pruning", False):
        # No pruning: just return all pages ranked by RRF
        rankings = [r for r in [dense_res, sparse_res, visual_res] if r]
        if not rankings:
            return []
        return rrf_fusion(rankings, k=cfg.retriever.fusion.rrf_k)

    rankings = []
    if dense_res and cfg.retriever.dense.enabled:
        rankings.append(dense_res)
    if sparse_res and cfg.retriever.sparse.enabled:
        rankings.append(sparse_res)
    if visual_res and cfg.retriever.visual.enabled:
        rankings.append(visual_res)

    if not rankings:
        return []

    fused = rrf_fusion(rankings, k=cfg.retriever.fusion.rrf_k)
    threshold = adaptive_threshold(fused, cfg.retriever.fusion.adaptive_threshold_ratio)
    capped = dynamic_cap(
        fused,
        threshold,
        top_k_cap=cfg.retriever.fusion.top_k_cap,
        min_k=cfg.retriever.fusion.min_k,
    )

    logger.debug(
        "Fusion: dense=%d sparse=%d visual=%d → fused=%d → capped=%d (τ=%.4f)",
        len(dense_res or []),
        len(sparse_res or []),
        len(visual_res or []),
        len(fused),
        len(capped),
        threshold,
    )
    return capped
