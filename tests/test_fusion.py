"""Test RRF fusion math."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lava26.retrieval.fusion import rrf_fusion, adaptive_threshold, dynamic_cap


class TestRRF:
    def test_rrf_basic(self):
        ranking1 = [(0, 0.9), (1, 0.7), (2, 0.5)]
        ranking2 = [(1, 0.8), (0, 0.6), (2, 0.3)]
        result = rrf_fusion([ranking1, ranking2], k=60)
        # Page 0 and 1 should be top-ranked
        result_ids = [idx for idx, _ in result]
        assert result_ids[0] in (0, 1), f"Top result unexpected: {result_ids}"
        assert result_ids[1] in (0, 1)

    def test_rrf_single_ranker(self):
        ranking = [(3, 0.9), (1, 0.5), (0, 0.2)]
        result = rrf_fusion([ranking], k=60)
        assert result[0][0] == 3  # Top should be page 3

    def test_rrf_empty(self):
        assert rrf_fusion([], k=60) == []

    def test_rrf_scores_positive(self):
        ranking = [(0, 1.0), (1, 0.5)]
        result = rrf_fusion([ranking], k=60)
        for _, score in result:
            assert score > 0

    def test_rrf_k_effect(self):
        ranking = [(0, 1.0), (1, 0.5)]
        result_k1 = rrf_fusion([ranking], k=1)
        result_k100 = rrf_fusion([ranking], k=100)
        # Higher k makes scores more uniform; verify ordering preserved
        assert result_k1[0][0] == result_k100[0][0] == 0


class TestAdaptiveThreshold:
    def test_basic(self):
        ranked = [(0, 1.0), (1, 0.8), (2, 0.5)]
        τ = adaptive_threshold(ranked, ratio=0.8)
        assert abs(τ - 0.8) < 1e-6

    def test_empty(self):
        assert adaptive_threshold([], 0.8) == 0.0

    def test_ratio_zero(self):
        ranked = [(0, 1.0)]
        assert adaptive_threshold(ranked, ratio=0.0) == 0.0


class TestDynamicCap:
    def test_cap_at_top_k(self):
        ranked = [(0, 1.0), (1, 0.9), (2, 0.8), (3, 0.7)]
        result = dynamic_cap(ranked, threshold=0.5, top_k_cap=2, min_k=1)
        assert len(result) == 2

    def test_min_k_enforced(self):
        ranked = [(0, 1.0), (1, 0.3), (2, 0.2)]
        result = dynamic_cap(ranked, threshold=0.99, top_k_cap=5, min_k=2)
        assert len(result) >= 2

    def test_threshold_filters(self):
        ranked = [(0, 1.0), (1, 0.5), (2, 0.1)]
        result = dynamic_cap(ranked, threshold=0.6, top_k_cap=5, min_k=1)
        ids = [r[0] for r in result]
        assert 2 not in ids  # Score 0.1 below threshold 0.6

    def test_empty(self):
        result = dynamic_cap([], threshold=0.5, top_k_cap=2, min_k=1)
        assert len(result) == 0
