import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class VisualRetriever:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.model_name = cfg.retriever.visual.retriever_model
        self.batch_size = cfg.retriever.visual.batch_size
        self.device = cfg.retriever.visual.device
        self.cache_dir = Path(cfg.retriever.visual.cache_dir)
        self.query_cache_path = Path(cfg.retriever.visual.query_cache)
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
            logger.info("Loading ColQwen: %s", self.model_name)
            self._model = ColQwen2_5.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            ).eval()
            self._processor = ColQwen2_5_Processor.from_pretrained(self.model_name)
        except Exception as e:
            logger.error("Failed to load ColQwen: %s", e)
            raise

    def encode_pages(self, images: List[Any], doc_id: str) -> Optional[torch.Tensor]:
        self._load_model()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{doc_id}.pt"

        if cache_path.exists():
            logger.debug("Loading visual embeddings from cache: %s", cache_path)
            return torch.load(cache_path, map_location="cpu", weights_only=False)

        valid_images = [img for img in images if img is not None]
        if not valid_images:
            return None

        all_embs = []
        for i in range(0, len(valid_images), self.batch_size):
            batch = valid_images[i: i + self.batch_size]
            try:
                inputs = self._processor.process_images(batch).to(self.device)
                with torch.no_grad():
                    embs = self._model(**inputs)
                all_embs.append(embs.cpu())
            except Exception as e:
                logger.warning("ColQwen batch encoding failed: %s", e)
                all_embs.append(torch.zeros(len(batch), 128, dtype=torch.float32))

        if not all_embs:
            return None
        page_embs = torch.cat(all_embs, dim=0)
        torch.save(page_embs, cache_path)
        logger.debug("Saved visual embeddings for %s (%d pages)", doc_id, len(valid_images))
        return page_embs

    def encode_query(self, question: str, question_id: str) -> Optional[torch.Tensor]:
        self._load_model()

        # Load existing query cache
        query_cache = {}
        if self.query_cache_path.exists():
            try:
                query_cache = torch.load(self.query_cache_path, map_location="cpu", weights_only=False)
            except Exception:
                pass

        if question_id in query_cache:
            return query_cache[question_id]

        try:
            inputs = self._processor.process_queries([question]).to(self.device)
            with torch.no_grad():
                query_emb = self._model(**inputs).cpu()
        except Exception as e:
            logger.warning("ColQwen query encoding failed: %s", e)
            return None

        query_cache[question_id] = query_emb
        self.query_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(query_cache, self.query_cache_path)
        return query_emb

    def retrieve(
        self,
        query_emb: torch.Tensor,
        page_embs: torch.Tensor,
    ) -> List[Tuple[int, float]]:
        if page_embs is None or query_emb is None:
            return []

        # MaxSim scoring: for each page, max over token dim, then sum
        # query_emb: (1, seq_q, dim) or (seq_q, dim)
        # page_embs: (n_pages, seq_p, dim) or (n_pages, dim)
        if query_emb.dim() == 2:
            query_emb = query_emb.unsqueeze(0)  # (1, seq_q, dim)
        if page_embs.dim() == 2:
            page_embs = page_embs.unsqueeze(1)  # (n_pages, 1, dim)

        if query_emb.dim() == 3 and page_embs.dim() == 3:
            # (n_pages, seq_q, seq_p) → max over seq_p → (n_pages, seq_q) → sum
            scores_matrix = torch.einsum("qsd,prd->pqs", query_emb, page_embs)
            scores = scores_matrix.max(dim=-1).values.sum(dim=-1)  # (n_pages,)
        else:
            # Fallback: simple dot product
            scores = torch.matmul(page_embs.squeeze(), query_emb.squeeze())

        results = [(i, float(scores[i])) for i in range(len(scores))]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def retrieve_from_cache(self, question_id: str, doc_id: str) -> List[Tuple[int, float]]:
        cache_path = self.cache_dir / f"{doc_id}.pt"
        if not cache_path.exists():
            logger.warning("No visual cache for doc %s", doc_id)
            return []

        page_embs = torch.load(cache_path, map_location="cpu", weights_only=False)

        query_cache = {}
        if self.query_cache_path.exists():
            query_cache = torch.load(self.query_cache_path, map_location="cpu", weights_only=False)

        query_emb = query_cache.get(question_id)
        if query_emb is None:
            logger.warning("No query cache for question %s", question_id)
            return []

        return self.retrieve(query_emb, page_embs)

    def unload(self):
        self._model = None
        self._processor = None
        torch.cuda.empty_cache()
