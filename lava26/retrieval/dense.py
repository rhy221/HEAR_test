import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.model_name = cfg.retriever.dense.model
        self.batch_size = cfg.retriever.dense.batch_size
        self.device = cfg.retriever.dense.device
        self.max_length = cfg.retriever.dense.max_length
        self.normalize = cfg.retriever.dense.normalize
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info("Loading dense retriever: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name, device=self.device)

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_tensor=True,
        )
        return embeddings.cpu()

    def encode_corpus(self, pages: List[Any]) -> torch.Tensor:
        texts = []
        for p in pages:
            if hasattr(p, "text_chunks"):
                texts.append(" ".join(p.text_chunks))
            elif hasattr(p, "text"):
                texts.append(p.text)
            else:
                texts.append(str(p))
        return self.encode_texts(texts)

    def encode_query(self, question: str) -> torch.Tensor:
        return self.encode_texts([question])[0]

    def retrieve(
        self,
        query_emb: torch.Tensor,
        corpus_embs: torch.Tensor,
    ) -> List[Tuple[int, float]]:
        if corpus_embs.shape[0] == 0:
            return []
        scores = torch.matmul(corpus_embs, query_emb).squeeze(-1)
        results = [(int(i), float(scores[i])) for i in range(len(scores))]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def unload(self):
        self._model = None
        torch.cuda.empty_cache()
