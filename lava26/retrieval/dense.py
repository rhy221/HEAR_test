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

        model_to_load = self.model_name
        # If model_name is a local path and doesn't exist, use HuggingFace fallback
        if Path(self.model_name).exists() or "/" not in self.model_name:
            # It's a local path or HuggingFace model ID
            if Path(self.model_name).is_dir() and not Path(self.model_name).exists():
                logger.warning("Local model path not found: %s. Using HuggingFace fallback.", self.model_name)
                model_to_load = "Alibaba-NLP/gte-multilingual-base"

        logger.info("Loading dense retriever: %s", model_to_load)
        try:
            self._model = SentenceTransformer(model_to_load, device=self.device, trust_remote_code=True)
        except FileNotFoundError:
            logger.warning("Failed to load model: %s. Falling back to HuggingFace.", model_to_load)
            self._model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", device=self.device, trust_remote_code=True)
        self._model.max_seq_length = self.max_length
        # Force tokenizer truncation directly — GTE's custom new-impl tokenizer
        # ignores ST's max_seq_length setter and has no model_max_length set,
        # causing position_ids to exceed max_position_embeddings on long pages.
        tok = getattr(self._model, "tokenizer", None)
        if tok is not None:
            tok.model_max_length = self.max_length

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        self._load_model()
        # Pre-truncate by characters as a hard safety net.
        # GTE uses ~2–4 chars/token for multilingual text; multiply by 2 to be safe.
        max_chars = self.max_length * 2
        texts = [t[:max_chars] if len(t) > max_chars else t for t in texts]
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
