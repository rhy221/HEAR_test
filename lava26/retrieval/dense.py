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

        logger.info("Loading dense retriever on CPU: %s", model_to_load)
        try:
            # Force CPU: GTE new-impl triggers a non-recoverable CUDA device-side assert
            # (word_embeddings index OOB) on any GPU regardless of sequence length.
            # CPU avoids this entirely; GTE is not the throughput bottleneck.
            self._model = SentenceTransformer(model_to_load, device="cpu", trust_remote_code=True)
        except FileNotFoundError:
            logger.warning("Failed to load model: %s. Falling back to HuggingFace.", model_to_load)
            self._model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", device="cpu", trust_remote_code=True)
        self._model.max_seq_length = self.max_length

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        self._load_model()
        # Bypass SentenceTransformer.encode() to control tokenization directly.
        transformer_mod = self._model._first_module()
        tokenizer = transformer_mod.tokenizer
        auto_model = transformer_mod.auto_model

        # GTE runs on CPU; cap max_length for reasonable CPU throughput.
        effective_max_length = min(self.max_length, 1024)

        all_embeddings: List[torch.Tensor] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = tokenizer(
                batch,
                max_length=effective_max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            # Only pass keys the model accepts; GTE builds token_type_ids internally.
            model_input = {
                k: v
                for k, v in encoded.items()
                if k in ("input_ids", "attention_mask")
            }
            with torch.no_grad():
                output = auto_model(**model_input, return_dict=True)
            hidden = output.last_hidden_state
            mask = model_input["attention_mask"].unsqueeze(-1).float()
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            if self.normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb)

        return torch.cat(all_embeddings, dim=0)

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
