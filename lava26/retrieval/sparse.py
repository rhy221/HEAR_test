import importlib.util
import logging
import re
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


def _tokenize_whitespace(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _tokenize_mecab(text: str) -> List[str]:
    try:
        import MeCab
        tagger = MeCab.Tagger("-Owakati")
        result = tagger.parse(text)
        return result.strip().split()
    except Exception as e:
        logger.warning("MeCab failed: %s, falling back to whitespace", e)
        return _tokenize_whitespace(text)


def _tokenize_pyvi(text: str) -> List[str]:
    try:
        from pyvi import ViTokenizer
        tokenized = ViTokenizer.tokenize(text)
        return tokenized.split()
    except Exception as e:
        logger.warning("pyvi failed: %s, falling back to whitespace", e)
        return _tokenize_whitespace(text)


def _tokenize_underthesea(text: str) -> List[str]:
    try:
        from underthesea import word_tokenize
        return word_tokenize(text)
    except Exception as e:
        logger.warning("underthesea failed: %s, falling back to pyvi", e)
        return _tokenize_pyvi(text)


def tokenize(text: str, lang: str, tokenizer: str = "auto") -> List[str]:
    if tokenizer == "auto":
        tokenizer = "mecab" if lang == "ja" else "pyvi"

    if tokenizer == "mecab":
        if importlib.util.find_spec("MeCab"):
            return _tokenize_mecab(text)
        return _tokenize_whitespace(text)
    elif tokenizer == "pyvi":
        return _tokenize_pyvi(text)
    elif tokenizer == "underthesea":
        return _tokenize_underthesea(text)
    else:
        return _tokenize_whitespace(text)


class SparseRetriever:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.tokenizer_mode = cfg.retriever.sparse.tokenizer
        self.k1 = cfg.retriever.sparse.bm25_k1
        self.b = cfg.retriever.sparse.bm25_b
        self._index = None
        self._tokenized_corpus = None

    def build_index(self, pages: List[Any], lang: str) -> None:
        from rank_bm25 import BM25Okapi
        corpus_texts = []
        for p in pages:
            if hasattr(p, "text_chunks"):
                text = " ".join(p.text_chunks)
            elif hasattr(p, "text"):
                text = p.text
            else:
                text = str(p)
            corpus_texts.append(text)

        self._tokenized_corpus = [
            tokenize(t, lang, self.tokenizer_mode) for t in corpus_texts
        ]
        self._index = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)
        logger.debug("BM25 index built: %d docs", len(corpus_texts))

    def retrieve(self, question: str, lang: str) -> List[Tuple[int, float]]:
        if self._index is None:
            logger.warning("SparseRetriever: index not built")
            return []
        query_tokens = tokenize(question, lang, self.tokenizer_mode)
        scores = self._index.get_scores(query_tokens)
        results = [(i, float(s)) for i, s in enumerate(scores)]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def clear(self) -> None:
        self._index = None
        self._tokenized_corpus = None
