import pickle
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

from src.utils.config import INDEX_CACHE_DIR

BM25_CACHE_FILE = INDEX_CACHE_DIR / "bm25_index.pkl"

try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


class FinancialBM25Retriever:
    """BM25 sparse retriever for exact keyword matching in financial documents."""

    def __init__(self):
        self.corpus_chunks: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi = None
        self._load_cache()

    def _load_cache(self):
        if BM25_CACHE_FILE.exists():
            try:
                with open(BM25_CACHE_FILE, "rb") as f:
                    data = pickle.load(f)
                    self.corpus_chunks = data.get("corpus_chunks", [])
                    self._build_index()
            except Exception as e:
                print(f"Warning: Failed loading BM25 cache: {e}")

    def _save_cache(self):
        with open(BM25_CACHE_FILE, "wb") as f:
            pickle.dump({"corpus_chunks": self.corpus_chunks}, f)

    def _build_index(self):
        if not self.corpus_chunks:
            self.bm25 = None
            return

        tokenized_corpus = [
            word_tokenize(c["content"].lower()) for c in self.corpus_chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def add_chunks(self, new_chunks: List[Dict[str, Any]]):
        """Append new chunks to corpus and rebuild BM25 index."""
        if not new_chunks:
            return

        # Simple deduplication by chunk ID or hash
        existing_ids = {c.get("id") for c in self.corpus_chunks if "id" in c}
        for chunk in new_chunks:
            if chunk.get("id") not in existing_ids:
                self.corpus_chunks.append(chunk)

        self._build_index()
        self._save_cache()

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query BM25 for top-k keyword matching documents."""
        if not self.bm25 or not self.corpus_chunks:
            return []

        tokenized_query = word_tokenize(query_text.lower())
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "content": self.corpus_chunks[idx]["content"],
                    "metadata": self.corpus_chunks[idx]["metadata"],
                    "score": float(scores[idx]),
                })
        return results
