from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

from src.utils.config import RERANKER_MODEL_NAME


class HybridReranker:
    """Reciprocal Rank Fusion (RRF) & Cross-Encoder re-ranking for hybrid retrieval."""

    def __init__(self, use_cross_encoder: bool = False):
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder = None
        if self.use_cross_encoder:
            try:
                self.cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)
            except Exception as e:
                print(f"Notice: CrossEncoder model loading deferred or fallback: {e}")

    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_k: int = 5,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """Combines dense and sparse retrieved documents using RRF score formula."""
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        # Process dense
        for rank, doc in enumerate(dense_results):
            content = doc["content"]
            doc_map[content] = doc
            doc_scores[content] = doc_scores.get(content, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Process sparse
        for rank, doc in enumerate(sparse_results):
            content = doc["content"]
            doc_map[content] = doc
            doc_scores[content] = doc_scores.get(content, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Sort by RRF score
        sorted_contents = sorted(doc_scores.keys(), key=lambda c: doc_scores[c], reverse=True)

        reranked_docs = []
        for c in sorted_contents[:top_k]:
            item = dict(doc_map[c])
            item["rrf_score"] = doc_scores[c]
            reranked_docs.append(item)

        return reranked_docs

    def rerank_with_cross_encoder(
        self, query: str, candidate_docs: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Re-ranks candidate documents using Cross-Encoder model."""
        if not candidate_docs:
            return []

        if not self.cross_encoder:
            # Fallback to RRF order
            return candidate_docs[:top_k]

        pairs = [[query, doc["content"]] for doc in candidate_docs]
        scores = self.cross_encoder.predict(pairs)

        for idx, doc in enumerate(candidate_docs):
            doc["cross_encoder_score"] = float(scores[idx])

        sorted_docs = sorted(candidate_docs, key=lambda d: d["cross_encoder_score"], reverse=True)
        return sorted_docs[:top_k]
