import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

from src.utils.config import CHROMA_DB_DIR, EMBEDDING_MODEL_NAME


class FinancialVectorStore:
    """ChromaDB persistent vector store wrapper for financial reports."""

    def __init__(self, collection_name: str = "financial_reports"):
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Upsert document chunks into persistent ChromaDB collection."""
        if not chunks:
            return

        ids = [c["id"] for c in chunks]
        texts = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        # Generate embeddings
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query vector database for top-k semantically similar text chunks."""
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedder.encode([query_text]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        retrieved_docs = []
        if results and "documents" in results and results["documents"]:
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                retrieved_docs.append({
                    "content": doc,
                    "metadata": meta,
                    "score": 1.0 - float(dist),  # Cosine similarity approximation
                })

        return retrieved_docs

    def count(self) -> int:
        return self.collection.count()
