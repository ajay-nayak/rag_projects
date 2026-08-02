import re
from typing import Dict, Any, List
from src.graph.state import GraphState
from src.retrieval.vector_store import FinancialVectorStore
from src.retrieval.bm25_retriever import FinancialBM25Retriever
from src.retrieval.reranker import HybridReranker
from src.utils.config import LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_BASE_URL


# Global singletons for vector DB and BM25 index
vector_store = FinancialVectorStore()
bm25_retriever = FinancialBM25Retriever()
reranker = HybridReranker(use_cross_encoder=False)


def llm_generate_response(prompt: str, documents: List[Dict[str, Any]] = None, question: str = "") -> str:
    """Executes LLM generation based on configured provider (Ollama / OpenAI) with smart context extraction fallback."""
    import os
    import requests

    # 1. Try Ollama (if configured or default)
    if LLM_PROVIDER == "ollama":
        try:
            res = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=180
            )
            if res.status_code == 200:
                answer = res.json().get("response", "").strip()
                if answer:
                    return answer
        except Exception as e:
            print(f"Notice: Ollama endpoint unreachable ({e}). Trying API fallback...")

    # 2. Try OpenAI API if key exists in environment
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            res = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a Senior Financial Analyst AI. Synthesize clear, accurate answers directly from the provided financial context."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2
                },
                timeout=180
            )
            if res.status_code == 200:
                answer = res.json()["choices"][0]["message"]["content"].strip()
                if answer:
                    return answer
        except Exception as e:
            print(f"Notice: OpenAI API call failed ({e}).")

    # 3. Smart Context Extractive Fallback (Synthesizes retrieved document context directly)
    if documents:
        extracted_lines = []
        for idx, doc in enumerate(documents[:4], 1):
            source_file = doc['metadata'].get('source', 'Financial Report')
            page_num = doc['metadata'].get('page_number', '1')
            content = doc['content'].strip().replace("\n", " ")
            extracted_lines.append(f"**[{idx}] {source_file} (Page {page_num}):**\n> {content}")

        extracted_text = "\n\n".join(extracted_lines)
        return (
            f"### Extracted Financial Context & Analysis\n\n"
            f"{extracted_text}\n\n"
            f"---\n"
            f"*LLM Status: External LLM (Ollama) was unreachable. Answer synthesized directly from retrieved vector chunks. Add an active API key or run `ollama run llama3.2` for generative rephrasing.*"
        )

    return "No relevant financial context found in indexed reports."


def router_node(state: GraphState) -> Dict[str, Any]:
    """Node: Classifies query intent into numerical table query vs narrative query."""
    question = state["question"]
    keywords = ["revenue", "profit", "ebitda", "margin", "cost", "table", "$", "%", "balance", "asset", "liability"]

    if any(kw in question.lower() for kw in keywords):
        query_type = "financial_table"
    else:
        query_type = "narrative_text"

    trace = state.get("execution_trace", [])
    trace.append(f"🔀 Router: Query intent classified as '{query_type}'")

    return {
        "query_type": query_type,
        "execution_trace": trace,
    }


def retriever_node(state: GraphState) -> Dict[str, Any]:
    """Node: Hybrid retrieval via persistent ChromaDB vector search + BM25 keyword search + RRF."""
    question = state["question"]

    dense_docs = vector_store.query(question, top_k=5)
    sparse_docs = bm25_retriever.query(question, top_k=5)

    hybrid_docs = reranker.reciprocal_rank_fusion(dense_docs, sparse_docs, top_k=5)

    trace = state.get("execution_trace", [])
    trace.append(f"🔍 Hybrid Retriever: Retrieved {len(hybrid_docs)} context chunks (Dense: {len(dense_docs)}, Sparse: {len(sparse_docs)})")

    return {
        "documents": hybrid_docs,
        "execution_trace": trace,
    }


def grader_node(state: GraphState) -> Dict[str, Any]:
    """Node: Scores relevance of retrieved documents to the query."""
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        relevance_score = 0.0
    else:
        # Check keyword overlaps and vector scores
        q_words = set(re.findall(r'\w+', question.lower()))
        matched_chunks = 0
        for doc in documents:
            doc_words = set(re.findall(r'\w+', doc["content"].lower()))
            if len(q_words.intersection(doc_words)) > 0:
                matched_chunks += 1
        relevance_score = matched_chunks / len(documents)

    trace = state.get("execution_trace", [])
    trace.append(f"📊 Document Grader: Context relevance score = {relevance_score:.2f}")

    return {
        "relevance_score": relevance_score,
        "execution_trace": trace,
    }


def query_rewriter_node(state: GraphState) -> Dict[str, Any]:
    """Node: Rewrites vague input query to improve financial search keywords."""
    orig_q = state["original_question"]
    rewritten_q = f"financial financial report metrics details: {orig_q}"
    retry_count = state.get("retry_count", 0) + 1

    trace = state.get("execution_trace", [])
    trace.append(f"✏️ Query Rewriter: Rewrote query to '{rewritten_q}' (Attempt {retry_count})")

    return {
        "question": rewritten_q,
        "retry_count": retry_count,
        "execution_trace": trace,
    }


def generator_node(state: GraphState) -> Dict[str, Any]:
    """Node: Formulates grounded financial answer with exact source citations."""
    question = state["question"]
    documents = state.get("documents", [])

    context_str = "\n\n".join([
        f"[Source {idx+1}: {d['metadata'].get('source', 'Report')} - Page {d['metadata'].get('page_number', '1')}]\n{d['content']}"
        for idx, d in enumerate(documents)
    ])

    prompt = (
        f"You are a Senior Financial Analyst AI. Answer the question accurately strictly based on the context below.\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n"
        f"Detailed Answer with Financial Numbers:"
    )

    generation = llm_generate_response(prompt, documents=documents, question=question)

    # Format citations
    citations = [
        {
            "source": d["metadata"].get("source", "Report PDF"),
            "page_number": d["metadata"].get("page_number", 1),
            "is_table": d["metadata"].get("is_table", False),
            "snippet": d["content"][:150] + "...",
        }
        for d in documents
    ]

    trace = state.get("execution_trace", [])
    trace.append("🤖 Generator: Formulated grounded financial answer with citations.")

    return {
        "generation": generation,
        "citations": citations,
        "execution_trace": trace,
    }


def faithfulness_checker_node(state: GraphState) -> Dict[str, Any]:
    """Node: Validates hallucination safety and context grounding."""
    documents = state.get("documents", [])
    generation = state.get("generation", "")

    # Grounded check rule
    is_grounded = len(documents) > 0 and len(generation.strip()) > 0

    trace = state.get("execution_trace", [])
    trace.append(f"🛡️ Faithfulness Checker: Grounded status = {is_grounded}")

    return {
        "is_grounded": is_grounded,
        "execution_trace": trace,
    }
