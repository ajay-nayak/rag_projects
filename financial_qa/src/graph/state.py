from typing import TypedDict, List, Dict, Any, Optional


class GraphState(TypedDict):
    """Represents the state of our agentic financial RAG graph."""

    question: str
    original_question: str
    documents: List[Dict[str, Any]]
    generation: str
    query_type: str  # 'financial_table' or 'narrative_text'
    relevance_score: float
    is_grounded: bool
    retry_count: int
    citations: List[Dict[str, Any]]
    execution_trace: List[str]
