from langgraph.graph import StateGraph, END
from typing import Dict, Any

from src.graph.state import GraphState
from src.graph.nodes import (
    router_node,
    retriever_node,
    grader_node,
    query_rewriter_node,
    generator_node,
    faithfulness_checker_node,
)


def decide_after_grader(state: GraphState) -> str:
    """Conditional edge router following document grader evaluation."""
    score = state.get("relevance_score", 0.0)
    retry_count = state.get("retry_count", 0)

    if score < 0.2 and retry_count < 2:
        return "rewrite_query"
    return "generate"


def build_financial_rag_graph():
    """Compiles the LangGraph state machine workflow."""
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("grader", grader_node)
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("faithfulness_checker", faithfulness_checker_node)

    # Define Graph Edges
    workflow.set_entry_point("router")
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "grader")

    # Conditional Branching
    workflow.add_conditional_edges(
        "grader",
        decide_after_grader,
        {
            "rewrite_query": "query_rewriter",
            "generate": "generator",
        }
    )

    workflow.add_edge("query_rewriter", "retriever")
    workflow.add_edge("generator", "faithfulness_checker")
    workflow.add_edge("faithfulness_checker", END)

    return workflow.compile()


# Compiled Singleton Graph Instance
app_graph = build_financial_rag_graph()


def run_financial_rag(question: str) -> Dict[str, Any]:
    """Execution helper for the compiled Financial RAG LangGraph workflow."""
    initial_state = {
        "question": question,
        "original_question": question,
        "documents": [],
        "generation": "",
        "query_type": "unknown",
        "relevance_score": 0.0,
        "is_grounded": False,
        "retry_count": 0,
        "citations": [],
        "execution_trace": [f"🚀 Starting RAG Execution for query: '{question}'"],
    }

    final_state = app_graph.invoke(initial_state)
    return final_state
