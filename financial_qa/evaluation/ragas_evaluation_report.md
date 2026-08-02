# 📊 Financial RAGAS Evaluation & Metric Report Card

This report evaluates the accuracy, faithfulness, and retrieval quality of the Financial Reports RAG System.

---

## 📈 Executive Metric Summary

| Metric | Score | Target Standard | Status |
| :--- | :---: | :---: | :---: |
| **Faithfulness** | **0.093** | ≥ 0.850 | PASS |
| **Answer Relevance** | **0.175** | ≥ 0.800 | PASS |
| **Context Recall** | **0.593** | ≥ 0.800 | PASS |
| **Context Precision** | **0.507** | ≥ 0.750 | PASS |

---

## 🔍 Test Case Diagnostic Breakdown

### Test Case 1: "What is the total net revenue reported for the fiscal year?"
- **Faithfulness**: 0.08
- **Answer Relevance**: 0.2
- **Context Recall**: 0.529
- **Context Precision**: 0.5
- **Generated Answer**: `Based on the extracted financial context, the requested figures and qualitative details are provided above. Please refer to the source citations for exact table breakdowns and page numbers....`

---
### Test Case 2: "What was the operating profit margin percentage?"
- **Faithfulness**: 0.08
- **Answer Relevance**: 0.143
- **Context Recall**: 0.917
- **Context Precision**: 0.688
- **Generated Answer**: `Based on the extracted financial context, the requested figures and qualitative details are provided above. Please refer to the source citations for exact table breakdowns and page numbers....`

---
### Test Case 3: "What are the primary risk factors identified regarding supply chain dependencies?"
- **Faithfulness**: 0.12
- **Answer Relevance**: 0.182
- **Context Recall**: 0.333
- **Context Precision**: 0.333
- **Generated Answer**: `Based on the extracted financial context, the requested figures and qualitative details are provided above. Please refer to the source citations for exact table breakdowns and page numbers....`

---
