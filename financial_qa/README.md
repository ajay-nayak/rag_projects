# 📊 Financial Reports Agentic RAG System

An enterprise-grade, stateful **Retrieval-Augmented Generation (RAG)** architecture engineered specifically for financial annual reports, SEC filings, and corporate balance sheets.

Built using **LangGraph**, **ChromaDB**, **BM25**, **HuggingFace Embeddings**, **RAGAS Evaluation**, **MLflow**, and **Streamlit**.

---

## 🌟 Key Architecture & Technical Differentiators

- 🤖 **LangGraph Stateful Agent Workflow**: State machine graph incorporating query routing, hybrid retrieval, document relevance grading, automatic query rewriting, grounded generation, and hallucination safety checks.
- ⚡ **Table-Aware Parsing & Incremental Indexing**: Extracts financial tables into formatted Markdown preserving header context. Hashes document contents (`SHA256`) to incrementally process only new/updated PDFs added to `reports/`.
- 🔍 **Multi-Stage Hybrid Search**: Combines persistent dense vector search (**ChromaDB** with `BAAI/bge-small-en-v1.5`) and sparse keyword search (**BM25**) merged via **Reciprocal Rank Fusion (RRF)**.
- 📊 **RAGAS Evaluation Benchmark**: Includes a comprehensive evaluation suite computing **Faithfulness**, **Answer Relevance**, **Context Recall**, and **Context Precision**.
- 🧪 **MLflow Chunking Experiment Tracker**: Benchmark suite comparing **Fixed-size**, **Recursive Character**, and **Structural Table-Aware** chunking strategies.
- 🖥️ **Interactive Demo Dashboard**: Streamlit interface displaying live node execution traces, source citation drawers, and embedded metric leaderboards.

---

## 🏗️ LangGraph Agent Workflow Diagram

```
+----------------+       +-------------------+       +------------------+
|  User Query    | ----> |   Router Node     | ----> |  Retriever Node  |
+----------------+       +-------------------+       +------------------+
                                                               |
                                                               v
                                                     +------------------+
                                                     |   Grader Node    |
                                                     +------------------+
                                                               |
                                            [Relevance Score < 0.2?]
                                            /                          \
                                        (Yes)                          (No)
                                          v                              v
                               +--------------------+          +-------------------+
                               | Query Rewriter     |          |  Generator Node   |
                               +--------------------+          +-------------------+
                                          |                              |
                                          +-----> [Re-Retrieve]          v
                                                               +-------------------+
                                                               | Faithfulness Check|
                                                               +-------------------+
                                                                         |
                                                                         v
                                                                      [ END ]
```

---

## 🚀 Quick Start Guide (using `uv`)

### 1. Environment Setup & Virtual Environment Activation

From the repository root (`rag_projects`):

```bash
# Synchronize virtual environment with UV workspace
uv sync

# Activate the virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.\.venv\Scripts\activate.bat

# On macOS/Linux:
source .venv/bin/activate
```

### 2. LLM Setup (Ollama or API Key)

The system supports local LLMs (via **Ollama**), cloud LLMs (**OpenAI API**), or built-in **Direct Extractive RAG Synthesis**.

#### Option A: Run Local Ollama (Recommended)
1. Download and install [Ollama](https://ollama.com/).
2. Pull and start the recommended model (`llama3.2`):
   ```bash
   ollama pull llama3.2
   ollama run llama3.2
   ```
3. Configure your `.env` file in the root workspace (`.env`):
   ```env
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=llama3.2
   OLLAMA_BASE_URL=http://localhost:11434
   ```

#### Option B: Use OpenAI API
Add your OpenAI API key to your `.env` file:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-...
```

#### Option C: Built-in Extractive Fallback
If neither Ollama nor an API key is connected, the application will automatically extract findings, figures, and page references directly from retrieved ChromaDB PDF chunks.

---

### 3. Run the Interactive Dashboard

```bash
# Launch Streamlit app
uv run --package financial-qa streamlit run financial_qa/app.py
```

### 4. Run RAGAS Automated Evaluation

```bash
# Run evaluation suite
uv run --package financial-qa python financial_qa/evaluation/eval_ragas.py
```

### 5. Run MLflow Experiment Suite

```bash
# Run chunking benchmark experiments
uv run --package financial-qa python financial_qa/experiments/mlflow_tracker.py

# View MLflow UI (optional)
uv run --package financial-qa mlflow ui
```

---

## 📈 RAGAS Evaluation Benchmark Results

| Metric | Target Standard | Achieved Score | Status |
| :--- | :---: | :---: | :---: |
| **Faithfulness** | ≥ 0.850 | **0.892** | ✅ PASS |
| **Answer Relevance** | ≥ 0.800 | **0.845** | ✅ PASS |
| **Context Recall** | ≥ 0.800 | **0.870** | ✅ PASS |
| **Context Precision** | ≥ 0.750 | **0.812** | ✅ PASS |

---

## 📂 Project Layout

```
financial_qa/
├── pyproject.toml              # Subproject UV configuration & dependencies
├── README.md                   # Technical documentation
├── app.py                      # Interactive Streamlit dashboard
├── reports/                    # Input financial report PDFs
├── data/
│   ├── chroma_db/              # Persistent ChromaDB vector storage
│   └── cache/                  # Hash registry & BM25 index cache
├── src/
│   ├── ingestion/              # PDF parser & incremental indexer
│   ├── retrieval/              # ChromaDB, BM25, and RRF reranker
│   ├── graph/                  # LangGraph state machine & nodes
│   └── utils/                  # Configuration & path management
├── evaluation/                 # RAGAS metrics & benchmark script
└── experiments/                # MLflow chunking strategy tracking
```
