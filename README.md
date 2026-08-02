This repository contains the projects, which are related to RAG(Retrieval Augmented Generation) 
# Retrieval-Augmented Generation (RAG) Projects

Welcome to my central repository for all projects, experiments, and implementations related to **Retrieval-Augmented Generation (RAG)**. 

This repository serves as a workspace to explore different RAG architectures, chunking strategies, embedding models, vector databases, and advanced retrieval techniques (like Hybrid Search and Re-ranking).

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI framework that improves the quality, accuracy, and reliability of Large Language Models (LLMs) by grounding them in external, up-to-date, or proprietary knowledge bases. 

Standard LLMs rely solely on their internal, pre-trained parametric memory. If they don't know the answer, they often hallucinate. RAG solves this by adding an information retrieval step: it searches a database for the exact facts needed, and passes those facts to the LLM to formulate the final answer.

### How it Works (The Short Version)
1. **Index:** Your private data (PDFs, docs, databases) is split into chunks, converted into numbers (embeddings), and stored in a Vector Database.
2. **Retrieve:** When a user asks a question, the system searches the Vector Database for the most semantically relevant text chunks.
3. **Generate:** The retrieved text chunks are combined with the user's original question and sent to the LLM. The LLM synthesizes this context to generate a highly accurate, grounded response.

---
## Basic pipeline of RAG

<img width="720" height="651" alt="1_FhMJ8OE_PoeOyeAavYjzlw" src="https://github.com/user-attachments/assets/efef644e-c6d2-4985-a6bd-2af9d80c8bd8" />

### Technical Architecture of a RAG System

<img width="2752" height="1536" alt="Gemini_Generated_Image_k51mu7k51mu7k51m" src="https://github.com/user-attachments/assets/c7e64286-9a2c-47b3-843c-9415e9db652e" />




## Pros and Cons of RAG

### Pros
* **Reduces Hallucinations:** The LLM's responses are grounded in factual, retrieved context rather than "best guesses."
* **Access to Private & Fresh Data:** You can query proprietary company documents or real-time data without having to expose that data to public training sets.
* **Cost-Effective:** It bypasses the need for expensive and time-consuming model fine-tuning or retraining. You just update your database.
* **Traceability & Auditing:** Because the system retrieves specific documents, you can provide exact source citations (e.g., "According to Document A...").

### Cons
* **Added Latency:** The extra steps of embedding the query and searching the vector database add response time compared to a simple LLM API call.
* **Architectural Complexity:** You must maintain a multi-component pipeline (Loaders, Splitters, Embedding Models, Vector DBs, LLMs).
* **Dependent on Retrieval Quality (GIGO):** If your chunking strategy is bad or your embedding model misses the right document, the LLM will still fail to answer correctly.
* **Context Window Limits:** You can only feed a limited number of retrieved documents into the LLM's prompt before it loses focus or hits token limits.

## 🛠️ Workspace Setup & Quick Start

### 1. Installation & Environment Activation
Install dependencies with `uv` and activate your virtual environment:

```bash
# Sync dependencies across all workspace projects
uv sync

# Activate the virtual environment (.venv)
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.\.venv\Scripts\activate.bat

# On macOS/Linux:
source .venv/bin/activate
```

### 2. Running an LLM (Ollama or API Keys)

#### Local Ollama Model (Recommended)
1. Download and install [Ollama](https://ollama.com/).
2. Pull and start your preferred model (e.g. `llama3.2`):
   ```bash
   ollama pull llama3.2
   ollama run llama3.2
   ```
3. Set your provider parameters in `.env`:
   ```env
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=llama3.2
   OLLAMA_BASE_URL=http://localhost:11434
   ```

#### Cloud LLM (OpenAI)
Add your OpenAI key to `.env`:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-...
```

---

## 📌 Projects in this Repository

- **[financial_qa](file:///c:/Data/Practice/github_projects/rag_projects/financial_qa)**: State-of-the-art Agentic RAG System for Financial Reports using LangGraph, ChromaDB, BM25 Hybrid RRF, RAGAS, and Streamlit.
- **[ai_article](file:///c:/Data/Practice/github_projects/rag_projects/ai_article)**: AI article summarization and retrieval pipelines.