import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from root or local dir
load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
INDEX_CACHE_DIR = DATA_DIR / "cache"

# Ensure essential directories exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Embedding Settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-small")

# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # options: ollama, openai, gemini, phi2
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Chunking Configuration Defaults
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 5
