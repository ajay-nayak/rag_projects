import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# === Chunking Strategy ===
def recursive_chunking(structured_data, max_chunk_size=512, overlap=100):
    """Recursively chunks structured data while preserving section integrity."""
    chunks = []
    for section, subsections in structured_data.items():
        section_text = f"### {section} ###\n"
        for subsection, content in subsections.items():
            full_text = section_text + f"## {subsection} ##\n{content}"
            if len(full_text) <= max_chunk_size:
                chunks.append(full_text)
            else:
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', full_text)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= max_chunk_size:
                        current_chunk += sentence + " "
                    else:
                        chunks.append(current_chunk.strip())
                        overlap_text = " ".join(current_chunk.split()[-(overlap // 10):])
                        current_chunk = overlap_text + " " + sentence + " "
                if current_chunk:
                    chunks.append(current_chunk.strip())
    return chunks

# Prepare FAISS
def embed_text_chunks(text_chunks, model_name="all-MiniLM-L6-v2"):
    """Converts text chunks into embeddings."""
    model = SentenceTransformer(model_name)
    embeddings = np.array([model.encode(chunk) for chunk in text_chunks])
    return model, embeddings

def store_embeddings_in_faiss(embeddings):
    """Stores embeddings in FAISS for fast retrieval."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# BM25 Retrieval
def prepare_bm25_index(text_chunks):
    """Prepares a BM25 index for keyword-based retrieval."""
    tokenized_corpus = [word_tokenize(chunk.lower()) for chunk in text_chunks]
    return BM25Okapi(tokenized_corpus)

def retrieve_relevant_text(query, model, faiss_index, text_chunks, top_k=3):
    """Retrieves relevant text chunks using FAISS similarity search."""
    if faiss_index.ntotal == 0:
        return ["⚠️ No data in FAISS. Please check embeddings."]

    query_embedding = model.encode(query).reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    retrieved_texts = [text_chunks[i] for i in indices[0] if i < len(text_chunks)]
    return retrieved_texts if retrieved_texts else ["⚠️ No relevant results found."]


def multi_stage_retrieval(query, model, faiss_index, bm25_index, text_chunks, top_k=5):
    """Multi-stage retrieval using FAISS + BM25."""
    
    if faiss_index.ntotal == 0 or len(text_chunks) == 0:
        return ["⚠️ No indexed data found."]

    query_embedding = model.encode(query).reshape(1, -1)
    faiss_distances, faiss_indices = faiss_index.search(query_embedding, top_k)
    retrieved_faiss_texts = [text_chunks[i] for i in faiss_indices[0] if i < len(text_chunks)]

    tokenized_query = word_tokenize(query.lower())
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[-top_k:]
    retrieved_bm25_texts = [text_chunks[i] for i in bm25_top_indices if i < len(text_chunks)]

    combined_results = list(dict.fromkeys(retrieved_faiss_texts + retrieved_bm25_texts))
    return combined_results[:top_k] if combined_results else ["⚠️ No relevant results found."]
