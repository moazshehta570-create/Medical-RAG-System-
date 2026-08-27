import torch
class Config:
    DATA_DIR = "data"
    RAG_DATA_DIR = "rag_data"
    INDEX_PATH = f"{RAG_DATA_DIR}/medical_faiss.index"
    CHUNKS_PATH = f"{RAG_DATA_DIR}/chunks.json"
    EMBEDDING_MODEL_NAME = "sentence-transformers/embeddinggemma-300m-medical"
    EMBEDDING_DIM = 768
    LLM_MODEL_NAME = "Qwen/Qwen3-8B"
    TOP_K = 5
    MIN_SCORE = 0.35
    MAX_NEW_TOKENS = 512
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHUNK_SIZE = 850
    CHUNK_OVERLAP = 150
