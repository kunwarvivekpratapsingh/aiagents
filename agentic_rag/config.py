import os
from dataclasses import dataclass, field


@dataclass
class Config:
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model: str = "claude-sonnet-4-6"
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "agentic_rag_docs"
    max_iterations: int = 3
    top_k_retrieval: int = 5
    max_tokens: int = 2048
    chunk_size: int = 400
    chunk_overlap: int = 60


config = Config()
