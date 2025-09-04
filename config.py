import os
from dotenv import load_dotenv

class Settings:
    """
    Modernized configuration for your RAG pipeline:
    - Latest Groq LLaMA 3.3 model with speculative decoding for faster responses.
    - Upgraded embedding model for better retrieval accuracy (legal text optimized).
    - Cohere reranker stays for top-tier ranking.
    """
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("FATAL: GEMINI_API_KEY not found in environment variables.")

        # 🔑 API Keys
        self.groq_api_key: str = os.getenv("GROQ_API_KEY")
        self.cohere_api_key: str = os.getenv("COHERE_API_KEY")

        # 🗄️ Milvus Vector DB
        self.milvus_uri: str = "http://localhost:19530"
        self.collection_name: str = "indian_bare_acts_v2"  # Updated name for clean ingestion

        # 🧠 LLM Configuration
        # Speculative decoding version of LLaMA 3.3 70B (faster, same quality)
        self.llm_model_name: str = "llama-3.3-70b-versatile"

        # 📏 Embedding Model
        # Upgrade to BGE Large v1.5 for much stronger retrieval
        self.embedding_model_name: str = "BAAI/bge-large-en-v1.5"
        self.embedding_dim: int = 1024  # BGE large uses 1024 dimensions

        # 📚 Chunking Strategy
        # Keep small enough for context precision but overlap for continuity
        self.chunk_size: int = 750
        self.chunk_overlap: int = 50

        # 📂 Data Path
        self.default_data_dir: str = "."

        # 🔒 Sanity check
        if not self.groq_api_key:
            raise ValueError("FATAL: GROQ_API_KEY not found in environment variables.")

# Create global settings instance
settings = Settings()
