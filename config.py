import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Qdrant Database Configuration
QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_storage")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "industrial_rag")

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Embedding Configuration
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))

# Retrieval Configuration
TOP_K = int(os.getenv("TOP_K", "20"))
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").lower() == "true"
RERANK_K = int(os.getenv("RERANK_K", "10"))

# Generator Configuration
GROQ_MODEL = os.getenv("GROQ_GENERATION_MODEL", "llama-3.3-70b-versatile")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
