import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLLECTION_NAME = "Book-embeddings"
CHROMA_DIR = PROJECT_ROOT / "llm" / "data" / "vector_store"