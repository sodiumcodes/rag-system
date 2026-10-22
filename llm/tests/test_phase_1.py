import os
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from llm.src.ingestion.loaders import load_pdf
from llm.src.indexing.chunking import chunk_documents
from llm.src.indexing.indexer import DocumentIndexer

FILE_PATH = ROOT /"llm"/"data" / "raw" / "Explainable-AI-for-Practitioners.pdf"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
def main():
    documents = load_pdf(FILE_PATH)
    print(f"Pages loaded: {len(documents)}")

    chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=100)
    print(f"chunks created : {len(chunks)}")

    indexer = DocumentIndexer(embedding_model =EMBEDDING_MODEL, collection_name= "Book-embeddings")

    indexed_count = indexer.index_documents(chunks)

    print(f"Chunks processed : {indexed_count}")
    print(f"Total chunks in databse : {indexer.count()}")

if __name__ == "__main__":
    main()
