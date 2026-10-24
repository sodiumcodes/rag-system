import os
from dotenv import load_dotenv
load_dotenv()

from ..src.config import EMBEDDING_MODEL, PROJECT_ROOT, COLLECTION_NAME
from llm.src.ingestion.loaders import load_pdf
from llm.src.indexing.chunking import clean_documents ,chunk_documents
from llm.src.indexing.indexer import DocumentIndexer

FILE_PATH = PROJECT_ROOT /"llm"/"data" / "raw" / "Explainable-AI-for-Practitioners.pdf"

def main():
    documents = load_pdf(FILE_PATH)
    print(f"Pages loaded: {len(documents)}")
    cleaned_docs = clean_documents(documents=documents)
    chunks = chunk_documents(cleaned_docs, chunk_size=1000, chunk_overlap=100)
    print(f"chunks created : {len(chunks)}")

    indexer = DocumentIndexer(embedding_model =EMBEDDING_MODEL, collection_name= COLLECTION_NAME )

    indexed_count = indexer.index_documents(chunks)

    print(f"Chunks processed : {indexed_count}")
    print(f"Total chunks in databse : {indexer.count()}")

if __name__ == "__main__":
    main()
