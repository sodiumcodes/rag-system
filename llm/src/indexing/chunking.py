from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int
) -> list[Document]:
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  #Maximum number of characters in each chunk
        chunk_overlap=chunk_overlap,  #Number of characters shared between consecutive chunks
        separators=["\n\n", "\n", ". ", " ", ""],  #splitting conditions

        length_function=len,  #Measure chunk size using the number of characters
        add_start_index=True  #Store each chunk's starting character index in its metadata
    )

    # Split the documents into smaller Document objects while preserving their metadata
    chunks = text_splitter.split_documents(documents)
    return chunks