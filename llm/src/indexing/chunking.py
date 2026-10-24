import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def clean_documents(
    documents: list[Document]
) -> list[Document]:

    cleaned_documents = []

    for doc in documents:
        text = doc.page_content

        # Replace tabs and non-breaking spaces with normal spaces.
        text = text.replace("\t", " ")
        text = text.replace("\u00a0", " ")

        # Remove spaces at the beginning and end of lines.
        text = "\n".join(
            line.strip() for line in text.splitlines()
        )

        # Replace multiple spaces with a single space.
        text = re.sub(r" {2,}", " ", text)

        # Preserve paragraph breaks, but remove excessive blank lines.
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove leading and trailing whitespace.
        text = text.strip()

        if text:
            cleaned_documents.append(
                Document(
                    page_content=text,
                    metadata=doc.metadata.copy()
                )
            )

    return cleaned_documents

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