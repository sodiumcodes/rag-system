from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class DocumentIndexer:

    def __init__(self, embedding_model, collection_name,documents, embeddings_model):
        #Setup embedding model
        embeddings = HuggingFaceEmbeddings(
            model = embedding_model , 
        )
        #Connect to ChromaDB
        vector_store = Chroma.from_documents(
            collection_name= collection_name,
            documents=documents,
            embedding=embeddings_model,     
            persist_directory= "llm/vectorstore",
        )
        return vector_store

    def index_documents(self, documents: list[Document]) -> int:
    
            # Ignore empty documents
            docs = [d for d in documents if d.page_content.strip()]
            if not docs:
                return 0
    
            #ChromaDB uses IDs to distinguish records
            ids = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "doc")
                ids.append(f"{source}_chunk_{i}") #eg:"resume.pdf_chunk_1"
    
            #ChromaDB's documents argument expects the text content, not LangChain Document objects.
            texts = [doc.page_content for doc in docs]
    
            #ChromaDB metadata supports basic scalar values 
            metadatas = [
                {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}
                for doc in docs
            ]
            #? Why preserve metadata? 
            #? When your retriever finds a relevant passage, you can identify its source and page number. This is especially useful when answering questions about your book.
    
    
            # Save to Chroma : upsert when the ids already exists
            self.collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
            )
    
            return len(docs)