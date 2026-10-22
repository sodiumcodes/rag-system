import os
from pathlib import Path
import requests
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")

class OpenRouterEmbeddings(Embeddings):
    """Generate embeddings through OpenRouter's hosted API."""

    def __init__(self, model: str):
        # Store the name of the embedding model provided by OpenRouter.
        self.model = model

        # Read the OpenRouter API key from the environment variables.
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        # Stop initialization if the API key is missing.
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is missing from .env")

        # OpenRouter's API endpoint for generating embeddings.
        self.url = "https://openrouter.ai/api/v1/embeddings"

        
    def _embed(self, texts: list[str]) -> list[list[float]]:
        # Store the embeddings returned for all input texts.
        all_embeddings = []

        # Number of texts to send in a single API request.
        batch_size = 16

        # Process the input texts in smaller batches.
        # range() generates the starting index of each batch.
        for start in range(0, len(texts), batch_size):

            # Select up to 16 texts for the current batch.
            batch = texts[start:start + batch_size]

            # Send a POST request to OpenRouter's embedding API.
            response = requests.post(
                self.url,

                # Provide authentication and specify that the body is JSON.
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },

                # Specify the embedding model and the texts to embed.
                json={
                    "model": self.model,
                    "input": batch,
                },

                # Allow up to 120 seconds for the API request.
                timeout=120,
            )

            # Check whether the API returned a successful status code.
            # If not, raise an error containing the status and response body.
            if not response.ok:
                raise RuntimeError(
                    f"OpenRouter embedding request failed "
                    f"(status {response.status_code}): {response.text}"
                )

            # Convert the successful JSON response into a Python dictionary
            # and extract the list of generated embeddings.
            data = response.json()["data"]

            # Sort the results by their original input index.
            # This ensures that embeddings match the order of the input texts.
            data.sort(key=lambda item: item["index"])

            # Extract only the numerical embedding vector from each result.
            # Each vector is a list of floating-point numbers.
            batch_embeddings = [
                item["embedding"] for item in data
            ]

            # Verify that the API returned one embedding for every input text.
            if len(batch_embeddings) != len(batch):
                raise RuntimeError(
                    f"Expected {len(batch)} embeddings, "
                    f"but received {len(batch_embeddings)}"
                )

            # Add the current batch's embeddings to the overall result.
            all_embeddings.extend(batch_embeddings)

        # Return embeddings for all texts, preserving their original order.
        return all_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Generate embeddings for multiple documents.
        # ChromaDB calls this method when documents need to be indexed.
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        # Generate an embedding for a single query.
        # Return the first (and only) embedding from the result.
        return self._embed([text])[0]


class DocumentIndexer:

    def __init__(self, embedding_model: str, collection_name: str):
        # Use OpenRouter instead of a local Sentence Transformers model.
        self.embeddings = OpenRouterEmbeddings(embedding_model)

        # Connect to persistent ChromaDB storage.
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(ROOT / "llm" / "data" / "vectorstore"),
        )

    def index_documents(self, documents: list[Document]) -> int:
        # Ignore empty chunks.
        docs = [
            doc for doc in documents
            if doc.page_content.strip()
        ]

        if not docs:
            return 0

        # Create IDs using the source and chunk position.
        ids = []
        for i, doc in enumerate(docs):
            source = Path(
                str(doc.metadata.get("source", "doc"))
            ).stem
            page = doc.metadata.get("page", "unknown")
            ids.append(f"{source}_page_{page}_chunk_{i}")

        # LangChain calls OpenRouter to embed the documents,
        # then stores the vectors, text, and metadata in Chroma.
        self.vector_store.add_documents(
            documents=docs,
            ids=ids,
        )

        return len(docs)

    def count(self) -> int:
        return self.vector_store._collection.count()
