# import statements
from pathlib import Path
import chromadb
import numpy as np
from llm.src.config import EMBEDDING_MODEL, PROJECT_ROOT, COLLECTION_NAME, CHROMA_DIR
from llm.src.indexing.indexer import OpenRouterEmbeddings
from rank_bm25 import BM25Okapi
# BM25Okapi is a Python implementation of the BM25 keyword-search algorithm. It helps find documents based on how well their words match the user's query.

# document retriever class
class DocumentRetriever : 
    # _init_ function
    def __init__(self):

        # Connect to the persistent ChromaDB database
        # creating a local instance
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        # Access the existing collection
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        # Initialize the same embedding model used during indexing
        self.embed_func = OpenRouterEmbeddings(EMBEDDING_MODEL)

        # This initializes the BM25 cache so hybrid retrieval can run correctly.
        self._bm25 = None
        self._bm25_documents = []

    # since we have different searching mechanisms, we will use a helper function to format the output, so that all outputs are of the same format
    def format_output (self, results: dict) -> list[dict]:
        #* Why [[]] instead of []? 
        #* Because the code immediately accesses [0]. The default [[]] ensures that accessing the first element gives you an empty list rather than raising an IndexError.
        documents = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []
        ids       = results.get("ids", [[]])[0] or [] 

        formatted = []

        for i, text in enumerate(documents):
            formatted.append({
                "id": ids[i] if i < len(ids) else None,
                "text": text,
                "metadata": metadatas[i] or {}
                if i < len(metadatas) else {},
                "distance": distances[i]
                if i < len(distances) else None,
            })

        return formatted        

    # searching 1 : Dense Similarity Search
    def similarity_search ( self, query: str, top_k: int = 5 ) -> list[dict] : 
        """Retrieve the closest chunks using dense embeddings."""

        # check if query is empty
        if not query.strip() :
            return []
        
        # check if there are documents present in db
        total_docs = self.collection.count()
        if total_docs == 0 :
            return []
        
        #embed user query
        query_vector = self.embed_func.embed_query(query) 

        # Search ChromaDB using the query vector
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(max(top_k, 1), total_docs),
            include=["documents", "metadatas", "distances"],
        )     
        return self.format_output(results= results) 

    # searching 2 : Similarity Search with a Distance Threshold
    def threshold_search(
        self, query:str, top_k:int = 5, max_distance: float = 0.7 
    ) -> list[dict] :
        """Return only dense-search results within the distance cutoff."""

        if not query.strip():
            return []
        if max_distance < 0:
            raise ValueError("max_distance must be non-negative.")

        results = self.similarity_search(query, top_k)

        return [
            result
                for result in results
                if result["distance"] is not None
                and result["distance"] <= max_distance
        ]

    # searching 3 : Hybrid Search: Dense Search + BM25
    def _get_bm25(self):
        """Build and cache a BM25 index from the stored chunks."""

        if self._bm25 is not None:
            return self._bm25

        data = self.collection.get(
            include=["documents", "metadatas"]
        )

        documents = data["documents"] or []
        metadatas = data["metadatas"] or []
        ids = data["ids"] or []

        self._bm25_documents = [
            {
                "id": ids[i],
                "text": text,
                "metadata": metadatas[i] or {},
                "distance": None,
            }
            for i, text in enumerate(documents)
        ]

        # Simple tokenization for keyword retrieval
        tokenized_docs = [
            doc["text"].lower().split()
            for doc in self._bm25_documents
        ]

        if not tokenized_docs:
            return None

        self._bm25 = BM25Okapi(tokenized_docs)

        return self._bm25

    def hybrid_search(
        self, query:str, top_k:int = 5, fetch_k: int = 20, rrf_k: int = 60,
    ) -> list[dict] :

        """Combine dense semantic search and BM25 using RRF."""

        if not query.strip():
            return []
        total_docs = self.collection.count()
        if total_docs == 0:
            return []
        
        fetch_k = min(max(fetch_k, top_k, 1), total_docs)
        top_k = min(max(top_k, 1), total_docs)
        
        # Get dense-search rankings
        dense_results = self.similarity_search(query, fetch_k)

        # Get keyword-search rankings
        bm25 = self._get_bm25()

        if bm25 is None:
            return dense_results[:top_k]

        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)

        bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:fetch_k]

        bm25_results = [
            self._bm25_documents[i] for i in bm25_indices
        ]

        # Reciprocal Rank Fusion combines rankings without
        # requiring dense distances and BM25 scores to be comparable.
        fused_scores = {}
        result_by_id = {}

        for ranking in (dense_results, bm25_results):
            for rank, result in enumerate(ranking, start=1):
                doc_id = result["id"]

                if doc_id is None:
                    continue

                fused_scores[doc_id] = (
                    fused_scores.get(doc_id, 0.0)
                    + 1.0 / (rrf_k + rank)
                )
                result_by_id[doc_id] = result

        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )        

        output = []
        for doc_id in ranked_ids[:top_k]:
            result = result_by_id[doc_id].copy()
            result["rrf_score"] = fused_scores[doc_id]
            output.append(result)

        return output 