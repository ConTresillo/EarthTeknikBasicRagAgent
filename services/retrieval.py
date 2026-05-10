from qdrant_client import QdrantClient
import config
from services.embeddings import EmbeddingService

class RetrievalService:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        # Connect to existing local Qdrant storage
        self.client = QdrantClient(path=config.QDRANT_PATH)

    def retrieve(self, query: str, top_k: int = None):
        if top_k is None:
            top_k = config.TOP_K
            
        # Generate embedding via Gemini API
        query_vector = self.embedding_service.generate_query_embedding(query)
        
        # Search Qdrant using the unified query_points API
        response = self.client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )
        
        # Format the results from the .points attribute
        results = []
        for hit in response.points:
            results.append({
                "score": hit.score,
                "text": hit.payload.get("text", "") if hit.payload else str(hit.payload),
                "metadata": hit.payload
            })
            
        return results
