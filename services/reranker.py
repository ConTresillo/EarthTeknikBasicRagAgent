import config
import logging

logger = logging.getLogger(__name__)

class RerankerService:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.model = None
        if self.enabled:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder model for reranking...")
            self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def rerank(self, query: str, documents: list, rerank_k: int = None):
        if rerank_k is None:
            rerank_k = config.RERANK_K
            
        if not self.enabled or not self.model or not documents:
            return documents[:rerank_k]
            
        # Prepare pairs for the cross-encoder
        pairs = [[query, doc["text"]] for doc in documents]
        
        # Predict relevance scores
        scores = self.model.predict(pairs)
        
        # Add rerank scores to documents
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])
            
        # Sort by rerank score descending
        documents.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return documents[:rerank_k]
