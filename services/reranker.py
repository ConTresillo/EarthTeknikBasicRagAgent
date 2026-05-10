import config
import logging

logger = logging.getLogger(__name__)

class RerankerService:
    def __init__(self, enabled: bool = False):
        self.enabled = False # Force disabled
        self.model = None
        # if self.enabled:
        #     from sentence_transformers import CrossEncoder
        #     logger.info("Loading cross-encoder model for reranking...")
        #     self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def rerank(self, query: str, documents: list, rerank_k: int = 5):
        # Fallback to just returning the top documents without reranking
        return documents[:rerank_k]
