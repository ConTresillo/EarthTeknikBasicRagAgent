import logging
from google import genai
from google.genai import types
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import config

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment variables.")
        
        # Initialize the new google-genai client
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = config.GEMINI_EMBEDDING_MODEL
        self.dimensions = config.EMBEDDING_DIMENSIONS

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(APIError),
        reraise=True
    )
    def generate_query_embedding(self, query: str) -> list[float]:
        """
        Generate an embedding for a retrieval query using Gemini Embedding 2.
        Includes retry logic for quota/rate limit errors.
        """
        try:
            response = self.client.models.embed_content(
                model=self.model,
                contents=[query],
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=self.dimensions
                )
            )
            # Depending on SDK version, it could be response.embeddings[0].values
            return response.embeddings[0].values
        except APIError as e:
            logger.warning(f"Gemini API rate limit or error encountered: {e}. Retrying...")
            raise
