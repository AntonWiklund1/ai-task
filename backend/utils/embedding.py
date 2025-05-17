import asyncio
import voyageai
from typing import List, Union
import logging
import os
from constants import MAX_EMBEDDING_LENGTH

logger = logging.getLogger(__name__)

class VoyageEmbeddings:
    def __init__(self):
        """Initialize the Voyage embeddings client."""
        self.client = voyageai.Client(os.getenv('VOYAGE_API_KEY'))
        self.model = "voyage-law-2"

    def embed_query(self, text: str, input_type: str = "query") -> List[float]:
        """Get embedding for a single text using a synchronous call to Voyage."""
        # Check if text exceeds maximum length
        if len(text) > MAX_EMBEDDING_LENGTH:
            logger.warning(f"Text exceeds maximum embedding length ({len(text)} > {MAX_EMBEDDING_LENGTH}). Truncating.")
            text = text[:MAX_EMBEDDING_LENGTH]
        try:
            response = self.client.embed(text, model=self.model, input_type=input_type)
            return response.embeddings[0]
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

    def get_embeddings(self, texts: Union[str, List[str]], input_type: str = "document") -> List[List[float]]:
        """Get embeddings for one or more texts synchronously using Voyage.
        
        This method now uses the batch API provided by voyageai.Client.embed.
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        logger.debug(f"Getting batch embeddings (Voyage) with input_type='{input_type}' for {len(texts)} texts")
        # Call the client embed method directly as it's now a synchronous method
        result = self.client.embed(
            texts,
            model=self.model,
            input_type=input_type,
            truncation=True
        )
        
        # result.embeddings is a list of embeddings for the batch of texts.
        return result.embeddings