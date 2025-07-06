import os
import asyncio
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import time

import cohere
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import numpy as np
from decorators import logger, timer
class_logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding service."""
    api_key: str
    model: str = "embed-english-light-v3.0"  # 384 dimensions, faster and more efficient
    input_type: str = "search_document"  # For indexing documents
    batch_size: int = 96  # Cohere's recommended batch size
    max_retries: int = 3
    timeout: int = 30
    rate_limit_delay: float = 1.0  # Seconds between requests


class EmbeddingError(Exception):
    """Base exception for embedding service errors."""
    pass


class EmbeddingAPIError(EmbeddingError):
    """API-related embedding errors."""
    pass


class EmbeddingRateLimitError(EmbeddingError):
    """Rate limit exceeded error."""
    pass


class EmbeddingService:
    """
    Service for generating text embeddings using Cohere SDK.
    
    Features:
    - Async/await support for FastAPI integration
    - Batch processing for efficiency
    - Automatic retries with exponential backoff
    - Rate limiting and error handling
    - Configurable model and parameters
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedding service.
        
        Args:
            config: EmbeddingConfig object, or None to use environment variables
        """
        self.config = config or self._load_config_from_env()
        self._client: Optional[cohere.AsyncClient] = None
        
        # Validate configuration
        if not self.config.api_key:
            raise EmbeddingError("Cohere API key is required. Set COHERE_API_KEY environment variable.")
        
        class_logger.info(f"EmbeddingService initialized with model: {self.config.model}")
    
    def _load_config_from_env(self) -> EmbeddingConfig:
        """Load configuration from environment variables."""
        api_key = os.getenv("COHERE_API_KEY", "")
        model = os.getenv("COHERE_MODEL", "embed-english-light-v3.0")
        batch_size = int(os.getenv("COHERE_BATCH_SIZE", "96"))
        max_retries = int(os.getenv("COHERE_MAX_RETRIES", "3"))
        timeout = int(os.getenv("COHERE_TIMEOUT", "30"))
        
        return EmbeddingConfig(
            api_key=api_key,
            model=model,
            batch_size=batch_size,
            max_retries=max_retries,
            timeout=timeout
        )
    
    def _get_client(self) -> cohere.AsyncClient:
        """Get or create Cohere client."""
        if self._client is None:
            self._client = cohere.AsyncClient(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
        return self._client
    
    async def close(self):
        """Close the client (placeholder for future async client)."""
        # Current Cohere SDK doesn't require explicit closing
        # But we keep this for consistency with async patterns
        self._client = None
    
    @logger
    @timer
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((EmbeddingAPIError, EmbeddingRateLimitError))
    )
    async def _call_cohere_api(self, texts: List[str]) -> List[List[float]]:
        """
        Call Cohere API to generate embeddings using the official SDK.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            EmbeddingAPIError: For API errors
            EmbeddingRateLimitError: For rate limit errors
        """
        if not texts:
            return []
        
        # Validate input
        if len(texts) > self.config.batch_size:
            raise EmbeddingError(f"Batch size {len(texts)} exceeds maximum {self.config.batch_size}")
        
        # Check for empty texts
        if any(not text or not text.strip() for text in texts):
            raise EmbeddingError("Cannot embed empty or whitespace-only texts")
        
        client = self._get_client()
        
        try:
            # Call the async client directly
            response = await client.embed(
                texts=texts,
                model=self.config.model,
                input_type=self.config.input_type
            )
            
            # Extract embeddings from response
            embeddings = response.embeddings
            
            if not embeddings:
                raise EmbeddingAPIError("No embeddings returned from API")
            
            if len(embeddings) != len(texts):
                raise EmbeddingAPIError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
            
            class_logger.debug(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except cohere.CohereAPIError as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                class_logger.warning(f"Rate limited: {e}")
                await asyncio.sleep(self.config.rate_limit_delay)
                raise EmbeddingRateLimitError("Rate limit exceeded")
            else:
                class_logger.error(f"Cohere API error: {e}")
                raise EmbeddingAPIError(f"API error: {e}")
        except Exception as e:
            class_logger.error(f"Unexpected error calling Cohere API: {e}")
            raise EmbeddingAPIError(f"Unexpected error: {e}")
    
    @logger
    @timer
    async def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            np.ndarray: Embedding vector as numpy array
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty or whitespace-only text")
        
        embeddings = await self._call_cohere_api([text])
        return np.array(embeddings[0], dtype=np.float32)
    

    @timer
    async def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[np.ndarray]: List of embedding vectors as numpy arrays
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []
        
        # Validate all texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise EmbeddingError(f"Text at index {i} is empty or whitespace-only")
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            # Add delay between batches to respect rate limits
            if i > 0:
                await asyncio.sleep(self.config.rate_limit_delay)
            
            batch_embeddings = await self._call_cohere_api(batch)
            # Convert to numpy arrays
            batch_arrays = [np.array(emb, dtype=np.float32) for emb in batch_embeddings]
            all_embeddings.extend(batch_arrays)
            
            class_logger.debug(f"Processed batch {i//self.config.batch_size + 1}: {len(batch)} texts")
        
        class_logger.info(f"Generated {len(all_embeddings)} embeddings for {len(texts)} texts")
        return all_embeddings
    
    @logger
    @timer
    async def generate_query_embedding(self, query_text: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        Args:
            query_text: Query text to embed
            
        Returns:
            np.ndarray: Query embedding vector as numpy array
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not query_text or not query_text.strip():
            raise EmbeddingError("Cannot embed empty or whitespace-only query")
        
        # Use different input_type for queries
        original_input_type = self.config.input_type
        
        try:
            # Temporarily change input type for query
            self.config.input_type = "search_query"
            embedding = await self.generate_embedding(query_text)
            return embedding
        finally:
            # Restore original input type
            self.config.input_type = original_input_type

# Global instance for application use
embedding_service = EmbeddingService() 