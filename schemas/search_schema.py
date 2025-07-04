from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

from schemas.chunk_schema import ChunkResponse


class IndexAlgorithm(str, Enum):
    """Supported vector index algorithms."""
    FLAT = "flat"           # Simple linear search - best for small datasets
    LSH = "lsh"             # Locality Sensitive Hashing - good for large datasets  
    GRID = "grid"           # Grid-based partitioning - balanced approach


class SearchRequest(BaseModel):
    """Request model for k-NN similarity search."""
    query: str = Field(..., description="Text to search for", min_length=1)
    k: int = Field(default=10, description="Number of top results to return", ge=1, le=100)


class SearchResult(BaseModel):
    """Individual search result."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    similarity_score: float = Field(..., description="Similarity score (0-1, higher is more similar)")
    chunk: ChunkResponse = Field(..., description="Full chunk data")


class SearchResponse(BaseModel):
    """Response model for k-NN similarity search."""
    query: str = Field(..., description="Original search query")
    library_id: str = Field(..., description="Library that was searched")
    results: List[SearchResult] = Field(..., description="Search results ordered by similarity")
    total_found: int = Field(..., description="Total number of results found")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")


class LibraryIndexInfo(BaseModel):
    """Information about a library's vector index."""
    library_id: str = Field(..., description="Library identifier")
    algorithm: str = Field(..., description="Index algorithm being used")
    is_built: bool = Field(..., description="Whether the index is built and ready")
    vector_count: int = Field(..., description="Number of vectors in the index")
    dimension: Optional[int] = Field(None, description="Vector dimension")
    index_stats: Dict[str, Any] = Field(default_factory=dict, description="Algorithm-specific statistics")


class IndexAlgorithmRequest(BaseModel):
    """Request to set or change a library's index algorithm."""
    algorithm: IndexAlgorithm = Field(..., description="Algorithm to use for this library")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Algorithm-specific parameters"
    )


class IndexAlgorithmResponse(BaseModel):
    """Response after setting a library's index algorithm."""
    library_id: str = Field(..., description="Library identifier")
    algorithm: str = Field(..., description="Algorithm that was set")
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Status message")
    index_info: Optional[LibraryIndexInfo] = Field(None, description="Updated index information") 