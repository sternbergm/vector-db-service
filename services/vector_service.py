import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from services.embedding_service import embedding_service, EmbeddingError
from services.chunk_service import ChunkService
from vector_db.storage import vector_storage
from vector_db.algorithms import VectorIndex, FlatIndex, LSHIndex, GridIndex
from schemas.chunk_schema import ChunkResponse, ChunkMetadata
from schemas.search_schema import IndexAlgorithm, SearchResponse, SearchResult, LibraryIndexInfo, SimilarityFunction, SearchRequest
from exceptions import LibraryNotFoundError, ChunkNotFoundError, DatabaseError
from decorators import logger, timer
class_logger = logging.getLogger(__name__)


class VectorIndexConfig:
    """Configuration for different index algorithms."""
    
    @staticmethod
    def create_index(algorithm: IndexAlgorithm, **kwargs) -> VectorIndex:
        """Factory method to create index instances."""
        if algorithm == IndexAlgorithm.FLAT:
            similarity_metric = kwargs.get("similarity_metric", "cosine")
            return FlatIndex(similarity_metric=similarity_metric)
        
        elif algorithm == IndexAlgorithm.LSH:
            num_hashes = kwargs.get("num_hashes", 10)
            seed = kwargs.get("seed", 42)
            return LSHIndex(num_hashes=num_hashes, seed=seed)
        
        elif algorithm == IndexAlgorithm.GRID:
            cell_size = kwargs.get("cell_size", 0.1)
            similarity_metric = kwargs.get("similarity_metric", "euclidean")
            return GridIndex(cell_size=cell_size, similarity_metric=similarity_metric)
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")


class VectorService:
    """
    Service for managing vector operations across libraries.
    
    Features:
    - Library-specific vector indexes
    - Algorithm selection and switching
    - Automatic index rebuilding on changes
    - Integration with embedding service and vector storage
    """
    
    def __init__(self, chunk_service: ChunkService, library_service=None):
        """
        Initialize vector service.
        
        Args:
            chunk_service: Service for chunk database operations
            library_service: Service for library operations (optional, for circular dependency)
        """
        self.chunk_service = chunk_service
        self.library_service = library_service
        
        # Library-specific indexes: library_id -> VectorIndex
        self._library_indexes: Dict[str, VectorIndex] = {}
        
        # Track which algorithm each library uses
        self._library_algorithms: Dict[str, IndexAlgorithm] = {}
        
        # Default algorithm for new libraries
        self.default_algorithm = IndexAlgorithm.FLAT
        
        class_logger.info("VectorService initialized")
    
    def set_library_service(self, library_service):
        """Set library service (for circular dependency resolution)"""
        self.library_service = library_service
    
    @logger
    @timer
    async def add_chunk_vector(self, chunk: ChunkResponse) -> bool:
        """
        Add a chunk vector to the library index.
        
        Args:
            chunk: ChunkResponse object with all chunk data
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            EmbeddingError: If embedding generation fails
            DatabaseError: If database operations fail
        """
        try:
            # 1. Generate embedding
            embedding = await embedding_service.generate_embedding(chunk.text)
            
            # 2. Add to vector storage
            success = vector_storage.add_vector(chunk.id, embedding, chunk.library_id, chunk.metadata)
            if not success:
                class_logger.error(f"Failed to add vector to storage for chunk {chunk.id}")
                return False
            
            # 3. Rebuild library index
            await self._rebuild_library_index(chunk.library_id)
            
            class_logger.info(f"Added vector for chunk {chunk.id} to library {chunk.library_id}")
            return True
            
        except EmbeddingError:
            class_logger.error(f"Failed to generate embedding for chunk {chunk.id}")
            raise
        except Exception as e:
            class_logger.error(f"Failed to add chunk vector {chunk.id}: {str(e)}")
            return False
    
    @logger
    @timer
    async def update_chunk_vector(self, chunk: ChunkResponse) -> bool:
        """
        Update a chunk vector in the library index.
        
        Args:
            chunk: ChunkResponse object with updated chunk data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # 1. Generate new embedding
            embedding = await embedding_service.generate_embedding(chunk.text)
            
            # 2. Remove old vector and add new one
            vector_storage.remove_vector(chunk.id)
            success = vector_storage.add_vector(chunk.id, embedding, chunk.library_id, chunk.metadata)
            if not success:
                class_logger.error(f"Failed to update vector in storage for chunk {chunk.id}")
                return False
            
            # 3. Rebuild library index
            await self._rebuild_library_index(chunk.library_id)
            
            class_logger.info(f"Updated vector for chunk {chunk.id} in library {chunk.library_id}")
            return True
            
        except EmbeddingError:
            class_logger.error(f"Failed to generate embedding for updated chunk {chunk.id}")
            raise
        except Exception as e:
            class_logger.error(f"Failed to update chunk vector {chunk.id}: {str(e)}")
            return False
    
    @logger
    @timer
    async def remove_chunk_vector(self, chunk_id: str, library_id: str) -> bool:
        """
        Remove a chunk vector from the library index.
        
        Args:
            chunk_id: Unique chunk identifier
            library_id: Library this chunk belongs to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # 1. Remove from vector storage
            success = vector_storage.remove_vector(chunk_id)
            if not success:
                class_logger.warning(f"Chunk {chunk_id} not found in vector storage")
            
            # 2. Rebuild library index
            await self._rebuild_library_index(library_id)
            
            class_logger.info(f"Removed vector for chunk {chunk_id} from library {library_id}")
            return True
            
        except Exception as e:
            class_logger.error(f"Failed to remove chunk vector {chunk_id}: {str(e)}")
            return False
    
    @logger
    @timer
    async def search_similar_chunks(self, 
                                  library_id: str, 
                                  search_request: SearchRequest
                                  ) -> SearchResponse:
        """
        Search for similar chunks in a library.
        
        Args:
            library_id: Library to search in
            similarity_function: Similarity function to use for comparison
            
        Returns:
            SearchResponse: Complete search response with results and metadata
            
        Raises:
            EmbeddingError: If query embedding generation fails
            LibraryNotFoundError: If library has no index
        """
        start_time = time.time()
        
        try:
            # 1. First, validate that the library exists
            if self.library_service:
                # Use library service to validate existence - this will throw LibraryNotFoundError if library doesn't exist
                library_exists = await self.library_service.library_exists(library_id)
                if not library_exists:
                    raise LibraryNotFoundError(library_id)
            
            # 2. Determine library's preferred algorithm
            library_algorithm = None
            if self.library_service:
                try:
                    library_algorithm = await self.library_service.get_preferred_algorithm(library_id)
                except LibraryNotFoundError:
                    # Library doesn't exist - raise the error instead of silently handling it
                    raise LibraryNotFoundError(library_id)
            else:
                library_algorithm = self.default_algorithm
            
            # 3. Check if library has an index or build one with its preferred algorithm
            if library_id not in self._library_indexes:
                # Try to build index if library exists and has chunks
                chunk_ids = vector_storage.filter_by_library(library_id)
                if not chunk_ids:
                    class_logger.warning(f"Library {library_id} has no chunks for search")
                    return SearchResponse(
                        query=search_request.query,
                        library_id=library_id,
                        results=[],
                        total_found=0,
                        search_time_ms=(time.time() - start_time) * 1000
                    )
                
                class_logger.info(f"Building index for library {library_id} with preferred algorithm {library_algorithm.value if library_algorithm else 'default'}")
                # Build index with library's preferred algorithm
                await self._rebuild_library_index(library_id, library_algorithm)
            

            
            # 4. If using a different algorithm for search, create temporary index
            index_to_use = self._library_indexes[library_id]
            
            # 5. Generate query embedding
            query_embedding = await embedding_service.generate_query_embedding(search_request.query)
            
            # 6. Search using the determined index
            raw_results = index_to_use.search(query_embedding, search_request.k, search_request.similarity_function.value if search_request.similarity_function else "cosine")
            
            # 6. Convert raw results to SearchResult objects with full chunk data using batch fetch
            search_results = []
            if raw_results:
                # Extract chunk IDs and create similarity score mapping
                chunk_ids = [chunk_id for chunk_id, _ in raw_results]
                similarity_scores = {chunk_id: score for chunk_id, score in raw_results}
                
                # Batch fetch all chunks at once
                chunks = await self.chunk_service.get_chunks_batch(chunk_ids)
                
                # Create search results with similarity scores
                for chunk in chunks:
                    if chunk.id in similarity_scores:
                        search_result = SearchResult(
                            chunk_id=chunk.id,
                            similarity_score=similarity_scores[chunk.id],
                            chunk=chunk
                        )
                        search_results.append(search_result)
                
                # Log if any chunks were found in index but not in database
                found_chunk_ids = {chunk.id for chunk in chunks}
                missing_chunk_ids = set(chunk_ids) - found_chunk_ids
                if missing_chunk_ids:
                    class_logger.warning(f"Chunks {missing_chunk_ids} found in index but not in database")
            
            # 7. Sort results by similarity score (highest first)
            search_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            search_time_ms = (time.time() - start_time) * 1000
            
            class_logger.info(f"Found {len(search_results)} similar chunks in library {library_id} using {library_algorithm.value if library_algorithm else 'default'} algorithm with {search_request.similarity_function.value if search_request.similarity_function else 'cosine'} similarity in {search_time_ms:.1f}ms")
            
            return SearchResponse(
                query=search_request.query,
                library_id=library_id,
                results=search_results,
                total_found=len(search_results),
                search_time_ms=search_time_ms
            )
            
        except EmbeddingError:
            class_logger.error(f"Failed to generate query embedding: {search_request.query}")
            raise
        except LibraryNotFoundError:
            class_logger.error(f"Library {library_id} not found")
            raise
        except Exception as e:
            class_logger.error(f"Failed to search library {library_id}: {str(e)}")
            # Return empty response on error
            return SearchResponse(
                query=search_request.query,
                library_id=library_id,
                results=[],
                total_found=0,
                search_time_ms=(time.time() - start_time) * 1000
            )
    
    @logger
    @timer
    async def set_library_algorithm(self, 
                                  library_id: str, 
                                  algorithm: IndexAlgorithm,
                                  **algorithm_params) -> bool:
        """
        Set or change the indexing algorithm for a library.
        
        Args:
            library_id: Library to update
            algorithm: New algorithm to use
            **algorithm_params: Algorithm-specific parameters
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # 1. Remove old index if exists
            if library_id in self._library_indexes:
                del self._library_indexes[library_id]
                class_logger.info(f"Removed old index for library {library_id}")
            
            # 2. Update algorithm tracking
            self._library_algorithms[library_id] = algorithm
            
            # 3. Rebuild index with new algorithm
            await self._rebuild_library_index(library_id, algorithm, **algorithm_params)
            
            class_logger.info(f"Set library {library_id} to use {algorithm.value} algorithm")
            return True
            
        except Exception as e:
            class_logger.error(f"Failed to set algorithm for library {library_id}: {str(e)}")
            return False
    
    @logger
    @timer
    async def _rebuild_library_index(self, 
                                   library_id: str, 
                                   algorithm: Optional[IndexAlgorithm] = None,
                                   **algorithm_params) -> None:
        """
        Rebuild the vector index for a library.
        
        Args:
            library_id: Library to rebuild index for
            algorithm: Algorithm to use (uses existing or default if not provided)
            **algorithm_params: Algorithm-specific parameters
        """
        try:
            # 1. Get chunk IDs for this library
            chunk_ids = vector_storage.filter_by_library(library_id)
            if not chunk_ids:
                # No chunks - remove index if exists
                if library_id in self._library_indexes:
                    del self._library_indexes[library_id]
                class_logger.info(f"No chunks in library {library_id}, removed index")
                return
            
            # 2. Determine algorithm to use
            if algorithm is None:
                algorithm = self._library_algorithms.get(library_id, self.default_algorithm)
            
            # 3. Create new index
            index = VectorIndexConfig.create_index(algorithm, **algorithm_params)
            
            # 4. Build index with chunk IDs
            index.build_index(chunk_ids)
            
            # 5. Store new index
            self._library_indexes[library_id] = index
            self._library_algorithms[library_id] = algorithm
            
            class_logger.info(f"Rebuilt {algorithm.value} index for library {library_id} with {len(chunk_ids)} chunks")
            
        except Exception as e:
            class_logger.error(f"Failed to rebuild index for library {library_id}: {str(e)}")
            raise
    
    @logger
    @timer
    async def get_library_index_info(self, library_id: str) -> Optional[LibraryIndexInfo]:
        """
        Get information about a library's index.
        
        Args:
            library_id: Library to get info for
            
        Returns:
            LibraryIndexInfo: Index information or None if no index exists
        """
        if library_id not in self._library_indexes:
            return None
        
        index = self._library_indexes[library_id]
        algorithm = self._library_algorithms.get(library_id, "unknown")
        
        # Get algorithm-specific stats
        index_stats = index.get_stats()
        
        return LibraryIndexInfo(
            library_id=library_id,
            algorithm=algorithm.value if isinstance(algorithm, IndexAlgorithm) else str(algorithm),
            is_built=index.is_built,
            vector_count=index.vector_count,
            dimension=index.dimension,
            index_stats=index_stats
        )
    
    @logger
    @timer
    async def get_all_library_indexes_info(self) -> Dict[str, LibraryIndexInfo]:
        """Get information about all library indexes."""
        info = {}
        for library_id in self._library_indexes.keys():
            library_info = await self.get_library_index_info(library_id)
            if library_info:
                info[library_id] = library_info
        return info
    
    @logger
    @timer
    async def delete_library_index(self, library_id: str) -> bool:
        """
        Delete a library's index and all its vectors.
        
        Args:
            library_id: Library to delete index for
            
        Returns:
            bool: True if successful
        """
        try:
            # 1. Remove from vector storage (by getting all chunk IDs and removing each)
            chunk_ids = vector_storage.filter_by_library(library_id)
            for chunk_id in chunk_ids:
                vector_storage.remove_vector(chunk_id)
            
            # 2. Remove index
            if library_id in self._library_indexes:
                del self._library_indexes[library_id]
            
            # 3. Remove algorithm tracking
            if library_id in self._library_algorithms:
                del self._library_algorithms[library_id]
            
            class_logger.info(f"Deleted index and vectors for library {library_id}")
            return True
            
        except Exception as e:
            class_logger.error(f"Failed to delete library index {library_id}: {str(e)}")
            return False


# Global instance for application use
vector_service: Optional[VectorService] = None

def get_vector_service() -> VectorService:
    """Get the global vector service instance."""
    global vector_service
    if vector_service is None:
        raise RuntimeError("VectorService not initialized. Call initialize_vector_service() first.")
    return vector_service

def initialize_vector_service(chunk_service: ChunkService, library_service=None) -> VectorService:
    """Initialize the global vector service instance."""
    global vector_service
    vector_service = VectorService(chunk_service, library_service)
    return vector_service 