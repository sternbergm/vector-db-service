from typing import Dict, List, Optional, Tuple, Any
import threading
import logging
from datetime import datetime, timezone
import numpy as np
from schemas.chunk_schema import ChunkMetadata

logger = logging.getLogger(__name__)


class VectorStorage:
    """
    Thread-safe in-memory vector storage for embeddings.
    
    Provides CRUD operations for storing and retrieving embedding vectors
    associated with chunk IDs. Uses read-write locks to ensure thread safety
    during concurrent operations.
    """
    
    def __init__(self):
        """Initialize empty vector storage with thread safety."""
        # Main storage: chunk_id -> embedding vector
        self._vectors: Dict[str, np.ndarray] = {}
        
        # Metadata storage: chunk_id -> ChunkMetadata
        self._metadata: Dict[str, ChunkMetadata] = {}
        
        # Library mapping: library_id -> list of chunk_ids (for efficient filtering)
        self._libraries: Dict[str, List[str]] = {}
        
        # Thread safety with reentrant lock
        self._lock = threading.RLock()
        
        # Statistics tracking
        self._stats = {
            "total_vectors": 0,
            "last_updated": None,
            "operations_count": 0
        }
        
        logger.info("VectorStorage initialized")
    
    def add_vector(self, chunk_id: str, embedding: np.ndarray, library_id: str, metadata: Optional[ChunkMetadata] = None) -> bool:
        """
        Add or update a vector embedding for a chunk.
        
        Args:
            chunk_id: Unique identifier for the chunk
            embedding: List of float values representing the embedding
            library_id: The library ID this chunk belongs to
            metadata: Optional ChunkMetadata associated with the vector
            
        Returns:
            bool: True if operation successful, False otherwise
            
        Raises:
            ValueError: If chunk_id is empty or embedding is invalid
        """
        if not chunk_id or not chunk_id.strip():
            raise ValueError("chunk_id cannot be empty")
            
        if not embedding.size:
            raise ValueError("embedding cannot be empty")
            
        if not library_id or not library_id.strip():
            raise ValueError("library_id cannot be empty")
            
        try:
            # Ensure correct dtype
            vector_array = embedding.astype(np.float32)
            
            with self._lock:
                # Check if this is an update or new insertion
                is_update = chunk_id in self._vectors
                
                # Store vector and metadata
                self._vectors[chunk_id] = vector_array
                self._metadata[chunk_id] = metadata or ChunkMetadata()
                
                if library_id in self._libraries.keys():
                    if chunk_id not in self._libraries[library_id]:
                        self._libraries[library_id].append(chunk_id)
                else:
                    self._libraries[library_id] = [chunk_id]
                
                # Update statistics
                if not is_update:
                    self._stats["total_vectors"] += 1
                    
                self._stats["last_updated"] = datetime.now(timezone.utc)
                self._stats["operations_count"] += 1
                
                action = "Updated" if is_update else "Added"
                logger.debug(f"{action} vector for chunk_id: {chunk_id}, dimension: {vector_array.shape[0]}")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to add vector for chunk_id {chunk_id}: {str(e)}")
            return False
    
    def get_vector(self, chunk_id: str) -> Optional[np.ndarray]:
        """
        Retrieve a vector embedding by chunk ID.
        
        Args:
            chunk_id: Unique identifier for the chunk
            
        Returns:
            numpy.ndarray: The embedding vector, or None if not found
        """
        if not chunk_id or not chunk_id.strip():
            return None
            
        with self._lock:
            vector = self._vectors.get(chunk_id)
            if vector is not None:
                # Return a copy to prevent external modification
                return vector.copy()
            return None
    
    def get_metadata(self, chunk_id: str) -> Optional[ChunkMetadata]:
        """
        Retrieve metadata for a chunk.
        
        Args:
            chunk_id: Unique identifier for the chunk
            
        Returns:
            ChunkMetadata: Metadata object, or None if not found
        """
        if not chunk_id or not chunk_id.strip():
            return None
            
        with self._lock:
            metadata = self._metadata.get(chunk_id)
            if metadata is not None:
                # Return a copy to prevent external modification
                return metadata.model_copy()
            return None
    
    def remove_vector(self, chunk_id: str) -> bool:
        """
        Remove a vector embedding by chunk ID.
        
        Args:
            chunk_id: Unique identifier for the chunk
            
        Returns:
            bool: True if vector was found and removed, False otherwise
        """
        if not chunk_id or not chunk_id.strip():
            return False
            
        with self._lock:
            if chunk_id in self._vectors:
                # Remove from library mapping
                # Find which library this chunk belongs to
                library_id = None
                for lib_id, chunk_ids in self._libraries.items():
                    if chunk_id in chunk_ids:
                        library_id = lib_id
                        break
                
                if library_id:
                    try:
                        self._libraries[library_id].remove(chunk_id)
                        # Remove library entry if empty
                        if not self._libraries[library_id]:
                            del self._libraries[library_id]
                    except ValueError:
                        pass  # chunk_id not in list
                
                # Remove from main storage
                del self._vectors[chunk_id]
                self._metadata.pop(chunk_id, None)  # Remove metadata too
                
                self._stats["total_vectors"] -= 1
                self._stats["last_updated"] = datetime.now(timezone.utc)
                self._stats["operations_count"] += 1
                
                logger.debug(f"Removed vector for chunk_id: {chunk_id}")
                return True
            return False
    
    def exists(self, chunk_id: str) -> bool:
        """
        Check if a vector exists for the given chunk ID.
        
        Args:
            chunk_id: Unique identifier for the chunk
            
        Returns:
            bool: True if vector exists, False otherwise
        """
        if not chunk_id or not chunk_id.strip():
            return False
            
        with self._lock:
            return chunk_id in self._vectors
    
    def list_chunk_ids(self) -> List[str]:
        """
        Get a list of all chunk IDs that have vectors stored.
        
        Returns:
            List[str]: List of chunk IDs
        """
        with self._lock:
            return list(self._vectors.keys())
    
    def get_all_vectors(self) -> Dict[str, np.ndarray]:
        """
        Get all vectors in storage. Use with caution for large datasets.
        
        Returns:
            Dict[str, numpy.ndarray]: Dictionary mapping chunk_id to vector
        """
        with self._lock:
            # Return copies to prevent external modification
            return {chunk_id: vector.copy() for chunk_id, vector in self._vectors.items()}
    
    def filter_by_library(self, library_id: str) -> List[str]:
        """
        Get chunk IDs that belong to a specific library.
        
        Args:
            library_id: The library ID to filter by
            
        Returns:
            List[str]: List of chunk IDs belonging to the library
        """
        with self._lock:
            return self._libraries.get(library_id, []).copy()
    
    def get_library_ids(self) -> List[str]:
        """
        Get a list of all library IDs that have vectors stored.
        
        Returns:
            List[str]: List of library IDs
        """
        with self._lock:
            return list(self._libraries.keys())
    
    def clear(self) -> None:
        """
        Clear all vectors and metadata from storage.
        Warning: This operation cannot be undone.
        """
        with self._lock:
            cleared_count = len(self._vectors)
            self._vectors.clear()
            self._metadata.clear()
            self._libraries.clear()
            
            self._stats["total_vectors"] = 0
            self._stats["last_updated"] = datetime.now(timezone.utc)
            self._stats["operations_count"] += 1
            
            logger.info(f"Cleared {cleared_count} vectors from storage")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            dict: Statistics including total vectors, last updated time, etc.
        """
        with self._lock:
            stats = self._stats.copy()
            
            # Add dimension information if vectors exist
            if self._vectors:
                sample_vector = next(iter(self._vectors.values()))
                stats["vector_dimension"] = sample_vector.shape[0]
            else:
                stats["vector_dimension"] = None
            
            # Add library statistics
            stats["total_libraries"] = len(self._libraries)
            
            return stats
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Estimate memory usage of the storage.
        
        Returns:
            dict: Memory usage statistics
        """
        with self._lock:
            total_vectors = len(self._vectors)
            if total_vectors == 0:
                return {
                    "total_vectors": 0,
                    "estimated_memory_mb": 0,
                    "avg_vector_size_bytes": 0
                }
            
            # Calculate memory usage
            sample_vector = next(iter(self._vectors.values()))
            vector_size_bytes = sample_vector.nbytes
            total_vector_memory = sum(vector.nbytes for vector in self._vectors.values())
            
            # Estimate metadata memory (rough approximation)
            metadata_memory = sum(len(str(metadata)) for metadata in self._metadata.values()) * 4  # rough estimate
            
            total_memory_bytes = total_vector_memory + metadata_memory
            
            return {
                "total_vectors": total_vectors,
                "estimated_memory_mb": total_memory_bytes / (1024 * 1024),
                "avg_vector_size_bytes": vector_size_bytes,
                "vector_dimension": sample_vector.shape[0]
            }


# Global instance for application use
vector_storage = VectorStorage() 