"""
Similarity and distance calculation functions for vector search.
Implements various distance metrics using NumPy for efficient computation.
Optimized for Cohere embed-english-light-v3.0 (384 dimensions, cosine-optimized).
"""

import numpy as np
import heapq
from typing import Union, Iterator, Tuple, Dict, List


class SimilarityCalculator:
    """
    Calculator for various similarity metrics.
    Optimized for Cohere embed-english-light-v3.0 characteristics:
    - 384 dimensions
    - Cosine similarity optimized
    - Speed-focused design
    """
    
    def __init__(self):
        self.expected_dimension = 384  # For Cohere embed-english-light-v3.0
    
    def cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Primary metric for Cohere embeddings - measures semantic similarity
        regardless of vector magnitude.
        
        Args:
            vector1: First embedding vector
            vector2: Second embedding vector
            
        Returns:
            float: Cosine similarity score between -1 and 1 (1 = identical)
            
        Raises:
            ValueError: If vectors have different dimensions
        """
        if vector1.shape != vector2.shape:
            raise ValueError(f"Vector dimensions don't match: {vector1.shape} vs {vector2.shape}")
        
        # Handle zero vectors to avoid division by zero
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vector1, vector2) / (norm1 * norm2))
    
    def dot_product_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate dot product similarity between two vectors.
        
        Faster than cosine if vectors are already normalized.
        For Cohere embeddings, this often gives similar results to cosine.
        
        Args:
            vector1: First embedding vector
            vector2: Second embedding vector
            
        Returns:
            float: Dot product similarity score
        """
        if vector1.shape != vector2.shape:
            raise ValueError(f"Vector dimensions don't match: {vector1.shape} vs {vector2.shape}")
        
        return float(np.dot(vector1, vector2))
    
    def euclidean_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two vectors.
        
        Less commonly used for text embeddings but supported by Cohere.
        Lower distance = more similar.
        
        Args:
            vector1: First embedding vector
            vector2: Second embedding vector
            
        Returns:
            float: Euclidean distance (0 = identical, higher = more different)
        """
        if vector1.shape != vector2.shape:
            raise ValueError(f"Vector dimensions don't match: {vector1.shape} vs {vector2.shape}")
        
        return float(np.linalg.norm(vector1 - vector2))
    
    def euclidean_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Convert Euclidean distance to similarity score.
        
        Transforms distance (lower = better) to similarity (higher = better)
        using: similarity = 1 / (1 + distance)
        
        Args:
            vector1: First embedding vector
            vector2: Second embedding vector
            
        Returns:
            float: Similarity score between 0 and 1 (1 = identical)
        """
        distance = self.euclidean_distance(vector1, vector2)
        return 1.0 / (1.0 + distance)
    
    def manhattan_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate Manhattan (L1) distance between two vectors.
        
        Alternative distance metric, less common for embeddings.
        
        Args:
            vector1: First embedding vector
            vector2: Second embedding vector
            
        Returns:
            float: Manhattan distance
        """
        if vector1.shape != vector2.shape:
            raise ValueError(f"Vector dimensions don't match: {vector1.shape} vs {vector2.shape}")
        
        return float(np.sum(np.abs(vector1 - vector2)))
    
    def manhattan_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Convert Manhattan distance to similarity score.
        
        Transforms distance (lower = better) to similarity (higher = better)
        using: similarity = 1 / (1 + distance)
        
        Args:
            vector1: First embedding vector
            vector2: Second embedding vector
            
        Returns:
            float: Similarity score between 0 and 1 (1 = identical)
        """
        distance = self.manhattan_distance(vector1, vector2)
        return 1.0 / (1.0 + distance)
    
    def batch_cosine_similarity(self, query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between a query vector and multiple vectors efficiently.
        
        Optimized for searching through many vectors at once.
        USE THIS for: Small to medium datasets (<50K vectors), when you have enough memory
        
        Args:
            query_vector: Single query vector (384,)
            vectors: Array of vectors to compare against (n, 384)
            
        Returns:
            np.ndarray: Array of similarity scores (n,)
        """
        if len(vectors.shape) != 2:
            raise ValueError("Vectors must be 2D array (n_vectors, dimensions)")
        
        if query_vector.shape[0] != vectors.shape[1]:
            raise ValueError(f"Dimension mismatch: query {query_vector.shape[0]} vs vectors {vectors.shape[1]}")
        
        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return np.zeros(vectors.shape[0])
        
        normalized_query = query_vector / query_norm
        
        # Normalize all vectors
        vector_norms = np.linalg.norm(vectors, axis=1)
        # Avoid division by zero
        non_zero_mask = vector_norms != 0
        
        similarities = np.zeros(vectors.shape[0])
        if np.any(non_zero_mask):
            normalized_vectors = vectors[non_zero_mask] / vector_norms[non_zero_mask, np.newaxis]
            similarities[non_zero_mask] = np.dot(normalized_vectors, normalized_query)
        
        return similarities
    
    def batch_euclidean_distance(self, query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Calculate Euclidean distance between a query vector and multiple vectors efficiently.
        
        Vectorized implementation of: distance = sqrt(sum((a - b)^2))
        
        MATHEMATICAL EQUIVALENCE:
        Individual: np.linalg.norm(query_vector - vector_i) for each vector_i
        Batch:      np.linalg.norm(vectors - query_vector, axis=1) for all vectors at once
        
        Args:
            query_vector: Single query vector (384,)
            vectors: Array of vectors to compare against (n, 384)
            
        Returns:
            np.ndarray: Array of distance scores (n,) - lower is more similar
        """
        if len(vectors.shape) != 2:
            raise ValueError("Vectors must be 2D array (n_vectors, dimensions)")
        
        if query_vector.shape[0] != vectors.shape[1]:
            raise ValueError(f"Dimension mismatch: query {query_vector.shape[0]} vs vectors {vectors.shape[1]}")
        
        # Vectorized: compute ALL differences at once, then ALL norms at once
        # Shape: (n_vectors, dimensions) - (dimensions,) = (n_vectors, dimensions)
        differences = vectors - query_vector
        
        # Compute Euclidean norm for each row (axis=1)
        distances = np.linalg.norm(differences, axis=1)
        
        return distances
    
    def batch_euclidean_similarity(self, query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Calculate Euclidean similarity (converted from distance) for multiple vectors.
        
        Converts distance to similarity using: similarity = 1 / (1 + distance)
        Higher values = more similar (opposite of distance)
        
        Args:
            query_vector: Single query vector (384,)
            vectors: Array of vectors to compare against (n, 384)
            
        Returns:
            np.ndarray: Array of similarity scores (n,) - higher is more similar
        """
        distances = self.batch_euclidean_distance(query_vector, vectors)
        # Convert distances to similarities: closer distance = higher similarity
        similarities = 1.0 / (1.0 + distances)
        return similarities
    
    def batch_manhattan_distance(self, query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Calculate Manhattan (L1) distance between a query vector and multiple vectors efficiently.
        
        Vectorized implementation of: distance = sum(|a - b|)
        
        MATHEMATICAL EQUIVALENCE:
        Individual: np.sum(np.abs(query_vector - vector_i)) for each vector_i  
        Batch:      np.sum(np.abs(vectors - query_vector), axis=1) for all vectors at once
        
        Args:
            query_vector: Single query vector (384,)
            vectors: Array of vectors to compare against (n, 384)
            
        Returns:
            np.ndarray: Array of distance scores (n,) - lower is more similar
        """
        if len(vectors.shape) != 2:
            raise ValueError("Vectors must be 2D array (n_vectors, dimensions)")
        
        if query_vector.shape[0] != vectors.shape[1]:
            raise ValueError(f"Dimension mismatch: query {query_vector.shape[0]} vs vectors {vectors.shape[1]}")
        
        # Vectorized: compute ALL differences at once, then ALL absolute values and sums
        # Shape: (n_vectors, dimensions) - (dimensions,) = (n_vectors, dimensions)
        differences = vectors - query_vector
        
        # Take absolute values and sum along each row (axis=1)
        distances = np.sum(np.abs(differences), axis=1)
        
        return distances
    
    def batch_manhattan_similarity(self, query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Calculate Manhattan similarity (converted from distance) for multiple vectors.
        
        Converts distance to similarity using: similarity = 1 / (1 + distance)
        
        Args:
            query_vector: Single query vector (384,)
            vectors: Array of vectors to compare against (n, 384)
            
        Returns:
            np.ndarray: Array of similarity scores (n,) - higher is more similar
        """
        distances = self.batch_manhattan_distance(query_vector, vectors)
        similarities = 1.0 / (1.0 + distances)
        return similarities
    
    def batch_dot_product_similarity(self, query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Calculate dot product similarity between a query vector and multiple vectors efficiently.
        
        Simplest vectorized operation: just matrix multiplication!
        
        Args:
            query_vector: Single query vector (384,)
            vectors: Array of vectors to compare against (n, 384)
            
        Returns:
            np.ndarray: Array of similarity scores (n,)
        """
        if len(vectors.shape) != 2:
            raise ValueError("Vectors must be 2D array (n_vectors, dimensions)")
        
        if query_vector.shape[0] != vectors.shape[1]:
            raise ValueError(f"Dimension mismatch: query {query_vector.shape[0]} vs vectors {vectors.shape[1]}")
        
        # Simple matrix multiplication: each row of vectors dot with query_vector
        similarities = np.dot(vectors, query_vector)
        
        return similarities
    
    def similarity_generator(self, 
                           query_vector: np.ndarray, 
                           vectors_dict: Dict[str, np.ndarray],
                           similarity_func: str = "cosine") -> Iterator[Tuple[str, float]]:
        """
        Generator that yields (chunk_id, similarity) one at a time.
        
        Memory-efficient: Only processes one vector at a time.
        USE THIS for: Large datasets (>100K vectors), memory-constrained environments
        
        Args:
            query_vector: Query vector to compare against
            vectors_dict: Dictionary of chunk_id -> vector
            similarity_func: Which similarity function to use
            
        Yields:
            Tuple[str, float]: (chunk_id, similarity_score)
        """
        similarity_methods = {
            "cosine": self.cosine_similarity,
            "dot_product": self.dot_product_similarity,
            "euclidean": self.euclidean_similarity,
            "manhattan": self.manhattan_similarity
        }
        
        if similarity_func not in similarity_methods:
            raise ValueError(f"Unknown similarity function: {similarity_func}")
        
        sim_func = similarity_methods[similarity_func]
        
        for chunk_id, vector in vectors_dict.items():
            try:
                similarity = sim_func(query_vector, vector)
                yield (chunk_id, similarity)
            except Exception as e:
                # Log error but continue processing other vectors
                continue
    
    def heap_based_top_k(self, 
                        query_vector: np.ndarray, 
                        vectors_dict: Dict[str, np.ndarray], 
                        k: int = 5,
                        similarity_func: str = "cosine") -> List[Tuple[str, float]]:
        """
        Memory-efficient top-k search using min-heap.
        
        Only keeps k items in memory at any time, regardless of dataset size.
        
        Time Complexity: O(n log k) where n = number of vectors
        Space Complexity: O(k) - only stores k items plus one vector at a time
        
        PERFORMANCE COMPARISON:
        - Memory usage: ~250 bytes (vs ~400KB for batch approach with 100K vectors)
        - Speed: ~2-3x slower than batch approach due to Python loops
        - Scales: Works with millions of vectors without memory issues
        
        USE THIS WHEN:
        - Large datasets (>100K vectors)
        - Memory-constrained environments  
        - k is small (k << dataset size)
        
        Args:
            query_vector: Query vector to search with
            vectors_dict: Dictionary mapping chunk_id -> vector
            k: Number of top results to return
            similarity_func: Similarity metric to use
            
        Returns:
            List[Tuple[str, float]]: Top-k results as [(chunk_id, similarity), ...]
                                   Sorted by similarity (highest first)
        """
        if k <= 0:
            return []
        
        min_heap = []  # Min-heap: smallest similarity at root
        
        # Process vectors one at a time using generator
        for chunk_id, similarity in self.similarity_generator(query_vector, vectors_dict, similarity_func):
            if len(min_heap) < k:
                # Heap not full yet - add new item
                heapq.heappush(min_heap, (similarity, chunk_id))
            elif similarity > min_heap[0][0]:  # Better than worst item in heap
                # Replace worst item with new better item
                heapq.heapreplace(min_heap, (similarity, chunk_id))
        
        # Convert min-heap to sorted list (highest similarity first)
        return [(chunk_id, similarity) for similarity, chunk_id in sorted(min_heap, reverse=True)]
    
    def choose_search_strategy(self, 
                             query_vector: np.ndarray,
                             vectors_dict: Dict[str, np.ndarray], 
                             k: int = 5,
                             similarity_func: str = "cosine") -> List[Tuple[str, float]]:
        """
        Automatically choose the best search strategy based on dataset size.
        
        Decision logic:
        - Small dataset (<10K): Use batch processing (fastest)
        - Medium dataset (10K-50K): Use batch if memory allows, else heap
        - Large dataset (>50K): Use heap-based approach (memory-efficient)
        
        Args:
            query_vector: Query vector to search with
            vectors_dict: Dictionary mapping chunk_id -> vector  
            k: Number of top results to return
            similarity_func: Similarity metric to use
            
        Returns:
            List[Tuple[str, float]]: Top-k results
        """
        dataset_size = len(vectors_dict)
        
        # Estimate memory requirements for batch approach
        # Each float32 similarity score = 4 bytes
        estimated_memory_mb = (dataset_size * 4) / (1024 * 1024)
        
        if dataset_size < 10_000:
            # Small dataset - use fast batch processing
            return self._batch_search(query_vector, vectors_dict, k, similarity_func)
        elif dataset_size < 50_000 and estimated_memory_mb < 100:  # Less than 100MB
            # Medium dataset with acceptable memory usage
            return self._batch_search(query_vector, vectors_dict, k, similarity_func)
        else:
            # Large dataset or memory constraints - use heap approach
            return self.heap_based_top_k(query_vector, vectors_dict, k, similarity_func)
    
    def _batch_search(self, 
                     query_vector: np.ndarray,
                     vectors_dict: Dict[str, np.ndarray], 
                     k: int,
                     similarity_func: str) -> List[Tuple[str, float]]:
        """Helper method for batch-based search."""
        chunk_ids = list(vectors_dict.keys())
        vectors_matrix = np.array(list(vectors_dict.values()))
        
        if similarity_func == "cosine":
            similarities = self.batch_cosine_similarity(query_vector, vectors_matrix)
        elif similarity_func == "euclidean":
            similarities = self.batch_euclidean_similarity(query_vector, vectors_matrix)
        elif similarity_func == "manhattan":
            similarities = self.batch_manhattan_similarity(query_vector, vectors_matrix)
        elif similarity_func == "dot_product":
            similarities = self.batch_dot_product_similarity(query_vector, vectors_matrix)
        else:
            # Fall back to generator approach for unsupported similarity functions
            return self.heap_based_top_k(query_vector, vectors_dict, k, similarity_func)
        
        # Get top-k indices
        if len(similarities) <= k:
            # Return all if dataset smaller than k
            sorted_indices = np.argsort(similarities)[::-1]
        else:
            # Use partial sort for efficiency
            top_k_indices = np.argpartition(similarities, -k)[-k:]
            sorted_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
        
        return [(chunk_ids[i], float(similarities[i])) for i in sorted_indices]


# Global instance optimized for Cohere embed-english-light-v3.0
similarity_calculator = SimilarityCalculator() 