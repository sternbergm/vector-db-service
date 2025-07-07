"""
Custom vector indexing algorithms for similarity search.
This module contains various indexing strategies implemented from scratch.
No external vector database libraries used - built with NumPy only.

Optimized for Cohere embed-english-light-v3.0 (384 dimensions, cosine similarity).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from abc import ABC, abstractmethod
import logging
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


class VectorIndex(ABC):
    """
    Abstract base class for vector indexing algorithms.
    
    All concrete index implementations must provide:
    1. build_index() - Create search structure from vectors
    2. search() - Find k most similar vectors to query
    3. get_stats() - Return algorithm-specific statistics
    4. Complexity analysis documentation
    """
    
    def __init__(self):
        self.is_built = False
        self.vector_count = 0
        self.dimension = None
    
    @abstractmethod
    def build_index(self, chunk_ids: List[str]) -> None:
        """
        Build index from chunk IDs stored in VectorStorage.
        
        Args:
            chunk_ids: List of chunk IDs to include in this index
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 10, similarity_function: str = "cosine") -> List[Tuple[str, float]]:
        """
        Search for k most similar vectors.
        
        Args:
            query_vector: Query vector to search with
            k: Number of top results to return
            similarity_function: Similarity function to use for comparison
            
        Returns:
            List[Tuple[str, float]]: Top-k results as [(chunk_id, similarity), ...]
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict:
        """
        Get algorithm-specific statistics and metadata.
        
        Returns:
            Dict: Statistics dictionary with at minimum:
                - index_type: str (algorithm name)
                - is_built: bool 
                - vector_count: int
                - dimension: Optional[int]
        """
        pass
    
    def _validate_built(self) -> None:
        """Ensure index has been built before searching."""
        if not self.is_built:
            raise RuntimeError("Index must be built before searching. Call build_index() first.")
    
    def _validate_query_vector(self, query_vector: np.ndarray) -> None:
        """Validate query vector dimensions."""
        if self.dimension is not None and query_vector.shape[0] != self.dimension:
            raise ValueError(f"Query vector dimension {query_vector.shape[0]} doesn't match index dimension {self.dimension}")


class FlatIndex(VectorIndex):
    """
    Flat/Linear Index - Simple brute force search through all vectors.
    
    ALGORITHM:
    - Stores all vectors in memory
    - Compares query against every vector using vectorized operations
    - Returns top-k results sorted by similarity
    
    COMPLEXITY ANALYSIS:
    - Build Time: O(1) - just stores vectors
    - Build Space: O(n * d) where n=vectors, d=dimensions
    - Search Time: O(n * d) - must compare against all vectors
    - Search Space: O(k) - stores top-k results
    
    WHEN TO USE:
    - Small datasets (<10K vectors)
    - When 100% accuracy is required
    - When memory is not a constraint
    - As baseline for comparison with other indexes
    
    PROS:
    - Guaranteed exact results
    - Simple implementation
    - Fast for small datasets
    - Works with any similarity metric
    
    CONS:
    - Slow for large datasets
    - Doesn't scale beyond ~50K vectors
    - No sub-linear search time
    """
    
    def __init__(self, similarity_metric: str = "cosine"):
        super().__init__()
        self.similarity_metric = similarity_metric
        self.chunk_ids: List[str] = []
        
        # Import storage and similarity calculator
        from .storage import vector_storage
        from .similarity import similarity_calculator
        self.storage = vector_storage
        self.similarity_calc = similarity_calculator
        
        logger.info(f"FlatIndex initialized with {similarity_metric} similarity")
    
    def build_index(self, chunk_ids: List[str]) -> None:
        """
        Build flat index from chunk IDs stored in VectorStorage.
        
        Args:
            chunk_ids: List of chunk IDs to include in this index
        """
        if not chunk_ids:
            raise ValueError("Cannot build index from empty chunk_ids list")
        
        # Verify all chunk IDs exist in storage
        missing_chunks = []
        for chunk_id in chunk_ids:
            if not self.storage.exists(chunk_id):
                missing_chunks.append(chunk_id)
        
        if missing_chunks:
            raise ValueError(f"Chunk IDs not found in storage: {missing_chunks}")
        
        # Store chunk IDs for this index
        self.chunk_ids = chunk_ids.copy()
        
        # Store metadata
        self.vector_count = len(chunk_ids)
        first_vector = self.storage.get_vector(chunk_ids[0])
        self.dimension = first_vector.shape[0] if first_vector is not None else None
        self.is_built = True
        
        logger.info(f"FlatIndex built: {self.vector_count} vectors, {self.dimension} dimensions")
    
    def search(self, query_vector: np.ndarray, k: int = 10, similarity_function: str = "cosine") -> List[Tuple[str, float]]:
        """
        Search for k most similar vectors using vectorized similarity computation.
        
        Args:
            query_vector: Query vector to search with
            k: Number of top results to return
            similarity_function: Similarity function to use for comparison
            
        Returns:
            List[Tuple[str, float]]: Top-k results sorted by similarity (highest first)
        """
        self._validate_built()
        self._validate_query_vector(query_vector)
        
        if k <= 0:
            return []
        
        # Get vectors from storage for the chunks in this index
        vectors_dict = {}
        for chunk_id in self.chunk_ids:
            vector = self.storage.get_vector(chunk_id)
            if vector is not None:
                vectors_dict[chunk_id] = vector
        
        # Use the similarity calculator's smart search strategy
        # Use the provided similarity function instead of the instance default
        results = self.similarity_calc.choose_search_strategy(
            query_vector, vectors_dict, k, similarity_function
        )
        
        logger.debug(f"FlatIndex search: found {len(results)} results for k={k} using {similarity_function}")
        return results
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        # Estimate memory usage based on stored vectors
        memory_mb = 0
        if self.is_built and self.dimension is not None:
            # Estimate: vector count * dimension * 4 bytes per float32
            memory_mb = (self.vector_count * self.dimension * 4) / (1024 * 1024)
        
        return {
            "index_type": "FlatIndex",
            "similarity_metric": self.similarity_metric,
            "vector_count": self.vector_count,
            "dimension": self.dimension,
            "memory_mb": memory_mb,
            "is_built": self.is_built
        }


class LSHIndex(VectorIndex):
    """
    LSH (Locality-Sensitive Hashing) Index - Hash-based approximate search.
    
    ALGORITHM:
    - Generates random hyperplanes as hash functions
    - Each vector gets hashed into buckets based on which side of hyperplanes it falls
    - Similar vectors (high cosine similarity) likely hash to same buckets
    - Search only checks vectors in same bucket(s) as query
    
    MATHEMATICAL FOUNDATION:
    - Uses random hyperplanes through origin
    - Hash bit = sign(vector · hyperplane_normal)
    - Probability that two vectors hash to same bit = 1 - (angle_between_vectors / π)
    - Approximates cosine similarity through hash collision probability
    
    COMPLEXITY ANALYSIS:
    - Build Time: O(n * h * d) where h=num_hashes
    - Build Space: O(n + 2^h) for vectors + hash tables
    - Search Time: O(h * d + b * d) where b=bucket_size (typically << n)
    - Search Space: O(k)
    
    WHEN TO USE:
    - Large datasets (>50K vectors)
    - When approximate results are acceptable (~90-95% accuracy)
    - Cosine similarity search
    - Memory constrained environments
    
    PROS:
    - Sub-linear search time O(1) to O(log n)
    - Scales to millions of vectors
    - Low memory overhead
    - Good for high-dimensional data
    
    CONS:
    - Approximate results (may miss some similar vectors)
    - Only works well with cosine similarity
    - Requires tuning of hash parameters
    - Performance depends on data distribution
    """
    
    def __init__(self, num_hashes: int = 10, bucket_width: float = 1.0, seed: int = 42):
        super().__init__()
        self.num_hashes = num_hashes
        self.bucket_width = bucket_width
        self.seed = seed
        self.hash_functions: List[np.ndarray] = []
        self.hash_tables: List[Dict[int, List[str]]] = []
        self.chunk_ids: List[str] = []
        
        # Import storage and similarity calculator
        from .storage import vector_storage
        from .similarity import similarity_calculator
        self.storage = vector_storage
        self.similarity_calc = similarity_calculator
        
        logger.info(f"LSHIndex initialized: {num_hashes} hashes, bucket_width={bucket_width}")
    
    def _generate_hash_functions(self, dimension: int) -> None:
        """Generate random hyperplanes as hash functions."""
        np.random.seed(self.seed)
        self.hash_functions = []
        
        for i in range(self.num_hashes):
            # Random hyperplane normal vector
            hyperplane = np.random.normal(0, 1, dimension)
            # Normalize to unit vector
            hyperplane = hyperplane / np.linalg.norm(hyperplane)
            self.hash_functions.append(hyperplane)
        
        logger.debug(f"Generated {len(self.hash_functions)} hash functions for {dimension}D vectors")
    
    def _hash_vector(self, vector: np.ndarray) -> List[int]:
        """
        Hash a vector using all hash functions.
        
        Args:
            vector: Input vector to hash
            
        Returns:
            List[int]: Hash values, one per hash function
        """
        hashes = []
        for hash_func in self.hash_functions:
            # Compute dot product with hyperplane
            dot_product = np.dot(vector, hash_func)
            # Hash based on which side of hyperplane (sign of dot product)
            hash_val = int(dot_product >= 0)
            hashes.append(hash_val)
        return hashes
    
    def _get_bucket_key(self, hash_values: List[int]) -> int:
        """Convert list of hash bits to single bucket key."""
        # Convert binary hash to integer
        bucket_key = 0
        for i, bit in enumerate(hash_values):
            bucket_key += bit * (2 ** i)
        return bucket_key
    
    def build_index(self, chunk_ids: List[str]) -> None:
        """
        Build LSH index by hashing all vectors into buckets.
        
        Args:
            chunk_ids: List of chunk IDs to include in this index
        """
        if not chunk_ids:
            raise ValueError("Cannot build index from empty chunk_ids list")
        
        # Verify all chunk IDs exist in storage
        missing_chunks = []
        for chunk_id in chunk_ids:
            if not self.storage.exists(chunk_id):
                missing_chunks.append(chunk_id)
        
        if missing_chunks:
            raise ValueError(f"Chunk IDs not found in storage: {missing_chunks}")
        
        self.chunk_ids = chunk_ids.copy()
        self.vector_count = len(chunk_ids)
        
        # Get dimension from first vector
        first_vector = self.storage.get_vector(chunk_ids[0])
        if first_vector is None:
            raise ValueError(f"Could not retrieve vector for chunk_id: {chunk_ids[0]}")
        
        self.dimension = first_vector.shape[0]
        
        # Generate hash functions
        self._generate_hash_functions(self.dimension)
        
        # Initialize hash tables
        self.hash_tables = [defaultdict(list) for _ in range(self.num_hashes)]
        
        # Hash all vectors into buckets
        for chunk_id in chunk_ids:
            vector = self.storage.get_vector(chunk_id)
            if vector is not None:
                hash_values = self._hash_vector(vector)
                
                # Add to each hash table
                for table_idx, hash_val in enumerate(hash_values):
                    self.hash_tables[table_idx][hash_val].append(chunk_id)
        
        self.is_built = True
        
        # Log statistics
        total_buckets = sum(len(table) for table in self.hash_tables)
        avg_bucket_size = self.vector_count / max(total_buckets, 1)
        
        logger.info(f"LSHIndex built: {self.vector_count} vectors, {total_buckets} buckets, "
                   f"avg bucket size: {avg_bucket_size:.1f}")
    
    def search(self, query_vector: np.ndarray, k: int = 10, similarity_function: str = "cosine") -> List[Tuple[str, float]]:
        """
        Search for k most similar vectors using LSH bucketing.
        
        Args:
            query_vector: Query vector to search with
            k: Number of top results to return
            similarity_function: Similarity function to use for comparison
            
        Returns:
            List[Tuple[str, float]]: Top-k results sorted by similarity (highest first)
        """
        self._validate_built()
        self._validate_query_vector(query_vector)
        
        if k <= 0:
            return []
        
        # Hash query vector
        query_hashes = self._hash_vector(query_vector)
        
        # Collect candidate vectors from all hash tables
        candidates: Set[str] = set()
        for table_idx, hash_val in enumerate(query_hashes):
            bucket_candidates = self.hash_tables[table_idx].get(hash_val, [])
            candidates.update(bucket_candidates)
        
        # If no candidates found, fall back to searching a few random buckets
        if not candidates:
            logger.warning("No LSH candidates found, falling back to random sampling")
            for table in self.hash_tables[:2]:  # Check first 2 tables
                for bucket in list(table.values())[:3]:  # Check first 3 buckets per table
                    candidates.update(bucket[:10])  # Max 10 candidates per bucket
        
        # Compute exact similarities for candidates only
        if not candidates:
            return []
        
        # Create candidates dictionary for similarity calculator
        candidates_dict = {}
        for chunk_id in candidates:
            vector = self.storage.get_vector(chunk_id)
            if vector is not None:
                candidates_dict[chunk_id] = vector
        
        # Use similarity calculator's smart search strategy for final evaluation
        # This handles the optimal similarity computation method automatically
        results = self.similarity_calc.choose_search_strategy(
            query_vector, candidates_dict, k, similarity_function
        )
        
        logger.debug(f"LSHIndex search: {len(candidates)} candidates, {len(results)} results for k={k} using {similarity_function}")
        return results
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        if not self.is_built:
            return {"index_type": "LSHIndex", "is_built": False}
        
        # Calculate bucket statistics
        bucket_sizes = []
        for table in self.hash_tables:
            bucket_sizes.extend([len(bucket) for bucket in table.values()])
        
        return {
            "index_type": "LSHIndex",
            "num_hashes": self.num_hashes,
            "vector_count": self.vector_count,
            "dimension": self.dimension,
            "total_buckets": len(bucket_sizes),
            "avg_bucket_size": np.mean(bucket_sizes) if bucket_sizes else 0,
            "max_bucket_size": max(bucket_sizes) if bucket_sizes else 0,
            "empty_buckets": sum(1 for size in bucket_sizes if size == 0),
            "is_built": self.is_built
        }


class GridIndex(VectorIndex):
    """
    Grid/Bucket Index - Space partitioning approach using uniform grid.
    
    ALGORITHM:
    - Divides vector space into uniform grid cells
    - Each vector assigned to grid cell based on coordinates
    - Search checks query cell and neighboring cells
    - Works best with Euclidean distance (spatial proximity)
    
    MATHEMATICAL FOUNDATION:
    - Grid cell = floor(vector_coordinate / cell_size)
    - Neighbors = cells within distance threshold
    - Distance preserved within local neighborhoods
    
    COMPLEXITY ANALYSIS:
    - Build Time: O(n * d) - compute grid coordinates for all vectors
    - Build Space: O(n + g^d) where g=grid_size_per_dimension
    - Search Time: O(d + c * d) where c=average_cell_size (typically << n)
    - Search Space: O(k)
    
    WHEN TO USE:
    - Medium datasets (10K-100K vectors)
    - Euclidean or Manhattan distance metrics
    - When data has spatial clustering
    - Balanced between speed and accuracy
    
    PROS:
    - Good balance of speed vs accuracy
    - Works well with Euclidean distance
    - Predictable performance
    - Easy to understand and debug
    
    CONS:
    - Performance degrades in high dimensions (curse of dimensionality)
    - Requires tuning cell_size parameter
    - Not optimal for cosine similarity
    - Empty cells waste memory
    """
    
    def __init__(self, cell_size: float = 0.1, similarity_metric: str = "euclidean"):
        super().__init__()
        self.cell_size = cell_size
        self.similarity_metric = similarity_metric
        self.grid: Dict[Tuple[int, ...], List[str]] = defaultdict(list)
        self.chunk_ids: List[str] = []
        self.min_coords: Optional[np.ndarray] = None
        self.max_coords: Optional[np.ndarray] = None
        
        # Import storage and similarity calculator
        from .storage import vector_storage
        from .similarity import similarity_calculator
        self.storage = vector_storage
        self.similarity_calc = similarity_calculator
        
        logger.info(f"GridIndex initialized: cell_size={cell_size}, metric={similarity_metric}")
    
    def _get_grid_coordinates(self, vector: np.ndarray) -> Tuple[int, ...]:
        """
        Get grid cell coordinates for a vector.
        
        Args:
            vector: Input vector
            
        Returns:
            Tuple[int, ...]: Grid cell coordinates
        """
        # Normalize vector to [0, 1] range first (using known min/max)
        if self.min_coords is not None and self.max_coords is not None:
            # Avoid division by zero
            ranges = self.max_coords - self.min_coords
            ranges = np.where(ranges == 0, 1, ranges)
            normalized = (vector - self.min_coords) / ranges
        else:
            normalized = vector
        
        # Compute grid cell coordinates
        grid_coords = np.floor(normalized / self.cell_size).astype(int)
        return tuple(grid_coords)
    
    def _get_neighbor_cells(self, center_coords: Tuple[int, ...], radius: int = 1) -> List[Tuple[int, ...]]:
        """
        Get neighboring grid cells within given radius.
        
        Args:
            center_coords: Center cell coordinates
            radius: Radius of neighborhood (in cells)
            
        Returns:
            List[Tuple[int, ...]]: List of neighbor cell coordinates
        """
        neighbors = []
        dimension = len(center_coords)
        
        # For high-dimensional spaces, limit the radius to prevent exponential explosion
        # In 8D with radius=1, we get 3^8 = 6561 neighbors, which is too many
        # Use a more conservative approach for high dimensions
        if dimension > 4:
            # Use Manhattan distance-based neighbors instead of full hypercube
            # This gives at most 2*dimension*radius + 1 neighbors
            neighbors = [center_coords]  # Include center
            
            for dim in range(dimension):
                for offset in range(-radius, radius + 1):
                    if offset != 0:
                        neighbor_coords = list(center_coords)
                        neighbor_coords[dim] += offset
                        neighbors.append(tuple(neighbor_coords))
        else:
            # For lower dimensions, use full hypercube neighbors
            def generate_offsets(dim_remaining, current_offset):
                if dim_remaining == 0:
                    neighbor_coords = tuple(center_coords[i] + current_offset[i] 
                                          for i in range(dimension))
                    neighbors.append(neighbor_coords)
                    return
                
                for offset in range(-radius, radius + 1):
                    generate_offsets(dim_remaining - 1, current_offset + [offset])
            
            generate_offsets(dimension, [])
        
        return neighbors
    
    def build_index(self, chunk_ids: List[str]) -> None:
        """
        Build grid index by assigning vectors to grid cells.
        
        Args:
            chunk_ids: List of chunk IDs to include in this index
        """
        if not chunk_ids:
            raise ValueError("Cannot build index from empty chunk_ids list")
        
        # Verify all chunk IDs exist in storage
        missing_chunks = []
        for chunk_id in chunk_ids:
            if not self.storage.exists(chunk_id):
                missing_chunks.append(chunk_id)
        
        if missing_chunks:
            raise ValueError(f"Chunk IDs not found in storage: {missing_chunks}")
        
        self.chunk_ids = chunk_ids.copy()
        self.vector_count = len(chunk_ids)
        
        # Get dimension from first vector
        first_vector = self.storage.get_vector(chunk_ids[0])
        if first_vector is None:
            raise ValueError(f"Could not retrieve vector for chunk_id: {chunk_ids[0]}")
        
        self.dimension = first_vector.shape[0]
        
        # Compute coordinate bounds for normalization
        all_vectors = []
        for chunk_id in chunk_ids:
            vector = self.storage.get_vector(chunk_id)
            if vector is not None:
                all_vectors.append(vector)
        
        if not all_vectors:
            raise ValueError("No valid vectors found for grid index")
        
        all_vectors_array = np.array(all_vectors)
        self.min_coords = np.min(all_vectors_array, axis=0)
        self.max_coords = np.max(all_vectors_array, axis=0)
        
        # Assign each vector to a grid cell
        for chunk_id in chunk_ids:
            vector = self.storage.get_vector(chunk_id)
            if vector is not None:
                grid_coords = self._get_grid_coordinates(vector)
                self.grid[grid_coords].append(chunk_id)
        
        self.is_built = True
        
        # Log statistics
        non_empty_cells = len(self.grid)
        avg_cell_size = self.vector_count / max(non_empty_cells, 1)
        max_cell_size = max(len(cell) for cell in self.grid.values()) if self.grid else 0
        
        logger.info(f"GridIndex built: {self.vector_count} vectors, {non_empty_cells} cells, "
                   f"avg cell size: {avg_cell_size:.1f}, max cell size: {max_cell_size}")
    
    def search(self, query_vector: np.ndarray, k: int = 10, similarity_function: str = "cosine") -> List[Tuple[str, float]]:
        """
        Search for k most similar vectors using grid spatial locality.
        
        Args:
            query_vector: Query vector to search with
            k: Number of top results to return
            similarity_function: Similarity function to use for comparison
            
        Returns:
            List[Tuple[str, float]]: Top-k results sorted by similarity (highest first)
        """
        self._validate_built()
        self._validate_query_vector(query_vector)
        
        if k <= 0:
            return []
        
        # Get query grid coordinates
        query_coords = self._get_grid_coordinates(query_vector)
        
        # Start with query cell, expand radius if needed
        candidates: Set[str] = set()
        radius = 0
        max_radius = 3  # Prevent infinite expansion
        
        while len(candidates) < k * 2 and radius <= max_radius:  # Get extra candidates
            neighbor_cells = self._get_neighbor_cells(query_coords, radius)
            
            for cell_coords in neighbor_cells:
                if cell_coords in self.grid:
                    candidates.update(self.grid[cell_coords])
            
            radius += 1
        
        # If still no candidates, fall back to some random cells
        if not candidates:
            logger.warning("No grid candidates found, falling back to random cells")
            for cell_coords, chunk_ids in list(self.grid.items())[:5]:
                candidates.update(chunk_ids[:10])
        
        # Compute exact similarities for candidates
        if not candidates:
            return []
        
        # Create candidates dictionary for similarity calculator
        candidates_dict = {}
        for chunk_id in candidates:
            vector = self.storage.get_vector(chunk_id)
            if vector is not None:
                candidates_dict[chunk_id] = vector
        
        # Use similarity calculator's smart search strategy for final evaluation
        # This handles the optimal similarity computation method automatically
        results = self.similarity_calc.choose_search_strategy(
            query_vector, candidates_dict, k, similarity_function
        )
        
        logger.debug(f"GridIndex search: {len(candidates)} candidates, {len(results)} results for k={k} using {similarity_function}")
        return results
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        if not self.is_built:
            return {"index_type": "GridIndex", "is_built": False}
        
        cell_sizes = [len(cell) for cell in self.grid.values()]
        
        return {
            "index_type": "GridIndex",
            "cell_size": self.cell_size,
            "similarity_metric": self.similarity_metric,
            "vector_count": self.vector_count,
            "dimension": self.dimension,
            "non_empty_cells": len(self.grid),
            "avg_cell_size": np.mean(cell_sizes) if cell_sizes else 0,
            "max_cell_size": max(cell_sizes) if cell_sizes else 0,
            "is_built": self.is_built
        } 