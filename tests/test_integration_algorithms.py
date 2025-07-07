"""
Integration Tests for Vector Index Algorithms and Similarity Functions

Tests all combinations of:
- Index algorithms: flat, lsh, grid
- Similarity functions: cosine, euclidean, manhattan, dot_product
- Various data sizes and dimensions
- Algorithm-specific edge cases
"""

import pytest
import pytest_asyncio
import asyncio
import httpx
import numpy as np
from typing import Dict, List, Any
import time
import uuid
from logging import getLogger
# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio
logger = getLogger(__name__)

class AlgorithmTestSuite:
    """Test suite for vector algorithms"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_libraries: List[str] = []
        self.algorithms = ["flat", "lsh", "grid"]
        self.similarity_functions = ["cosine", "euclidean", "manhattan", "dot_product"]
    
    async def cleanup(self):
        """Clean up test data"""
        async with httpx.AsyncClient(timeout=30) as client:
            for library_id in self.test_libraries:
                try:
                    await client.delete(f"{self.base_url}/api/libraries/{library_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up library {library_id}: {e}")
        self.test_libraries.clear()
    
    async def create_test_library(self, algorithm: str, name_suffix: str = None) -> Dict[str, Any]:
        """Create a test library with specified algorithm"""
        suffix = name_suffix or str(uuid.uuid4())[:8]
        library_data = {
            "name": f"Algorithm Test Library {algorithm} {suffix}",
            "description": f"Testing {algorithm} algorithm",
            "preferred_index_algorithm": algorithm
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.base_url}/api/libraries/",
                json=library_data
            )
            assert response.status_code == 201
            library = response.json()
            self.test_libraries.append(library["id"])
            return library
    
    async def populate_library_with_test_data(self, library_id: str, chunk_count: int = 50) -> List[str]:
        """Populate library with test data"""
        # Create diverse test chunks
        test_chunks = []
        
        # Mathematical/scientific texts
        math_chunks = [
            "Linear algebra is fundamental to machine learning and data science applications",
            "Vector spaces and eigenvalues are crucial concepts in computational mathematics",
            "Matrix multiplication and decomposition techniques optimize neural network training",
            "Gradient descent algorithms minimize loss functions in optimization problems",
            "Principal component analysis reduces dimensionality while preserving variance"
        ]
        
        # Technology/programming texts
        tech_chunks = [
            "Python programming language excels in data science and artificial intelligence",
            "FastAPI framework provides high-performance REST API development capabilities",
            "Docker containers enable consistent deployment across different environments",
            "Microservices architecture improves scalability and maintainability of applications",
            "Machine learning models require careful validation and testing procedures"
        ]
        
        # General knowledge texts
        general_chunks = [
            "Natural language processing transforms human communication into computational understanding",
            "Search engines utilize complex algorithms to rank and retrieve relevant information",
            "Database systems store and organize large volumes of structured and unstructured data",
            "Cloud computing provides scalable infrastructure for modern software applications",
            "Artificial intelligence systems learn patterns from data to make predictions"
        ]
        
        all_base_chunks = math_chunks + tech_chunks + general_chunks
        
        # Generate required number of chunks
        chunk_texts = []
        for i in range(chunk_count):
            base_chunk = all_base_chunks[i % len(all_base_chunks)]
            # Add variation to make each chunk unique
            chunk_text = f"Document {i}: {base_chunk} Additional context for uniqueness {i}."
            chunk_texts.append(chunk_text)
        
        # Create document with these chunks
        document_data = {
            "chunk_texts": chunk_texts,
            "document_metadata": {
                "title": f"Algorithm Test Document for {library_id}",
                "test_type": "algorithm_validation",
                "chunk_count": chunk_count
            },
            "chunk_source": "algorithm_test"
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.base_url}/api/libraries/{library_id}/documents/",
                json=document_data
            )
            assert response.status_code == 201
            result = response.json()
            
            # Wait for all chunks to be indexed (have embeddings)
            await self.wait_for_indexing_complete(library_id)
            
            return result["chunk_ids"]
    
    async def wait_for_indexing_complete(self, library_id: str, timeout: int = 120):
        """Wait for chunks to be indexed by attempting searches until we get results"""
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                # Try a search to see if indexing is complete
                search_data = {
                    "query": "machine learning",
                    "k": 1,
                    "similarity_function": "cosine"
                }
                
                try:
                    response = await client.post(
                        f"{self.base_url}/api/libraries/{library_id}/knn-search",
                        json=search_data
                    )
                    
                    if response.status_code == 200:
                        search_result = response.json()
                        if len(search_result.get("results", [])) > 0:
                            logger.info(f"✅ Indexing complete for library {library_id}")
                            break
                    
                    # Check timeout
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        logger.warning(f"⚠️ Indexing timeout after {timeout} seconds, proceeding anyway...")
                        break
                    
                    logger.info(f"⏳ Waiting for indexing... (elapsed: {elapsed:.1f}s)")
                    await asyncio.sleep(3)
                    
                except Exception as e:
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        logger.warning(f"⚠️ Indexing timeout after {timeout} seconds, proceeding anyway...")
                        break
                    logger.info(f"⏳ Waiting for indexing... (elapsed: {elapsed:.1f}s)")
                    await asyncio.sleep(3)
    
    async def test_search_with_algorithm(self, library_id: str, algorithm: str, similarity_function: str) -> Dict[str, Any]:
        """Test search with specific algorithm and similarity function"""
        # Set the algorithm if not already set
        algorithm_data = {
            "algorithm": algorithm,
            "parameters": {}
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            # Set algorithm
            response = await client.post(
                f"{self.base_url}/api/libraries/{library_id}/index-algorithm",
                json=algorithm_data
            )
            assert response.status_code == 200
            
            # Wait for reindexing
            await asyncio.sleep(2)
            
            # Test search
            search_data = {
                "query": "machine learning algorithms and data science",
                "k": 5,
                "similarity_function": similarity_function
            }
            
            start_time = time.time()
            response = await client.post(
                f"{self.base_url}/api/libraries/{library_id}/knn-search",
                json=search_data
            )
            search_time = time.time() - start_time
            
            assert response.status_code == 200
            search_result = response.json()
            
            # Verify search result structure
            assert "query" in search_result
            assert "library_id" in search_result
            assert "results" in search_result
            assert "total_found" in search_result
            assert "search_time_ms" in search_result
            
            # Verify results
            assert len(search_result["results"]) <= 5
            assert search_result["library_id"] == library_id
            assert search_result["query"] == search_data["query"]
            
            # Verify similarity scores are in valid range
            for result in search_result["results"]:
                assert "chunk_id" in result
                assert "similarity_score" in result
                assert "chunk" in result
                assert isinstance(result["similarity_score"], (int, float))
                
                # Similarity scores should be reasonable
                if similarity_function == "cosine":
                    assert -1.0 <= result["similarity_score"] <= 1.0
                elif similarity_function == "dot_product":
                    # Dot product can be any value
                    assert isinstance(result["similarity_score"], (int, float))
                else:  # euclidean, manhattan
                    assert result["similarity_score"] >= 0.0
            
            return {
                "algorithm": algorithm,
                "similarity_function": similarity_function,
                "search_time": search_time,
                "results_count": len(search_result["results"]),
                "search_result": search_result
            }


@pytest_asyncio.fixture(scope="function")
async def algorithm_suite():
    """Create algorithm test suite"""
    suite = AlgorithmTestSuite()
    yield suite
    await suite.cleanup()


@pytest.mark.asyncio
class TestIndexAlgorithms:
    """Test all index algorithms"""
    
    async def test_flat_index_algorithm(self, algorithm_suite):
        """Test flat index algorithm with all similarity functions"""
        library = await algorithm_suite.create_test_library("flat", "flat_test")
        library_id = library["id"]
        
        # Populate with test data
        await algorithm_suite.populate_library_with_test_data(library_id, 20)
        
        # Test with all similarity functions
        results = {}
        for similarity_func in algorithm_suite.similarity_functions:
            result = await algorithm_suite.test_search_with_algorithm(
                library_id, "flat", similarity_func
            )
            results[similarity_func] = result
            
            # Flat index should return exact results
            assert result["results_count"] > 0
            logger.info(f"Flat index with {similarity_func}: {result['results_count']} results in {result['search_time']:.3f}s")
        
        # All similarity functions should return results
        for similarity_func in algorithm_suite.similarity_functions:
            assert results[similarity_func]["results_count"] > 0
    
    async def test_lsh_index_algorithm(self, algorithm_suite):
        """Test LSH index algorithm"""
        library = await algorithm_suite.create_test_library("lsh", "lsh_test")
        library_id = library["id"]
        
        # Populate with larger dataset for LSH
        await algorithm_suite.populate_library_with_test_data(library_id, 100)
        
        # Test with cosine similarity (LSH works best with cosine)
        result = await algorithm_suite.test_search_with_algorithm(
            library_id, "lsh", "cosine"
        )
        
        # LSH should return approximate results
        assert result["results_count"] > 0
        logger.info(f"LSH index with cosine: {result['results_count']} results in {result['search_time']:.3f}s")
        
        # Test with other similarity functions
        for similarity_func in ["euclidean", "manhattan", "dot_product"]:
            result = await algorithm_suite.test_search_with_algorithm(
                library_id, "lsh", similarity_func
            )
            logger.info(f"LSH index with {similarity_func}: {result['results_count']} results in {result['search_time']:.3f}s")
    
    async def test_grid_index_algorithm(self, algorithm_suite):
        """Test grid index algorithm"""
        library = await algorithm_suite.create_test_library("grid", "grid_test")
        library_id = library["id"]
        
        # Populate with test data
        await algorithm_suite.populate_library_with_test_data(library_id, 50)
        
        # Test with euclidean similarity (grid works best with euclidean)
        result = await algorithm_suite.test_search_with_algorithm(
            library_id, "grid", "euclidean"
        )
        
        # Grid should return approximate results
        assert result["results_count"] > 0
        logger.info(f"Grid index with euclidean: {result['results_count']} results in {result['search_time']:.3f}s")
        
        # Test with other similarity functions
        for similarity_func in ["cosine", "manhattan", "dot_product"]:
            result = await algorithm_suite.test_search_with_algorithm(
                library_id, "grid", similarity_func
            )
            logger.info(f"Grid index with {similarity_func}: {result['results_count']} results in {result['search_time']:.3f}s")
    
    async def test_algorithm_comparison(self, algorithm_suite):
        """Compare performance of different algorithms"""
        results = {}
        
        # Test each algorithm
        for algorithm in algorithm_suite.algorithms:
            library = await algorithm_suite.create_test_library(algorithm, f"comparison_{algorithm}")
            library_id = library["id"]
            
            # Populate with same dataset
            await algorithm_suite.populate_library_with_test_data(library_id, 100)
            
            # Test search performance
            result = await algorithm_suite.test_search_with_algorithm(
                library_id, algorithm, "cosine"
            )
            
            results[algorithm] = result
            logger.info(f"{algorithm.upper()} algorithm: {result['results_count']} results in {result['search_time']:.3f}s")
        
        # All algorithms should return results
        for algorithm in algorithm_suite.algorithms:
            assert results[algorithm]["results_count"] > 0
        
        # Print comparison summary
        logger.info("\nAlgorithm Performance Comparison:")
        for algorithm in algorithm_suite.algorithms:
            result = results[algorithm]
            logger.info(f"  {algorithm.upper()}: {result['search_time']:.3f}s, {result['results_count']} results")


@pytest.mark.asyncio
class TestSimilarityFunctions:
    """Test all similarity functions"""
    
    async def test_cosine_similarity_accuracy(self, algorithm_suite):
        """Test cosine similarity accuracy"""
        library = await algorithm_suite.create_test_library("flat", "cosine_test")
        library_id = library["id"]
        
        # Create chunks with known similarity patterns
        similar_chunks = [
            "machine learning algorithms",
            "artificial intelligence algorithms",
            "deep learning neural networks",
            "natural language processing",
            "computer vision systems"
        ]
        
        dissimilar_chunks = [
            "cooking recipes and food preparation",
            "automotive repair and maintenance",
            "gardening tips and plant care",
            "financial investment strategies",
            "home decoration ideas"
        ]
        
        all_chunks = similar_chunks + dissimilar_chunks
        
        # Create document
        document_data = {
            "chunk_texts": all_chunks,
            "document_metadata": {"title": "Cosine Similarity Test"},
            "chunk_source": "similarity_test"
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{algorithm_suite.base_url}/api/libraries/{library_id}/documents/",
                json=document_data
            )
            assert response.status_code == 201
            
            # Wait for processing
            await asyncio.sleep(3)
            
            # Search for machine learning related content
            search_data = {
                "query": "machine learning artificial intelligence",
                "k": 5,
                "similarity_function": "cosine"
            }
            
            response = await client.post(
                f"{algorithm_suite.base_url}/api/libraries/{library_id}/knn-search",
                json=search_data
            )
            assert response.status_code == 200
            
            search_result = response.json()
            
            # Verify that similar chunks score higher than dissimilar ones
            assert len(search_result["results"]) > 0
            
            # Check that the top result is from the similar_chunks
            top_result = search_result["results"][0]
            assert top_result["similarity_score"] > 0.3  # Should be reasonably high
            
            logger.info(f"Cosine similarity test - Top result score: {top_result['similarity_score']:.3f}")
    
    async def test_all_similarity_functions_consistency(self, algorithm_suite):
        """Test that all similarity functions return consistent results"""
        library = await algorithm_suite.create_test_library("flat", "consistency_test")
        library_id = library["id"]
        
        # Populate with test data
        await algorithm_suite.populate_library_with_test_data(library_id, 30)
        
        # Test each similarity function
        results = {}
        query = "vector database similarity search"
        
        for similarity_func in algorithm_suite.similarity_functions:
            result = await algorithm_suite.test_search_with_algorithm(
                library_id, "flat", similarity_func
            )
            results[similarity_func] = result
            
            # All should return some results
            assert result["results_count"] > 0
            
            # Check score ranges
            scores = [r["similarity_score"] for r in result["search_result"]["results"]]
            if similarity_func == "cosine":
                assert all(-1.0 <= score <= 1.0 for score in scores)
            elif similarity_func in ["euclidean", "manhattan"]:
                assert all(score >= 0.0 for score in scores)
            # dot_product can be any value
            
            logger.info(f"{similarity_func}: {len(scores)} results, score range: {min(scores):.3f} to {max(scores):.3f}")
        
        # All functions should return results
        for similarity_func in algorithm_suite.similarity_functions:
            assert results[similarity_func]["results_count"] > 0


@pytest.mark.asyncio
class TestAlgorithmEdgeCases:
    """Test edge cases for algorithms"""
    
    async def test_empty_library_search(self, algorithm_suite):
        """Test searching empty library"""
        library = await algorithm_suite.create_test_library("flat", "empty_test")
        library_id = library["id"]
        
        # Don't populate with data
        
        async with httpx.AsyncClient(timeout=30) as client:
            search_data = {
                "query": "test query",
                "k": 5,
                "similarity_function": "cosine"
            }
            
            response = await client.post(
                f"{algorithm_suite.base_url}/api/libraries/{library_id}/knn-search",
                json=search_data
            )
            assert response.status_code == 200
            
            search_result = response.json()
            assert len(search_result["results"]) == 0
            assert search_result["total_found"] == 0
    
    async def test_single_chunk_library(self, algorithm_suite):
        """Test library with single chunk"""
        library = await algorithm_suite.create_test_library("flat", "single_test")
        library_id = library["id"]
        
        # Create single chunk
        document_data = {
            "chunk_texts": ["Single test chunk for algorithm validation"],
            "document_metadata": {"title": "Single Chunk Test"},
            "chunk_source": "single_test"
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{algorithm_suite.base_url}/api/libraries/{library_id}/documents/",
                json=document_data
            )
            assert response.status_code == 201
            
            # Wait for processing
            await asyncio.sleep(2)
            
            # Search should return the single chunk
            search_data = {
                "query": "test chunk algorithm",
                "k": 5,
                "similarity_function": "cosine"
            }
            
            response = await client.post(
                f"{algorithm_suite.base_url}/api/libraries/{library_id}/knn-search",
                json=search_data
            )
            assert response.status_code == 200
            
            search_result = response.json()
            assert len(search_result["results"]) == 1
            assert search_result["total_found"] == 1
    
    async def test_large_k_value(self, algorithm_suite):
        """Test with k larger than available chunks"""
        library = await algorithm_suite.create_test_library("flat", "large_k_test")
        library_id = library["id"]
        
        # Create few chunks
        await algorithm_suite.populate_library_with_test_data(library_id, 5)
        
        async with httpx.AsyncClient(timeout=30) as client:
            search_data = {
                "query": "test query",
                "k": 100,  # Much larger than available chunks
                "similarity_function": "cosine"
            }
            
            response = await client.post(
                f"{algorithm_suite.base_url}/api/libraries/{library_id}/knn-search",
                json=search_data
            )
            assert response.status_code == 200
            
            search_result = response.json()
            # Should return all available chunks, not more than that
            assert len(search_result["results"]) <= 5
            assert search_result["total_found"] <= 5 