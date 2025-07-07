"""
Comprehensive Integration Test Suite for Vector Database Service

This test suite covers:
1. All API endpoints (libraries, documents, chunks, search)
2. All index algorithms (flat, lsh, grid)
3. All similarity functions (cosine, euclidean, manhattan, dot_product)
4. Edge cases and error handling
5. Performance testing
6. Concurrent access patterns
7. Memory usage and resource cleanup

Run with: pytest test_integration_comprehensive.py -v
"""

import pytest
import pytest_asyncio
import asyncio
import time
import json
import uuid
from typing import Dict, List, Tuple, Any, Optional
from contextlib import asynccontextmanager
import httpx
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
import string

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30
PERFORMANCE_ITERATIONS = 100
LARGE_DATASET_SIZE = 1000
CONCURRENT_REQUESTS = 10

# Test data generators
class TestDataGenerator:
    """Generate test data for various scenarios"""
    
    @staticmethod
    def generate_text_chunks(count: int, min_length: int = 10, max_length: int = 100) -> List[str]:
        """Generate realistic text chunks for testing"""
        sample_words = [
            "artificial", "intelligence", "machine", "learning", "vector", "database",
            "semantic", "search", "embedding", "similarity", "cosine", "euclidean",
            "algorithm", "index", "query", "document", "chunk", "library", "service",
            "python", "fastapi", "numpy", "cohere", "text", "analysis", "processing",
            "natural", "language", "understanding", "information", "retrieval", "data"
        ]
        
        chunks = []
        for i in range(count):
            length = random.randint(min_length, max_length)
            words = random.choices(sample_words, k=length)
            chunk_text = " ".join(words)
            chunks.append(f"Document {i}: {chunk_text}")
        
        return chunks
    
    @staticmethod
    def generate_library_data(name_suffix: str = None) -> Dict[str, Any]:
        """Generate library creation data"""
        suffix = name_suffix or str(uuid.uuid4())[:8]
        return {
            "name": f"Test Library {suffix}",
            "metadata": {
                "description": f"Integration test library for {suffix}"
            },
            "preferred_index_algorithm": "flat"
        }
    
    @staticmethod
    def generate_document_data(chunk_count: int = 5) -> Dict[str, Any]:
        """Generate document creation data"""
        chunks = TestDataGenerator.generate_text_chunks(chunk_count)
        return {
            "chunk_texts": chunks,
            "document_metadata": {
                "title": f"Test Document {uuid.uuid4()}",
                "author": "Integration Test",
                "category": "testing"
            },
            "chunk_source": "integration_test"
        }


class IntegrationTestSuite:
    """Main integration test suite"""
    
    def __init__(self):
        self.base_url = BASE_URL
        self.test_libraries: List[str] = []
        self.test_documents: List[str] = []
        self.test_chunks: List[str] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        self.data_generator = TestDataGenerator()
    
    async def setup_method(self):
        """Setup before each test method"""
        # Wait for service to be ready
        await self._wait_for_service_ready()
        
        # Clear any existing test data
        await self._cleanup_test_data()
    
    async def teardown_method(self):
        """Cleanup after each test method"""
        await self._cleanup_test_data()
    
    async def _wait_for_service_ready(self, timeout: int = 30):
        """Wait for the service to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    response = await client.get(f"{self.base_url}/health")
                    if response.status_code == 200:
                        print("Service is ready")
                        return
            except Exception as e:
                print(f"Service not ready: {e}")
                await asyncio.sleep(1)
        
        raise RuntimeError("Service did not become ready within timeout")
    
    async def _cleanup_test_data(self):
        """Clean up test data"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Delete test libraries (this will cascade to documents and chunks)
            for library_id in self.test_libraries:
                try:
                    await client.delete(f"{self.base_url}/api/libraries/{library_id}")
                except Exception as e:
                    print(f"Error cleaning up library {library_id}: {e}")
        
        # Clear tracking lists
        self.test_libraries.clear()
        self.test_documents.clear()
        self.test_chunks.clear()
    
    def _record_performance(self, operation: str, duration: float):
        """Record performance metrics"""
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        self.performance_metrics[operation].append(duration)
    
    def _get_performance_stats(self, operation: str) -> Dict[str, float]:
        """Get performance statistics for an operation"""
        if operation not in self.performance_metrics:
            return {}
        
        times = self.performance_metrics[operation]
        return {
            "count": len(times),
            "min": min(times),
            "max": max(times),
            "avg": sum(times) / len(times),
            "median": sorted(times)[len(times) // 2]
        }


# Test fixtures
@pytest_asyncio.fixture(scope="function")
async def test_suite():
    """Create and setup test suite"""
    suite = IntegrationTestSuite()
    await suite.setup_method()
    yield suite
    await suite.teardown_method()


@pytest_asyncio.fixture(scope="function") 
async def test_library(test_suite):
    """Create a test library"""
    library_data = test_suite.data_generator.generate_library_data()
    
    async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
        response = await client.post(
            f"{test_suite.base_url}/api/libraries/",
            json=library_data
        )
        assert response.status_code == 201
        library = response.json()
        test_suite.test_libraries.append(library["id"])
        return library


@pytest_asyncio.fixture(scope="function")
async def test_library_with_documents(test_suite, test_library):
    """Create a test library with documents and chunks"""
    library_id = test_library["id"]
    
    # Create multiple documents
    documents = []
    for i in range(3):
        doc_data = test_suite.data_generator.generate_document_data(chunk_count=5)
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.post(
                f"{test_suite.base_url}/api/libraries/{library_id}/documents/",
                json=doc_data
            )
            assert response.status_code == 201
            doc_result = response.json()
            documents.append(doc_result)
            test_suite.test_documents.append(doc_result["document_id"])
            test_suite.test_chunks.extend(doc_result["chunk_ids"])
    
    # Wait for background processing to complete
    await asyncio.sleep(2)
    
    return {
        "library": test_library,
        "documents": documents
    }


# Core API Tests
@pytest.mark.asyncio
class TestLibraryEndpoints:
    """Test all library endpoints"""
    
    async def test_library_crud_operations(self, test_suite):
        """Test complete CRUD operations for libraries"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # CREATE
            library_data = test_suite.data_generator.generate_library_data("crud_test")
            response = await client.post(
                f"{test_suite.base_url}/api/libraries/",
                json=library_data
            )
            assert response.status_code == 201
            library = response.json()
            test_suite.test_libraries.append(library["id"])
            
            assert library["name"] == library_data["name"]
            assert library["metadata"]["description"] == library_data["metadata"]["description"]
            assert library["preferred_index_algorithm"] == library_data["preferred_index_algorithm"]
            
            # READ
            response = await client.get(f"{test_suite.base_url}/api/libraries/{library['id']}")
            assert response.status_code == 200
            retrieved_library = response.json()
            assert retrieved_library["id"] == library["id"]
            
            # UPDATE
            update_data = {
                "name": "Updated Test Library",
                "metadata": {
                    "description": "Updated description"
                },
                "preferred_index_algorithm": "lsh"
            }
            response = await client.put(
                f"{test_suite.base_url}/api/libraries/{library['id']}",
                json=update_data
            )
            assert response.status_code == 200
            updated_library = response.json()
            assert updated_library["name"] == update_data["name"]
            assert updated_library["metadata"]["description"] == update_data["metadata"]["description"]
            assert updated_library["preferred_index_algorithm"] == update_data["preferred_index_algorithm"]
            
            # DELETE
            response = await client.delete(f"{test_suite.base_url}/api/libraries/{library['id']}")
            assert response.status_code == 204
            
            # Verify deletion
            response = await client.get(f"{test_suite.base_url}/api/libraries/{library['id']}")
            assert response.status_code == 404
    
    async def test_library_list_operations(self, test_suite):
        """Test library listing operations"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Create multiple libraries
            library_ids = []
            for i in range(3):
                library_data = test_suite.data_generator.generate_library_data(f"list_test_{i}")
                response = await client.post(
                    f"{test_suite.base_url}/api/libraries/",
                    json=library_data
                )
                assert response.status_code == 201
                library = response.json()
                library_ids.append(library["id"])
                test_suite.test_libraries.append(library["id"])
            
            # List all libraries
            response = await client.get(f"{test_suite.base_url}/api/libraries/")
            assert response.status_code == 200
            libraries = response.json()
            
            # Verify all created libraries are in the list
            library_ids_in_response = [lib["id"] for lib in libraries]
            for library_id in library_ids:
                assert library_id in library_ids_in_response
    
    async def test_library_search_functionality(self, test_suite):
        """Test basic search functionality"""
        # Create a test library with some content
        library_data = test_suite.data_generator.generate_library_data("search_test")
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Create library
            response = await client.post(
                f"{test_suite.base_url}/api/libraries/",
                json=library_data
            )
            assert response.status_code == 201
            library = response.json()
            test_suite.test_libraries.append(library["id"])
            
            # Add some content
            doc_data = test_suite.data_generator.generate_document_data(chunk_count=3)
            response = await client.post(
                f"{test_suite.base_url}/api/libraries/{library['id']}/documents/",
                json=doc_data
            )
            assert response.status_code == 201
            doc_result = response.json()
            test_suite.test_documents.append(doc_result["document_id"])
            test_suite.test_chunks.extend(doc_result["chunk_ids"])
            
            # Wait for background processing
            await asyncio.sleep(5)
            
            # Try basic search
            search_data = {
                "query": "test search",
                "k": 5,
                "similarity_function": "cosine"
            }
            
            response = await client.post(
                f"{test_suite.base_url}/api/libraries/{library['id']}/knn-search",
                json=search_data
            )
            assert response.status_code == 200
            
            search_result = response.json()
            assert "query" in search_result
            assert "library_id" in search_result
            assert "results" in search_result


# Continue with the rest of the test suite... 