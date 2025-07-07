"""
Performance and Edge Case Tests for Vector Database Service

Tests:
1. Performance under load
2. Concurrent access patterns
3. Memory usage monitoring
4. Large dataset handling
5. Error conditions and recovery
6. Resource cleanup
7. Background task monitoring
"""

import pytest
import pytest_asyncio
import asyncio
import httpx
import time
import threading
import uuid
import psutil
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import random

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio


class PerformanceTestSuite:
    """Performance and stress testing suite"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_libraries: List[str] = []
        self.performance_results: Dict[str, Any] = {}
        self.process = psutil.Process(os.getpid())
    
    async def cleanup(self):
        """Clean up test resources"""
        async with httpx.AsyncClient(timeout=30) as client:
            for library_id in self.test_libraries:
                try:
                    await client.delete(f"{self.base_url}/api/libraries/{library_id}")
                except Exception as e:
                    print(f"Error cleaning up library {library_id}: {e}")
        self.test_libraries.clear()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory_info = self.process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": self.process.memory_percent()
        }
    
    def check_memory_safety(self, max_usage_mb: float = 8000) -> bool:
        """Check if memory usage is within safe limits (default 8GB)"""
        memory = self.get_memory_usage()
        if memory["rss_mb"] > max_usage_mb:
            print(f"⚠️  Memory usage ({memory['rss_mb']:.1f} MB) exceeds safe limit ({max_usage_mb} MB)")
            return False
        return True
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        return self.process.cpu_percent(interval=0.1)
    
    async def create_large_dataset(self, library_id: str, chunk_count: int = 1000) -> List[str]:
        """Create large dataset for performance testing"""
        print(f"Creating large dataset with {chunk_count} chunks...")
        
        # Generate diverse text patterns
        base_patterns = [
            "machine learning artificial intelligence neural networks deep learning",
            "database management systems distributed computing cloud infrastructure",
            "natural language processing text analysis semantic understanding",
            "computer vision image recognition pattern detection algorithms",
            "software engineering development practices agile methodologies",
            "data science analytics statistics predictive modeling",
            "cybersecurity network security information protection",
            "web development frontend backend full stack programming",
            "mobile application development cross platform solutions",
            "blockchain cryptocurrency decentralized systems"
        ]
        
        chunks = []
        for i in range(chunk_count):
            base_pattern = base_patterns[i % len(base_patterns)]
            # Add unique variations
            variation = f"Document {i} section {i % 10}: {base_pattern} with additional context {i} and metadata {uuid.uuid4()}"
            chunks.append(variation)
        
        # Create document in batches to avoid timeouts
        batch_size = 50
        created_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Check memory safety before creating each batch
            if not self.check_memory_safety():
                print(f"❌ Aborting dataset creation due to memory safety concerns")
                break
            
            document_data = {
                "chunk_texts": batch,
                "document_metadata": {
                    "title": f"Performance Test Document Batch {i // batch_size}",
                    "batch_number": i // batch_size,
                    "total_chunks": len(batch)
                },
                "chunk_source": f"performance_test_batch_{i // batch_size}"
            }
            
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.base_url}/api/libraries/{library_id}/documents/",
                    json=document_data
                )
                
                if response.status_code == 201:
                    result = response.json()
                    created_chunks.extend(result["chunk_ids"])
                    print(f"Created batch {i // batch_size + 1}, total chunks: {len(created_chunks)}")
                else:
                    print(f"Failed to create batch {i // batch_size}: {response.status_code}")
        
        # Wait for background processing
        print("Waiting for background processing...")
        await asyncio.sleep(10)
        
        # Final memory check
        final_memory = self.get_memory_usage()
        print(f"Final memory usage: {final_memory['rss_mb']:.1f} MB ({final_memory['percent']:.1f}%)")
        
        return created_chunks
    
    async def measure_search_performance(self, library_id: str, query: str, k: int = 10, iterations: int = 10) -> Dict[str, Any]:
        """Measure search performance over multiple iterations"""
        times = []
        results_counts = []
        
        async with httpx.AsyncClient(timeout=30) as client:
            for i in range(iterations):
                search_data = {
                    "query": query,
                    "k": k,
                    "similarity_function": "cosine"
                }
                
                start_time = time.time()
                response = await client.post(
                    f"{self.base_url}/api/libraries/{library_id}/knn-search",
                    json=search_data
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    times.append(end_time - start_time)
                    results_counts.append(len(result["results"]))
                else:
                    print(f"Search failed: {response.status_code}")
        
        if times:
            return {
                "iterations": len(times),
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "median_time": sorted(times)[len(times) // 2],
                "avg_results": sum(results_counts) / len(results_counts),
                "times": times
            }
        else:
            return {"error": "No successful searches"}
    
    async def concurrent_search_test(self, library_id: str, concurrent_users: int = 10, searches_per_user: int = 5) -> Dict[str, Any]:
        """Test concurrent search performance"""
        print(f"Testing concurrent searches: {concurrent_users} users, {searches_per_user} searches each")
        
        async def single_user_searches(user_id: int):
            """Simulate single user performing multiple searches"""
            user_times = []
            queries = [
                "machine learning algorithms",
                "database management systems",
                "natural language processing",
                "computer vision techniques",
                "software development practices"
            ]
            
            async with httpx.AsyncClient(timeout=30) as client:
                for i in range(searches_per_user):
                    query = queries[i % len(queries)]
                    search_data = {
                        "query": f"{query} user {user_id}",
                        "k": 5,
                        "similarity_function": "cosine"
                    }
                    
                    start_time = time.time()
                    try:
                        response = await client.post(
                            f"{self.base_url}/api/libraries/{library_id}/knn-search",
                            json=search_data
                        )
                        end_time = time.time()
                        
                        if response.status_code == 200:
                            user_times.append(end_time - start_time)
                    except Exception as e:
                        print(f"User {user_id} search failed: {e}")
            
            return {"user_id": user_id, "times": user_times}
        
        # Run concurrent users
        start_time = time.time()
        
        tasks = []
        for user_id in range(concurrent_users):
            task = asyncio.create_task(single_user_searches(user_id))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Analyze results
        all_times = []
        successful_users = 0
        
        for result in results:
            if result["times"]:
                all_times.extend(result["times"])
                successful_users += 1
        
        if all_times:
            return {
                "concurrent_users": concurrent_users,
                "searches_per_user": searches_per_user,
                "successful_users": successful_users,
                "total_searches": len(all_times),
                "total_time": total_time,
                "avg_search_time": sum(all_times) / len(all_times),
                "min_search_time": min(all_times),
                "max_search_time": max(all_times),
                "searches_per_second": len(all_times) / total_time
            }
        else:
            return {"error": "No successful concurrent searches"}


@pytest_asyncio.fixture(scope="function")
async def performance_suite():
    """Create performance test suite"""
    suite = PerformanceTestSuite()
    yield suite
    await suite.cleanup()


@pytest.mark.asyncio
class TestLargeDatasetPerformance:
    """Test performance with large datasets"""
    
    async def test_large_dataset_creation(self, performance_suite):
        """Test creating large dataset"""
        # Create library
        library_data = {
            "name": "Large Dataset Performance Test",
            "description": "Testing performance with large datasets",
            "preferred_index_algorithm": "flat"
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{performance_suite.base_url}/api/libraries/",
                json=library_data
            )
            assert response.status_code == 201
            library = response.json()
            performance_suite.test_libraries.append(library["id"])
        
        # Measure memory before
        memory_before = performance_suite.get_memory_usage()
        
        # Create large dataset
        start_time = time.time()
        chunk_ids = await performance_suite.create_large_dataset(library["id"], chunk_count=500)
        creation_time = time.time() - start_time
        
        # Measure memory after
        memory_after = performance_suite.get_memory_usage()
        
        # Verify creation
        assert len(chunk_ids) > 0
        print(f"Created {len(chunk_ids)} chunks in {creation_time:.2f}s")
        print(f"Memory usage: {memory_before['rss_mb']:.1f} MB → {memory_after['rss_mb']:.1f} MB")
        
        # Test search performance
        search_perf = await performance_suite.measure_search_performance(
            library["id"], 
            "machine learning artificial intelligence",
            k=10,
            iterations=5
        )
        
        print(f"Search performance: {search_perf['avg_time']:.3f}s average")
        assert search_perf["avg_time"] < 1.0  # Should be fast even with large dataset
    
    async def test_algorithm_performance_comparison(self, performance_suite):
        """Compare algorithm performance with large dataset"""
        algorithms = ["flat", "lsh", "grid"]
        results = {}
        
        for algorithm in algorithms:
            # Create library with specific algorithm
            library_data = {
                "name": f"Performance Test {algorithm.upper()}",
                "description": f"Testing {algorithm} performance",
                "preferred_index_algorithm": algorithm
            }
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{performance_suite.base_url}/api/libraries/",
                    json=library_data
                )
                assert response.status_code == 201
                library = response.json()
                performance_suite.test_libraries.append(library["id"])
            
            # Create dataset
            await performance_suite.create_large_dataset(library["id"], chunk_count=200)
            
            # Measure search performance
            search_perf = await performance_suite.measure_search_performance(
                library["id"],
                "machine learning data science algorithms",
                k=10,
                iterations=10
            )
            
            results[algorithm] = search_perf
            print(f"{algorithm.upper()} performance: {search_perf['avg_time']:.3f}s ± {search_perf['max_time'] - search_perf['min_time']:.3f}s")
        
        # All algorithms should perform reasonably
        for algorithm in algorithms:
            assert results[algorithm]["avg_time"] < 2.0  # Should be under 2 seconds
    
    async def test_concurrent_access_performance(self, performance_suite):
        """Test concurrent access performance"""
        # Create library with substantial data
        library_data = {
            "name": "Concurrent Access Test",
            "description": "Testing concurrent access patterns",
            "preferred_index_algorithm": "flat"
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{performance_suite.base_url}/api/libraries/",
                json=library_data
            )
            assert response.status_code == 201
            library = response.json()
            performance_suite.test_libraries.append(library["id"])
        
        # Create dataset
        await performance_suite.create_large_dataset(library["id"], chunk_count=300)
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        for concurrent_users in concurrency_levels:
            print(f"\nTesting {concurrent_users} concurrent users...")
            
            # Check memory safety before starting concurrent operations
            if not performance_suite.check_memory_safety():
                print(f"❌ Skipping {concurrent_users} concurrent users due to memory safety concerns")
                continue
            
            memory_before = performance_suite.get_memory_usage()
            cpu_before = performance_suite.get_cpu_usage()
            
            result = await performance_suite.concurrent_search_test(
                library["id"],
                concurrent_users=concurrent_users,
                searches_per_user=3
            )
            
            memory_after = performance_suite.get_memory_usage()
            cpu_after = performance_suite.get_cpu_usage()
            
            if "error" not in result:
                print(f"  {result['searches_per_second']:.1f} searches/second")
                print(f"  Average search time: {result['avg_search_time']:.3f}s")
                print(f"  Memory: {memory_before['rss_mb']:.1f} → {memory_after['rss_mb']:.1f} MB")
                print(f"  CPU: {cpu_before:.1f}% → {cpu_after:.1f}%")
                
                # Performance should not degrade significantly
                assert result["avg_search_time"] < 5.0
                assert result["searches_per_second"] > 0.5
            else:
                print(f"  Error: {result['error']}")
        
        # Final memory check after all concurrent tests
        final_memory = performance_suite.get_memory_usage()
        print(f"\nFinal memory usage after concurrent tests: {final_memory['rss_mb']:.1f} MB ({final_memory['percent']:.1f}%)")


@pytest.mark.asyncio
class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling"""
    
    async def test_invalid_library_operations(self, performance_suite):
        """Test operations on invalid libraries"""
        fake_library_id = str(uuid.uuid4())
        
        async with httpx.AsyncClient(timeout=30) as client:
            # Try to get non-existent library
            response = await client.get(f"{performance_suite.base_url}/api/libraries/{fake_library_id}")
            assert response.status_code == 404
            
            # Try to update non-existent library
            response = await client.put(
                f"{performance_suite.base_url}/api/libraries/{fake_library_id}",
                json={"name": "Updated Name"}
            )
            assert response.status_code == 404
            
            # Try to delete non-existent library
            response = await client.delete(f"{performance_suite.base_url}/api/libraries/{fake_library_id}")
            assert response.status_code == 404
            
            # Try to search non-existent library
            response = await client.post(
                f"{performance_suite.base_url}/api/libraries/{fake_library_id}/knn-search",
                json={"query": "test", "k": 5}
            )
            assert response.status_code == 404
    
    async def test_invalid_search_parameters(self, performance_suite):
        """Test invalid search parameters"""
        # Create valid library
        library_data = {
            "name": "Invalid Search Test",
            "description": "Testing invalid search parameters",
            "preferred_index_algorithm": "flat"
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{performance_suite.base_url}/api/libraries/",
                json=library_data
            )
            assert response.status_code == 201
            library = response.json()
            performance_suite.test_libraries.append(library["id"])
        
        # Add some data
        await performance_suite.create_large_dataset(library["id"], chunk_count=10)
        
        async with httpx.AsyncClient(timeout=30) as client:
            # Test invalid k values
            response = await client.post(
                f"{performance_suite.base_url}/api/libraries/{library['id']}/knn-search",
                json={"query": "test", "k": 0}
            )
            assert response.status_code == 422  # Validation error
            
            response = await client.post(
                f"{performance_suite.base_url}/api/libraries/{library['id']}/knn-search",
                json={"query": "test", "k": 101}  # Over limit
            )
            assert response.status_code == 422  # Validation error
            
            # Test empty query
            response = await client.post(
                f"{performance_suite.base_url}/api/libraries/{library['id']}/knn-search",
                json={"query": "", "k": 5}
            )
            assert response.status_code == 422  # Validation error
            
            # Test invalid similarity function
            response = await client.post(
                f"{performance_suite.base_url}/api/libraries/{library['id']}/knn-search",
                json={"query": "test", "k": 5, "similarity_function": "invalid"}
            )
            assert response.status_code == 422  # Validation error
    
    async def test_service_health_monitoring(self, performance_suite):
        """Test service health and monitoring endpoints"""
        async with httpx.AsyncClient(timeout=30) as client:
            # Test health endpoint
            response = await client.get(f"{performance_suite.base_url}/health")
            assert response.status_code == 200
            health = response.json()
            assert health["status"] == "healthy"
            
            # Test root endpoint
            response = await client.get(f"{performance_suite.base_url}/")
            assert response.status_code == 200
            root = response.json()
            assert "message" in root
            assert "docs" in root
            
            # Test vector service status
            response = await client.get(f"{performance_suite.base_url}/vector-service/status")
            assert response.status_code == 200
            status = response.json()
            assert "status" in status
            assert "libraries_indexed" in status
    
    async def test_background_task_monitoring(self, performance_suite):
        """Test background task processing"""
        # Create library
        library_data = {
            "name": "Background Task Test",
            "description": "Testing background task processing",
            "preferred_index_algorithm": "flat"
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{performance_suite.base_url}/api/libraries/",
                json=library_data
            )
            assert response.status_code == 201
            library = response.json()
            performance_suite.test_libraries.append(library["id"])
        
        # Create document (should trigger background tasks)
        document_data = {
            "chunk_texts": ["Test chunk for background processing"],
            "document_metadata": {"title": "Background Test"},
            "chunk_source": "background_test"
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{performance_suite.base_url}/api/libraries/{library['id']}/documents/",
                json=document_data
            )
            assert response.status_code == 201
            result = response.json()
            assert result["background_task_scheduled"] is True
            
            # Wait for background processing
            await asyncio.sleep(3)
            
            # Verify chunks are indexed
            response = await client.get(f"{performance_suite.base_url}/api/libraries/{library['id']}/chunks/")
            assert response.status_code == 200
            chunks = response.json()
            assert len(chunks) > 0
            
            # Test search to verify indexing worked
            response = await client.post(
                f"{performance_suite.base_url}/api/libraries/{library['id']}/knn-search",
                json={"query": "test chunk", "k": 5}
            )
            assert response.status_code == 200
            search_result = response.json()
            assert len(search_result["results"]) > 0
    
    async def test_memory_leak_detection(self, performance_suite):
        """Test for memory leaks during operations"""
        # Create library
        library_data = {
            "name": "Memory Leak Test",
            "description": "Testing for memory leaks",
            "preferred_index_algorithm": "flat"
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{performance_suite.base_url}/api/libraries/",
                json=library_data
            )
            assert response.status_code == 201
            library = response.json()
            performance_suite.test_libraries.append(library["id"])
        
        # Measure initial memory
        initial_memory = performance_suite.get_memory_usage()
        
        # Perform many operations
        for i in range(20):
            # Create document
            document_data = {
                "chunk_texts": [f"Memory test chunk {i} with content"],
                "document_metadata": {"title": f"Memory Test {i}"},
                "chunk_source": f"memory_test_{i}"
            }
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{performance_suite.base_url}/api/libraries/{library['id']}/documents/",
                    json=document_data
                )
                assert response.status_code == 201
                
                # Perform search
                response = await client.post(
                    f"{performance_suite.base_url}/api/libraries/{library['id']}/knn-search",
                    json={"query": f"memory test {i}", "k": 5}
                )
                assert response.status_code == 200
        
        # Wait for background processing
        await asyncio.sleep(5)
        
        # Measure final memory
        final_memory = performance_suite.get_memory_usage()
        
        # Memory should not have increased excessively
        memory_increase = final_memory["rss_mb"] - initial_memory["rss_mb"]
        print(f"Memory increase: {memory_increase:.1f} MB")
        
        # Allow for some memory growth but not excessive
        assert memory_increase < 200  # Should not use more than 200MB additional 