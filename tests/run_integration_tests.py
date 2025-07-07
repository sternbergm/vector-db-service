#!/usr/bin/env python3
"""
Comprehensive Integration Test Runner for Vector Database Service

This script runs all integration tests and generates a detailed report:
1. Basic functionality tests
2. Algorithm and similarity function tests
3. Performance and load tests
4. Edge cases and error handling
5. Memory and resource usage monitoring

Usage:
    python run_integration_tests.py [--quick] [--performance] [--report]

Options:
    --quick: Run only basic functionality tests
    --performance: Run performance tests (takes longer)
    --report: Generate detailed HTML report
"""

import asyncio
import subprocess
import sys
import time
import json
import argparse
import os
from datetime import datetime
from typing import Dict, List, Any
import httpx
import pytest


class TestRunner:
    """Comprehensive test runner for vector database service"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        self.service_process = None
    
    async def check_service_health(self) -> bool:
        """Check if the service is running and healthy"""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False
    
    async def wait_for_service(self, timeout: int = 60) -> bool:
        """Wait for service to be ready"""
        print("Waiting for service to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if await self.check_service_health():
                print("Service is ready!")
                return True
            await asyncio.sleep(1)
        
        print("Service did not become ready within timeout")
        return False
    
    def run_test_suite(self, test_file: str, test_name: str) -> Dict[str, Any]:
        """Run a specific test suite"""
        print(f"\n{'='*60}")
        print(f"Running {test_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Run pytest with detailed output
        cmd = [
            sys.executable, "-m", "pytest", 
            test_file, 
            "-v", "--tb=short", "--no-header",
            "--json-report", "--json-report-file=temp_report.json"
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout per test suite
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse JSON report if available
            test_details = {}
            if os.path.exists("temp_report.json"):
                try:
                    with open("temp_report.json", "r") as f:
                        test_details = json.load(f)
                    os.remove("temp_report.json")
                except Exception as e:
                    print(f"Could not parse test report: {e}")
            
            return {
                "name": test_name,
                "file": test_file,
                "duration": duration,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
                "details": test_details
            }
            
        except subprocess.TimeoutExpired:
            return {
                "name": test_name,
                "file": test_file,
                "duration": 300,
                "return_code": -1,
                "stdout": "",
                "stderr": "Test suite timed out after 5 minutes",
                "success": False,
                "details": {}
            }
        except Exception as e:
            return {
                "name": test_name,
                "file": test_file,
                "duration": 0,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False,
                "details": {}
            }
    
    async def run_all_tests(self, quick: bool = False, performance: bool = False):
        """Run all integration tests"""
        self.start_time = time.time()
        
        # Check service health
        if not await self.wait_for_service():
            print("ERROR: Service is not running or not healthy")
            print("Please start the service with: python main.py")
            return False
        
        # Define test suites
        test_suites = [
            ("test_integration_comprehensive.py", "Core API Functionality Tests"),
            ("test_integration_algorithms.py", "Algorithm & Similarity Function Tests"),
        ]
        
        if performance:
            test_suites.append(("test_integration_performance.py", "Performance & Load Tests"))
        
        if quick:
            test_suites = test_suites[:1]  # Only run basic tests
        
        # Run test suites
        for test_file, test_name in test_suites:
            if os.path.exists(test_file):
                result = self.run_test_suite(test_file, test_name)
                self.test_results[test_name] = result
                
                if result["success"]:
                    print(f"✅ {test_name} - PASSED ({result['duration']:.1f}s)")
                else:
                    print(f"❌ {test_name} - FAILED ({result['duration']:.1f}s)")
                    print(f"   Error: {result['stderr'][:200]}...")
            else:
                print(f"⚠️  Test file not found: {test_file}")
        
        self.end_time = time.time()
        return True
    
    def print_summary(self):
        """Print test summary"""
        if not self.test_results:
            print("No test results to summarize")
            return
        
        total_duration = self.end_time - self.start_time
        passed = sum(1 for r in self.test_results.values() if r["success"])
        total = len(self.test_results)
        
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total test suites: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Total duration: {total_duration:.1f}s")
        
        print(f"\nDetailed Results:")
        for name, result in self.test_results.items():
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            print(f"  {status} {name} ({result['duration']:.1f}s)")
            
            if not result["success"]:
                print(f"    Error: {result['stderr'][:100]}...")
        
        if passed == total:
            print(f"\n🎉 All tests passed! Vector database service is working correctly.")
        else:
            print(f"\n⚠️  {total - passed} test suite(s) failed. Please check the errors above.")
    
    def generate_html_report(self, filename: str = "test_report.html"):
        """Generate HTML test report"""
        if not self.test_results:
            print("No test results to generate report")
            return
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Vector Database Service - Integration Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; margin-bottom: 20px; }}
        .summary {{ background-color: #e8f4f8; padding: 15px; margin: 20px 0; }}
        .test-suite {{ border: 1px solid #ddd; margin: 20px 0; padding: 15px; }}
        .passed {{ border-left: 5px solid #28a745; }}
        .failed {{ border-left: 5px solid #dc3545; }}
        .details {{ background-color: #f8f9fa; padding: 10px; margin: 10px 0; }}
        pre {{ background-color: #f1f1f1; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Vector Database Service - Integration Test Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total test suites:</strong> {len(self.test_results)}</p>
        <p><strong>Passed:</strong> {sum(1 for r in self.test_results.values() if r['success'])}</p>
        <p><strong>Failed:</strong> {sum(1 for r in self.test_results.values() if not r['success'])}</p>
        <p><strong>Total duration:</strong> {self.end_time - self.start_time:.1f}s</p>
    </div>
"""
        
        for name, result in self.test_results.items():
            status_class = "passed" if result["success"] else "failed"
            status_text = "PASSED" if result["success"] else "FAILED"
            
            html_content += f"""
    <div class="test-suite {status_class}">
        <h3>{name} - {status_text}</h3>
        <p><strong>Duration:</strong> {result['duration']:.1f}s</p>
        <p><strong>Return Code:</strong> {result['return_code']}</p>
        
        {f'''
        <div class="details">
            <h4>Output:</h4>
            <pre>{result['stdout']}</pre>
        </div>
        ''' if result['stdout'] else ''}
        
        {f'''
        <div class="details">
            <h4>Errors:</h4>
            <pre>{result['stderr']}</pre>
        </div>
        ''' if result['stderr'] else ''}
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open(filename, "w") as f:
            f.write(html_content)
        
        print(f"HTML report generated: {filename}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run integration tests for vector database service")
    parser.add_argument("--quick", action="store_true", help="Run only basic functionality tests")
    parser.add_argument("--performance", action="store_true", help="Include performance tests")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    print("Vector Database Service - Integration Test Runner")
    print("=" * 60)
    
    if args.quick:
        print("Running in QUICK mode (basic tests only)")
    elif args.performance:
        print("Running in PERFORMANCE mode (includes load tests)")
    else:
        print("Running in STANDARD mode")
    
    success = await runner.run_all_tests(quick=args.quick, performance=args.performance)
    
    if success:
        runner.print_summary()
        
        if args.report:
            runner.generate_html_report()
    else:
        print("Failed to run tests - service not available")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 