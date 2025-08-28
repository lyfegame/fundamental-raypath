#!/usr/bin/env python3
"""
Test script for custom model endpoints via LiteLLM proxy
Tests both OpenAI and Anthropic format endpoints
"""

import asyncio
import json
import sys
import os
import time
import requests
from typing import List, Dict

class CustomModelTester:
    """Test custom model endpoints via LiteLLM proxy"""

    def __init__(self):
        self.base_url = "http://localhost:4000"
        self.headers = {
            "Authorization": "Bearer sk-1234",
            "Content-Type": "application/json"
        }
        self.results = {
            "proxy_health": {},
            "openai_tests": {},
            "anthropic_tests": {},
            "summary": {},
            "timestamp": time.time()
        }

        # Test queries for comprehensive evaluation
        self.test_queries = [
            {
                "type": "basic_chat",
                "messages": [{"role": "user", "content": "Hello! Can you help me with a simple task?"}],
                "max_tokens": 100
            },
            {
                "type": "reasoning",
                "messages": [{"role": "user", "content": "What's 2+2? Explain your reasoning step by step."}],
                "max_tokens": 150
            },
            {
                "type": "arc_pattern",
                "messages": [{"role": "user", "content": "Pattern: 1→2, 2→4, 3→6. What comes after 4? Explain the pattern."}],
                "max_tokens": 200
            }
        ]

    def check_proxy_health(self):
        """Check if the LiteLLM proxy is running"""
        print("🔍 Checking Proxy Health")
        print("="*50)

        try:
            # Test basic connectivity
            response = requests.get(f"{self.base_url}/health", headers=self.headers, timeout=10)

            if response.status_code == 200:
                print("✅ Proxy server is healthy")
                self.results["proxy_health"] = {
                    "status": "healthy",
                    "response": response.json()
                }
                return True
            else:
                print(f"❌ Proxy health check failed: {response.status_code}")
                self.results["proxy_health"] = {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                return False

        except requests.RequestException as e:
            print(f"❌ Cannot connect to proxy: {e}")
            self.results["proxy_health"] = {
                "status": "unreachable",
                "error": str(e)
            }
            return False

    def test_openai_format(self):
        """Test OpenAI format endpoints"""
        print("\n🤖 Testing OpenAI Format Endpoints")
        print("="*50)

        models_to_test = ["our-custom-openai"]  # Focus on working models

        for model in models_to_test:
            print(f"\n📡 Testing model: {model}")
            model_results = {}

            for query_info in self.test_queries:
                query_type = query_info["type"]
                print(f"   🔧 Testing {query_type}...")

                try:
                    start_time = time.time()

                    payload = {
                        "model": model,
                        "messages": query_info["messages"],
                        "max_tokens": query_info["max_tokens"],
                        "temperature": 0.7
                    }

                    response = requests.post(
                        f"{self.base_url}/v1/chat/completions",
                        json=payload,
                        headers=self.headers,
                        timeout=30
                    )

                    end_time = time.time()

                    if response.status_code == 200:
                        data = response.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        tokens = data.get("usage", {}).get("total_tokens", 0)

                        result = {
                            "success": True,
                            "response": content,
                            "tokens": tokens,
                            "response_time": end_time - start_time,
                            "status_code": response.status_code
                        }

                        print(f"      ✅ Success: {tokens} tokens, {result['response_time']:.2f}s")
                        print(f"         Response: {content[:100]}...")

                    else:
                        result = {
                            "success": False,
                            "error": f"HTTP {response.status_code}: {response.text}",
                            "response_time": end_time - start_time,
                            "status_code": response.status_code
                        }
                        print(f"      ❌ Failed: {result['error']}")

                    model_results[query_type] = result

                except Exception as e:
                    model_results[query_type] = {
                        "success": False,
                        "error": str(e),
                        "response_time": 0,
                        "status_code": None
                    }
                    print(f"      ❌ Exception: {e}")

                time.sleep(1)  # Rate limiting

            self.results["openai_tests"][model] = model_results

    def test_anthropic_format(self):
        """Test Anthropic format endpoints"""
        print("\n🧠 Testing Anthropic Format Endpoints")
        print("="*50)

        models_to_test = ["our-custom-anthropic"]  # Focus on working models

        for model in models_to_test:
            print(f"\n📡 Testing model: {model}")
            model_results = {}

            for query_info in self.test_queries:
                query_type = query_info["type"]
                print(f"   🔧 Testing {query_type}...")

                try:
                    start_time = time.time()

                    payload = {
                        "model": model,
                        "messages": query_info["messages"],
                        "max_tokens": query_info["max_tokens"],
                        "temperature": 0.7
                    }

                    response = requests.post(
                        f"{self.base_url}/v1/chat/completions",
                        json=payload,
                        headers=self.headers,
                        timeout=30
                    )

                    end_time = time.time()

                    if response.status_code == 200:
                        data = response.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        tokens = data.get("usage", {}).get("total_tokens", 0)

                        result = {
                            "success": True,
                            "response": content,
                            "tokens": tokens,
                            "response_time": end_time - start_time,
                            "status_code": response.status_code
                        }

                        print(f"      ✅ Success: {tokens} tokens, {result['response_time']:.2f}s")
                        print(f"         Response: {content[:100]}...")

                    else:
                        result = {
                            "success": False,
                            "error": f"HTTP {response.status_code}: {response.text}",
                            "response_time": end_time - start_time,
                            "status_code": response.status_code
                        }
                        print(f"      ❌ Failed: {result['error']}")

                    model_results[query_type] = result

                except Exception as e:
                    model_results[query_type] = {
                        "success": False,
                        "error": str(e),
                        "response_time": 0,
                        "status_code": None
                    }
                    print(f"      ❌ Exception: {e}")

                time.sleep(1)  # Rate limiting

            self.results["anthropic_tests"][model] = model_results

    def test_model_list(self):
        """Test model listing endpoint"""
        print("\n📋 Testing Model List Endpoint")
        print("="*50)

        try:
            response = requests.get(f"{self.base_url}/v1/models", headers=self.headers, timeout=10)

            if response.status_code == 200:
                models = response.json()
                print(f"✅ Retrieved {len(models.get('data', []))} models")
                for model in models.get('data', [])[:5]:  # Show first 5
                    print(f"   📦 {model.get('id', 'Unknown')}")
                return True
            else:
                print(f"❌ Model list failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Model list error: {e}")
            return False

    def generate_summary(self):
        """Generate comprehensive test summary"""
        print("\n" + "="*60)
        print("📊 CUSTOM MODEL TEST SUMMARY")
        print("="*60)

        # Count totals
        total_tests = 0
        successful_tests = 0
        total_tokens = 0
        total_time = 0

        print(f"\n🔍 Proxy Health: {self.results['proxy_health'].get('status', 'unknown')}")

        # Analyze OpenAI tests
        print(f"\n🤖 OpenAI Format Results:")
        for model, tests in self.results["openai_tests"].items():
            print(f"   Model: {model}")
            for test_type, result in tests.items():
                total_tests += 1
                if result.get("success"):
                    successful_tests += 1
                    total_tokens += result.get("tokens", 0)
                    total_time += result.get("response_time", 0)
                    print(f"     ✅ {test_type}: {result.get('tokens', 0)} tokens, {result.get('response_time', 0):.2f}s")
                else:
                    print(f"     ❌ {test_type}: {result.get('error', 'Unknown error')}")

        # Analyze Anthropic tests
        print(f"\n🧠 Anthropic Format Results:")
        for model, tests in self.results["anthropic_tests"].items():
            print(f"   Model: {model}")
            for test_type, result in tests.items():
                total_tests += 1
                if result.get("success"):
                    successful_tests += 1
                    total_tokens += result.get("tokens", 0)
                    total_time += result.get("response_time", 0)
                    print(f"     ✅ {test_type}: {result.get('tokens', 0)} tokens, {result.get('response_time', 0):.2f}s")
                else:
                    print(f"     ❌ {test_type}: {result.get('error', 'Unknown error')}")

        # Calculate statistics
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        avg_response_time = (total_time / successful_tests) if successful_tests > 0 else 0

        self.results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "total_tokens": total_tokens,
            "average_response_time": avg_response_time,
            "proxy_healthy": self.results["proxy_health"].get("status") == "healthy"
        }

        print(f"\n📈 Overall Statistics:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful: {successful_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Total Tokens: {total_tokens}")
        print(f"   Avg Response Time: {avg_response_time:.2f}s")

        # Save results
        with open('custom_model_test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n💾 Results saved to: custom_model_test_results.json")

        return success_rate > 75  # Return True if most tests passed

def main():
    """Main test execution"""
    print("🚀 Custom Model Testing via LiteLLM Proxy")
    print("Testing OpenAI and Anthropic format endpoints")
    print("="*60)

    tester = CustomModelTester()

    # Check proxy health first
    if not tester.check_proxy_health():
        print("\n❌ Proxy is not healthy. Please start the proxy first:")
        print("   python start_proxy.py")
        return False

    # Test model listing
    tester.test_model_list()

    # Test OpenAI format
    tester.test_openai_format()

    # Test Anthropic format  
    tester.test_anthropic_format()

    # Generate summary
    success = tester.generate_summary()

    if success:
        print("\n🎉 Custom model testing completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Review custom_model_test_results.json for detailed results")
        print("2. Run ARC-AGI-2 benchmark: python arc_benchmark.py")
        print("3. Run complete benchmark: python run_complete_benchmark.py")
    else:
        print("\n⚠️ Some tests failed. Check the results for details.")
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
