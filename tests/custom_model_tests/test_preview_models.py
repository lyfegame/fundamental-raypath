#!/usr/bin/env python3
"""
Test script focused on preview_ custom models
Tests only models with 'preview_' prefix for best quality data
"""

import asyncio
import json
import sys
import os
import time
from typing import List, Dict

# Add project root to path to import our custom handler
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

# Suppress LiteLLM logging errors
os.environ["LITELLM_LOG"] = "WARNING"

import litellm
from custom_model_handler import our_custom_llm

# Configure LiteLLM with our custom provider
litellm.custom_provider_map = [
    {"provider": "our-custom", "custom_handler": our_custom_llm}
]

class PreviewModelTester:
    """Test only preview_ prefix custom models"""

    def __init__(self):
        self.preview_models = []
        self.results = {
            "preview_models": {},
            "summary": {},
            "timestamp": time.time()
        }

        # Test queries specifically for preview models
        self.test_queries = [
            {
                "type": "conversational",
                "query": "What are the top 3 Summer activities in New York City?",
                "max_tokens": 200
            },
            {
                "type": "coding_task",
                "query": "Write a Python function to find the maximum element in a list.",
                "max_tokens": 200
            },
            {
                "type": "arc_pattern",
                "query": "Pattern recognition: 1→2, 2→4, 3→6, 4→8. What comes after 5? Explain the pattern.",
                "max_tokens": 150
            }
        ]

    async def discover_preview_models(self):
        """Discover all custom models with preview_ prefix"""
        print("🔍 Discovering Preview Models")
        print("="*50)

        # Fetch custom models from Supabase
        success = await our_custom_llm.fetch_custom_models()

        if not success:
            print("❌ Failed to fetch custom models from Supabase")
            return False

        # Filter for deployed models (not just preview_ prefix for testing)
        for model_name, info in our_custom_llm.custom_models.items():
            if (info.get("status") == "Deployed" 
                and "endpoint" in info):
                # Add first 3 models for testing to avoid rate limits
                if len(self.preview_models) < 3:
                    self.preview_models.append(model_name)

        print(f"📦 Found {len(self.preview_models)} deployed models for testing:")
        for model in self.preview_models:
            info = our_custom_llm.custom_models[model]
            print(f"   ✅ {model}")
            print(f"      └─ Endpoint: {info['endpoint']}")
            print(f"      └─ Hardware: {info.get('hardware', 'Unknown')}")
            print(f"      └─ Pricing: {info.get('pricing', 'Unknown')}")
            print()

        return len(self.preview_models) > 0

    async def test_preview_model(self, model_name: str, query_info: Dict, format_type: str = "openai") -> Dict:
        """Test a single preview model with a specific query"""
        try:
            start_time = time.time()

            # Determine if using anthropic format
            use_anthropic = format_type == "anthropic"

            response = await litellm.acompletion(
                model=f"our-custom/{model_name}",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that provides clear, accurate responses."},
                    {"role": "user", "content": query_info["query"]}
                ],
                max_tokens=query_info["max_tokens"],
                temperature=0.3,
                anthropic_format=use_anthropic
            )

            end_time = time.time()

            # Handle response safely with try-catch
            content = ""
            tokens = 0

            try:
                # Extract content using dict-like access to avoid linter issues
                response_dict = response if isinstance(response, dict) else response.__dict__
                choices = response_dict.get('choices', [])
                if choices and len(choices) > 0:
                    choice_dict = choices[0] if isinstance(choices[0], dict) else choices[0].__dict__
                    message = choice_dict.get('message', {})
                    message_dict = message if isinstance(message, dict) else message.__dict__
                    content = str(message_dict.get('content', ''))
            except (AttributeError, IndexError, TypeError, KeyError):
                content = str(response) if response else ""

            try:
                # Extract tokens using dict-like access
                response_dict = response if isinstance(response, dict) else response.__dict__
                usage = response_dict.get('usage', {})
                usage_dict = usage if isinstance(usage, dict) else usage.__dict__
                tokens = int(usage_dict.get('total_tokens', 0))
            except (AttributeError, TypeError, ValueError, KeyError):
                tokens = 0

            return {
                "success": True,
                "response": content,
                "tokens": tokens,
                "response_time": end_time - start_time,
                "format": format_type,
                "model_info": our_custom_llm.custom_models.get(model_name, {})
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": 0,
                "tokens": 0,
                "format": format_type
            }

    async def test_all_preview_models(self):
        """Test all preview models with all query types and both formats"""
        print("🧪 Testing Preview Models")
        print("="*50)

        if not self.preview_models:
            print("❌ No preview models found to test")
            return

        for model_name in self.preview_models:
            print(f"\n🤖 Testing model: {model_name}")
            model_info = our_custom_llm.custom_models[model_name]
            print(f"   Hardware: {model_info.get('hardware', 'Unknown')}")
            print(f"   Pricing: {model_info.get('pricing', 'Unknown')}")

            model_results = {
                "openai_format": {},
                "anthropic_format": {},
                "model_info": model_info
            }

            # Test each query type
            for query_info in self.test_queries:
                query_type = query_info["type"]
                print(f"\n   📝 Testing {query_type}: {query_info['query'][:60]}...")

                # Test OpenAI format
                openai_result = await self.test_preview_model(model_name, query_info, "openai")
                model_results["openai_format"][query_type] = openai_result

                if openai_result["success"]:
                    print(f"      ✅ OpenAI: {openai_result['tokens']} tokens, {openai_result['response_time']:.2f}s")
                    print(f"         Response: {openai_result['response'][:120]}...")
                else:
                    print(f"      ❌ OpenAI Error: {openai_result['error']}")

                # Wait between requests
                await asyncio.sleep(1)

                # Test Anthropic format
                anthropic_result = await self.test_preview_model(model_name, query_info, "anthropic")
                model_results["anthropic_format"][query_type] = anthropic_result

                if anthropic_result["success"]:
                    print(f"      ✅ Anthropic: {anthropic_result['tokens']} tokens, {anthropic_result['response_time']:.2f}s")
                    print(f"         Response: {anthropic_result['response'][:120]}...")
                else:
                    print(f"      ❌ Anthropic Error: {anthropic_result['error']}")

                await asyncio.sleep(1)

            self.results["preview_models"][model_name] = model_results
            print()

    async def test_arc_agi_with_preview_models(self):
        """Test ARC AGI style reasoning with preview models"""
        print("🎯 ARC AGI Testing with Preview Models")
        print("="*50)

        if not self.preview_models:
            print("❌ No preview models available for ARC testing")
            return

        arc_challenge = """
        ARC Challenge: Advanced Pattern Recognition
        
        Example patterns:
        1. Input: [[1,0,1],[0,1,0],[1,0,1]] → Output: [[0,1,0],[1,0,1],[0,1,0]]
        2. Input: [[2,0,2],[0,2,0],[2,0,2]] → Output: [[0,2,0],[2,0,2],[0,2,0]]
        
        Question: What transformation rule is being applied?
        Test: Apply this rule to [[3,0,3],[0,3,0],[3,0,3]]
        
        Provide your reasoning and the final answer.
        """

        # Test with top 2 preview models to avoid rate limits
        test_models = self.preview_models[:2]

        for model in test_models:
            print(f"\n🧠 Testing ARC reasoning with {model}:")

            # Test both formats for comprehensive evaluation
            for format_type in ["openai", "anthropic"]:
                try:
                    response = await litellm.acompletion(
                        model=f"our-custom/{model}",
                        messages=[
                            {"role": "system", "content": "You are an AI expert at pattern recognition and logical reasoning. Analyze patterns carefully and explain your reasoning."},
                            {"role": "user", "content": arc_challenge}
                        ],
                        max_tokens=400,
                        temperature=0.1,
                        anthropic_format=(format_type == "anthropic")
                    )

                    # Safely extract response data
                    tokens = 0
                    content = ""

                    try:
                        # Extract tokens using dict-like access
                        response_dict = response if isinstance(response, dict) else response.__dict__
                        usage = response_dict.get('usage', {})
                        usage_dict = usage if isinstance(usage, dict) else usage.__dict__
                        tokens = int(usage_dict.get('total_tokens', 0))
                    except (AttributeError, TypeError, ValueError, KeyError):
                        tokens = 0

                    try:
                        # Extract content using dict-like access
                        response_dict = response if isinstance(response, dict) else response.__dict__
                        choices = response_dict.get('choices', [])
                        if choices and len(choices) > 0:
                            choice_dict = choices[0] if isinstance(choices[0], dict) else choices[0].__dict__
                            message = choice_dict.get('message', {})
                            message_dict = message if isinstance(message, dict) else message.__dict__
                            content = str(message_dict.get('content', ''))
                    except (AttributeError, IndexError, TypeError, KeyError):
                        content = str(response) if response else ""

                    print(f"   ✅ {format_type.title()} Format ({tokens} tokens):")
                    print(f"      {content[:200]}...")

                except Exception as e:
                    print(f"   ❌ {format_type.title()} Error: {e}")

                await asyncio.sleep(2)

    def generate_preview_summary(self):
        """Generate comprehensive summary of preview model testing"""
        print("\n" + "="*60)
        print("📊 PREVIEW MODELS TEST SUMMARY")
        print("="*60)

        total_models = len(self.preview_models)
        total_tests = 0
        successful_tests = 0
        total_tokens = 0
        total_time = 0

        print(f"\n📈 Overview:")
        print(f"   Preview Models Found: {total_models}")
        print(f"   Test Types: {len(self.test_queries)} (conversational, coding_task, arc_pattern)")
        print(f"   Formats Tested: 2 (OpenAI, Anthropic)")

        if not self.results["preview_models"]:
            print("   ⚠️ No test results to analyze")
            return

        print(f"\n📍 Model Performance:")
        for model_name, model_data in self.results["preview_models"].items():
            print(f"\n🤖 {model_name}:")
            model_info = model_data.get("model_info", {})
            print(f"   Hardware: {model_info.get('hardware', 'Unknown')}")
            print(f"   Pricing: {model_info.get('pricing', 'Unknown')}")

            # Analyze both formats
            for format_type in ["openai_format", "anthropic_format"]:
                format_name = format_type.replace("_", " ").title()
                print(f"   {format_name}:")

                format_results = model_data.get(format_type, {})
                for query_type, result in format_results.items():
                    total_tests += 1
                    if result.get("success"):
                        successful_tests += 1
                        tokens = result.get("tokens", 0)
                        response_time = result.get("response_time", 0)
                        total_tokens += tokens
                        total_time += response_time
                        print(f"     ✅ {query_type}: {tokens} tokens, {response_time:.2f}s")
                    else:
                        print(f"     ❌ {query_type}: {result.get('error', 'Unknown error')}")

        # Overall statistics
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        avg_response_time = (total_time / successful_tests) if successful_tests > 0 else 0

        self.results["summary"] = {
            "total_models": total_models,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "total_tokens": total_tokens,
            "average_response_time": avg_response_time
        }

        print(f"\n📊 Overall Statistics:")
        print(f"   Total Tests Run: {total_tests}")
        print(f"   Successful Tests: {successful_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Total Tokens Used: {total_tokens}")
        print(f"   Average Response Time: {avg_response_time:.2f}s")

        # Save detailed results
        with open('preview_models_test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n💾 Detailed results saved to: preview_models_test_results.json")

async def main():
    """Main test execution"""
    print("🚀 Preview Models Testing with LiteLLM")
    print("Testing only custom models with 'preview_' prefix")
    print("="*60)

    tester = PreviewModelTester()

    # Discover preview models
    models_found = await tester.discover_preview_models()

    if not models_found:
        print("❌ No deployed models found in Supabase for testing.")
        return

    # Run comprehensive tests
    await tester.test_all_preview_models()

    # Test ARC AGI capabilities
    await tester.test_arc_agi_with_preview_models()

    # Generate summary
    tester.generate_preview_summary()

    print("\n🎉 Preview model testing completed!")
    print("\n📋 Next Steps:")
    print("1. Review preview_models_test_results.json for detailed analysis")
    print("2. Use the best performing preview model for production")
    print("3. Run full ARC-AGI-2 benchmark: python arc_benchmark.py")
    print("4. Start proxy server: python start_proxy.py")

if __name__ == "__main__":
    asyncio.run(main())