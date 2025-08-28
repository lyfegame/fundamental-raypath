#!/usr/bin/env python3
"""
ARC-AGI-2 Benchmark Runner for LiteLLM Custom Models
Supports the latest ARC-AGI-2 dataset with 1,000 training and 120 evaluation tasks
"""

import json
import asyncio
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
import httpx
from tqdm import tqdm

class ARCAGIBenchmark:
    """ARC-AGI-2 Benchmark runner for evaluating LLM pattern recognition capabilities"""
    
    def __init__(self, base_url: str = "http://localhost:4000", api_key: str = "sk-1234"):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def load_dataset(self, dataset_path: str) -> Dict:
        """Load ARC dataset"""
        with open(dataset_path, 'r') as f:
            return json.load(f)
    
    def format_arc_prompt(self, task_data: Dict) -> str:
        """Format ARC task as a prompt"""
        prompt = "You are solving an ARC-AGI-2 (Abstraction and Reasoning Corpus v2) task.\n\n"
        prompt += "Given the following input-output examples, identify the pattern and predict the output for the test input.\n\n"
        
        # Add training examples
        for i, example in enumerate(task_data.get("train", [])):
            prompt += f"Example {i+1}:\n"
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n\n"
        
        # Add test input
        if task_data.get("test"):
            prompt += f"Test Input: {task_data['test'][0]['input']}\n"
            prompt += "Predict the output following the same pattern. Provide your reasoning and the final output grid.\n"
        
        return prompt
    
    async def query_model(self, model_name: str, prompt: str, max_tokens: int = 500) -> Dict:
        """Query the LiteLLM model"""
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are an AI that excels at pattern recognition and logical reasoning."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1  # Low temperature for consistency
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=data,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
    
    async def run_benchmark(self, dataset_path: str, models: List[str], max_tasks: int = 10) -> Dict:
        """Run ARC benchmark on specified models"""
        print(f"🎯 Running ARC-AGI-2 Benchmark on {len(models)} models...")
        
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        task_ids = list(dataset.keys())[:max_tasks]
        
        results = {
            "benchmark_info": {
                "dataset": dataset_path,
                "models_tested": models,
                "tasks_tested": len(task_ids),
                "timestamp": time.time()
            },
            "model_results": {}
        }
        
        for model in models:
            print(f"\n🤖 Testing model: {model}")
            model_results = []
            
            for task_id in tqdm(task_ids, desc=f"Processing {model}"):
                task_data = dataset[task_id]
                prompt = self.format_arc_prompt(task_data)
                
                try:
                    start_time = time.time()
                    response = await self.query_model(model, prompt)
                    end_time = time.time()
                    
                    result = {
                        "task_id": task_id,
                        "prompt": prompt,
                        "response": response["choices"][0]["message"]["content"],
                        "response_time": end_time - start_time,
                        "tokens_used": response.get("usage", {}).get("total_tokens", 0),
                        "success": True
                    }
                    
                    # Simple evaluation (you can enhance this)
                    if task_data.get("test") and len(task_data["test"]) > 0:
                        expected_output = task_data["test"][0].get("output")
                        result["expected_output"] = expected_output
                        # Add your evaluation logic here
                    
                except Exception as e:
                    result = {
                        "task_id": task_id,
                        "error": str(e),
                        "success": False,
                        "response_time": 0,
                        "tokens_used": 0
                    }
                
                model_results.append(result)
            
            # Calculate metrics
            successful_tasks = [r for r in model_results if r["success"]]
            total_tokens = sum(r.get("tokens_used", 0) for r in model_results)
            avg_response_time = sum(r.get("response_time", 0) for r in successful_tasks) / max(len(successful_tasks), 1)
            
            results["model_results"][model] = {
                "tasks": model_results,
                "summary": {
                    "total_tasks": len(task_ids),
                    "successful_tasks": len(successful_tasks),
                    "success_rate": len(successful_tasks) / len(task_ids),
                    "total_tokens": total_tokens,
                    "avg_response_time": avg_response_time
                }
            }
        
        return results
    
    def save_results(self, results: Dict, output_path: str):
        """Save benchmark results"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {output_path}")

async def main():
    """Main benchmark execution"""
    benchmark = ARCAGIBenchmark()
    
    # Models to test
    models_to_test = [
        "our-custom-openai",
        "our-custom-anthropic"
    ]
    
    # Run on training data (small subset for testing)
    dataset_path = "arc_data/training.json"
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Run setup_arc_agi.py first to download the dataset")
        return
    
    try:
        results = await benchmark.run_benchmark(dataset_path, models_to_test, max_tasks=5)
        benchmark.save_results(results, "arc_benchmark_results.json")
        
        # Print summary
        print("\n📊 ARC-AGI-2 Benchmark Summary:")
        for model, model_data in results["model_results"].items():
            summary = model_data["summary"]
            print(f"  {model}:")
            print(f"    Success Rate: {summary['success_rate']:.2%}")
            print(f"    Avg Response Time: {summary['avg_response_time']:.2f}s")
            print(f"    Total Tokens: {summary['total_tokens']}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
