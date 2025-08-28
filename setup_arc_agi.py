#!/usr/bin/env python3
"""
Setup script for ARC AGI benchmark system
"""

import os
import sys
import subprocess
import json
import requests
from pathlib import Path

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing ARC AGI dependencies...")

    packages = [
        'numpy',
        'matplotlib', 
        'pandas',
        'tqdm',
        'requests',
        'Pillow'
    ]

    for package in packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         check=True, capture_output=True)
            print(f"   ✅ {package}")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")

def download_arc_data():
    """Download real ARC-AGI-2 dataset from GitHub"""
    print("📥 Downloading ARC-AGI-2 dataset...")

    data_dir = Path("arc_data")
    data_dir.mkdir(exist_ok=True)

    # Official ARC-AGI-2 repository URLs
    base_url = "https://raw.githubusercontent.com/arcprize/ARC-AGI-2/main/data"

    datasets = {
        "training": f"{base_url}/training",     # 1,000 tasks
        "evaluation": f"{base_url}/evaluation", # 120 tasks
    }

    for dataset_type, url_base in datasets.items():
        print(f"   📁 Downloading ARC-AGI-2 {dataset_type} dataset...")
        file_path = data_dir / f"{dataset_type}.json"

        try:
            # Download individual task files and combine them
            # ARC-AGI-2 stores tasks as individual JSON files in subdirectories
            print(f"   🔄 Downloading individual ARC-AGI-2 tasks...")
            combined_data = {}

            # Get the directory listing from GitHub API
            api_url = f"https://api.github.com/repos/arcprize/ARC-AGI-2/contents/data/{dataset_type}"

            try:
                response = requests.get(api_url, timeout=30)
                if response.status_code == 200:
                    files = response.json()
                    json_files = [f for f in files if f['name'].endswith('.json')]

                    print(f"   📂 Found {len(json_files)} task files in {dataset_type}")

                    for file_info in json_files[:50]:  # Limit to first 50 for testing
                        task_id = file_info['name'].replace('.json', '')
                        task_url = file_info['download_url']

                        try:
                            task_response = requests.get(task_url, timeout=10)
                            if task_response.status_code == 200:
                                combined_data[task_id] = task_response.json()
                            else:
                                print(f"     ⚠️  Failed to download task {task_id}")
                        except requests.RequestException as e:
                            print(f"     ⚠️  Network error for task {task_id}: {e}")
                            continue

                    if combined_data:
                        with open(file_path, 'w') as f:
                            json.dump(combined_data, f, indent=2)
                        print(f"   ✅ Combined {len(combined_data)} ARC-AGI-2 tasks into {dataset_type}.json")
                    else:
                        print(f"   ⚠️  No tasks downloaded for {dataset_type}, creating sample data...")
                        create_sample_data(file_path, dataset_type)
                else:
                    print(f"   ⚠️  Could not access GitHub API for {dataset_type}, creating sample data...")
                    create_sample_data(file_path, dataset_type)

            except requests.RequestException as e:
                print(f"   ⚠️  GitHub API error for {dataset_type}: {e}")
                print(f"   🔄 Creating sample data for development...")
                create_sample_data(file_path, dataset_type)

        except Exception as e:
            print(f"   ⚠️  Error downloading {dataset_type}: {e}")
            print(f"   🔄 Creating sample data for development...")
            create_sample_data(file_path, dataset_type)

def create_sample_data(file_path: Path, dataset_type: str):
    """Create sample ARC data for development/testing"""
    sample_tasks = {
        "00000001": {
            "train": [
                {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
                {"input": [[2, 0], [0, 2]], "output": [[0, 2], [2, 0]]}
            ],
            "test": [{"input": [[3, 0], [0, 3]], "output": [[0, 3], [3, 0]]}]
        },
        "00000002": {
            "train": [
                {"input": [[1, 1, 0], [0, 0, 0], [0, 0, 0]], "output": [[0, 0, 0], [0, 0, 0], [1, 1, 0]]},
                {"input": [[2, 2, 0], [0, 0, 0], [0, 0, 0]], "output": [[0, 0, 0], [0, 0, 0], [2, 2, 0]]}
            ],
            "test": [{"input": [[3, 3, 0], [0, 0, 0], [0, 0, 0]], "output": [[0, 0, 0], [0, 0, 0], [3, 3, 0]]}]
        },
        "00000003": {
            "train": [
                {"input": [[0, 1, 2], [3, 4, 5]], "output": [[3, 4, 5], [0, 1, 2]]},
                {"input": [[6, 7], [8, 9]], "output": [[8, 9], [6, 7]]}
            ],
            "test": [{"input": [[1, 2, 3], [4, 5, 6]], "output": [[4, 5, 6], [1, 2, 3]]}]
        }
    }

    with open(file_path, 'w') as f:
        json.dump(sample_tasks, f, indent=2)
    print(f"   ✅ Created sample {dataset_type}.json")

def create_arc_benchmark():
    """Create ARC AGI benchmark runner"""
    print("🔧 Creating ARC AGI benchmark runner...")

    benchmark_code = '''#!/usr/bin/env python3
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
        prompt = "You are solving an ARC-AGI-2 (Abstraction and Reasoning Corpus v2) task.\\n\\n"
        prompt += "Given the following input-output examples, identify the pattern and predict the output for the test input.\\n\\n"
        
        # Add training examples
        for i, example in enumerate(task_data.get("train", [])):
            prompt += f"Example {i+1}:\\n"
            prompt += f"Input: {example['input']}\\n"
            prompt += f"Output: {example['output']}\\n\\n"
        
        # Add test input
        if task_data.get("test"):
            prompt += f"Test Input: {task_data['test'][0]['input']}\\n"
            prompt += "Predict the output following the same pattern. Provide your reasoning and the final output grid.\\n"
        
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
            print(f"\\n🤖 Testing model: {model}")
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
        print("\\n📊 ARC-AGI-2 Benchmark Summary:")
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
'''

    with open("arc_benchmark.py", 'w') as f:
        f.write(benchmark_code)

    print("   ✅ Created arc_benchmark.py")

def main():
    """Main setup function"""
    print("🎯 Setting up ARC-AGI-2 Benchmark System\n")

    # Install dependencies
    install_dependencies()

    # Download ARC-AGI-2 data
    download_arc_data()

    # Create benchmark runner
    create_arc_benchmark()

    print("\n✅ ARC-AGI-2 setup complete!")
    print("\n📋 Next steps:")
    print("1. Start LiteLLM proxy: python start_proxy.py")
    print("2. Run ARC-AGI-2 benchmark: python arc_benchmark.py")
    print("3. Check results in arc_benchmark_results.json")
    print("\n🎯 ARC-AGI-2 Features:")
    print("   • 1,000 training tasks (vs 400 in ARC-AGI-1)")
    print("   • 120 evaluation tasks (focused set)")
    print("   • Enhanced pattern complexity")
    print("   • Latest benchmark standard")

if __name__ == "__main__":
    main()