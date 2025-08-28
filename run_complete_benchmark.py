#!/usr/bin/env python3
"""
Complete benchmark runner that demonstrates:
1. LiteLLM calling custom model in OpenAI and Anthropic formats
2. Running ARC AGI benchmark (aragi2) 
3. Generating comprehensive results

This is the main deliverable script.
"""

import asyncio
import subprocess
import time
import sys
import json
from pathlib import Path
import signal
import os

class CompleteBenchmarkRunner:
    """Orchestrates the complete benchmark process"""

    def __init__(self):
        self.proxy_process = None
        self.results = {
            "timestamp": time.time(),
            "setup_results": {},
            "proxy_status": {},
            "custom_model_tests": {},
            "arc_agi_results": {},
            "summary": {}
        }

    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

    async def setup_environment(self):
        """Setup the environment"""
        self.log("🔧 Setting up environment...")

        try:
            # Run setup script
            result = subprocess.run([sys.executable, "setup_arc_agi.py"], 
                                  capture_output=True, text=True, timeout=120)

            self.results["setup_results"] = {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

            if result.returncode == 0:
                self.log("✅ Environment setup completed")
                return True
            else:
                self.log(f"❌ Environment setup failed: {result.stderr}", "ERROR")
                return False

        except Exception as e:
            self.log(f"❌ Environment setup error: {e}", "ERROR")
            self.results["setup_results"] = {"success": False, "error": str(e)}
            return False

    def start_proxy_server(self):
        """Start LiteLLM proxy server"""
        self.log("🚀 Starting LiteLLM proxy server...")

        try:
            # Start proxy server directly with litellm command
            cmd = [
                "litellm",
                "--config", "custom_config.yaml",
                "--host", "0.0.0.0",
                "--port", "4000",
                "--debug"
            ]

            self.proxy_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for server to start and test health
            time.sleep(15)  # Give more time for startup

            # Test if server is responding
            try:
                import requests
                response = requests.get("http://localhost:4000/health", 
                                      headers={"Authorization": "Bearer sk-1234"}, 
                                      timeout=5)
                if response.status_code == 200 or response.status_code == 401:
                    self.log("✅ Proxy server started successfully")
                    self.results["proxy_status"] = {"success": True, "pid": self.proxy_process.pid}
                    return True
                else:
                    self.log(f"❌ Proxy server health check failed: {response.status_code}", "ERROR")
                    self.results["proxy_status"] = {"success": False, "error": f"Health check failed: {response.status_code}"}
                    return False
            except requests.RequestException as e:
                self.log(f"❌ Cannot reach proxy server: {e}", "ERROR")
                self.results["proxy_status"] = {"success": False, "error": f"Connection failed: {str(e)}"}
                return False

        except Exception as e:
            self.log(f"❌ Proxy server error: {e}", "ERROR")
            self.results["proxy_status"] = {"success": False, "error": str(e)}
            return False

    async def test_custom_models(self):
        """Test custom model implementations"""
        self.log("🧪 Testing custom model implementations...")

        try:
            # Run test script
            result = subprocess.run([sys.executable, "tests/custom_model_tests/test_custom_model.py"], 
                                  capture_output=True, text=True, timeout=180)

            self.results["custom_model_tests"] = {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

            if result.returncode == 0:
                self.log("✅ Custom model tests passed")
                return True
            else:
                self.log(f"❌ Custom model tests failed: {result.stderr}", "ERROR")
                return False

        except Exception as e:
            self.log(f"❌ Custom model test error: {e}", "ERROR")
            self.results["custom_model_tests"] = {"success": False, "error": str(e)}
            return False

    async def run_arc_agi_benchmark(self):
        """Run ARC AGI benchmark"""
        self.log("🎯 Running ARC AGI benchmark...")

        try:
            # Run ARC benchmark
            result = subprocess.run([sys.executable, "arc_benchmark.py"], 
                                  capture_output=True, text=True, timeout=300)

            # Load results if available
            results_file = Path("arc_benchmark_results.json")
            if results_file.exists():
                with open(results_file, 'r') as f:
                    arc_results = json.load(f)
                self.results["arc_agi_results"] = arc_results

            if result.returncode == 0:
                self.log("✅ ARC AGI benchmark completed successfully")
                return True
            else:
                self.log(f"❌ ARC AGI benchmark failed: {result.stderr}", "ERROR")
                self.results["arc_agi_results"] = {"success": False, "error": result.stderr}
                return False

        except Exception as e:
            self.log(f"❌ ARC AGI benchmark error: {e}", "ERROR")
            self.results["arc_agi_results"] = {"success": False, "error": str(e)}
            return False

    def cleanup(self):
        """Cleanup resources"""
        self.log("🧹 Cleaning up...")

        if self.proxy_process and self.proxy_process.poll() is None:
            self.log("Stopping proxy server...")
            self.proxy_process.terminate()
            try:
                self.proxy_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proxy_process.kill()
                self.proxy_process.wait()
            self.log("✅ Proxy server stopped")

    def generate_summary(self):
        """Generate benchmark summary"""
        self.log("📊 Generating benchmark summary...")

        # Count successes
        setup_ok = self.results["setup_results"].get("success", False)
        proxy_ok = self.results["proxy_status"].get("success", False)
        tests_ok = self.results["custom_model_tests"].get("success", False)
        arc_ok = isinstance(self.results["arc_agi_results"], dict) and "model_results" in self.results["arc_agi_results"]

        total_steps = 4
        successful_steps = sum([setup_ok, proxy_ok, tests_ok, arc_ok])

        self.results["summary"] = {
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "success_rate": successful_steps / total_steps,
            "steps": {
                "environment_setup": setup_ok,
                "proxy_server": proxy_ok,
                "custom_model_tests": tests_ok,
                "arc_agi_benchmark": arc_ok
            }
        }

        # Print summary
        print("\\n" + "="*60)
        print("🏁 BENCHMARK SUMMARY")
        print("="*60)
        print(f"Overall Success Rate: {successful_steps}/{total_steps} ({successful_steps/total_steps:.1%})")
        print()

        print("📋 Step Results:")
        steps = [
            ("Environment Setup", setup_ok),
            ("Proxy Server", proxy_ok), 
            ("Custom Model Tests", tests_ok),
            ("ARC AGI Benchmark", arc_ok)
        ]

        for step_name, success in steps:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {step_name}: {status}")

        if arc_ok:
            print("\\n🎯 ARC AGI Results:")
            for model, model_data in self.results["arc_agi_results"]["model_results"].items():
                summary = model_data["summary"]
                print(f"  {model}:")
                print(f"    Success Rate: {summary['success_rate']:.1%}")
                print(f"    Avg Response Time: {summary['avg_response_time']:.2f}s")
                print(f"    Total Tokens: {summary['total_tokens']}")

        print("\\n💾 Detailed results saved to: complete_benchmark_results.json")
        print("="*60)

    def save_results(self):
        """Save complete results"""
        with open("complete_benchmark_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)

    async def run_complete_benchmark(self):
        """Run the complete benchmark process"""
        self.log("🎯 Starting Complete LiteLLM + ARC AGI Benchmark")

        try:
            # Step 1: Setup environment
            if not await self.setup_environment():
                return False

            # Step 2: Start proxy server  
            if not self.start_proxy_server():
                return False

            # Wait for server to be fully ready
            await asyncio.sleep(5)

            # Step 3: Test custom models
            if not await self.test_custom_models():
                return False

            # Step 4: Run ARC AGI benchmark
            if not await self.run_arc_agi_benchmark():
                return False

            self.log("🎉 Complete benchmark finished successfully!")
            return True

        except KeyboardInterrupt:
            self.log("⚠️ Benchmark interrupted by user", "WARN")
            return False
        except Exception as e:
            self.log(f"❌ Benchmark failed: {e}", "ERROR")
            return False
        finally:
            self.cleanup()
            self.generate_summary()
            self.save_results()

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\\n⚠️ Interrupt received, cleaning up...")
    sys.exit(0)

async def main():
    """Main function"""
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    print("🚀 LiteLLM Custom Model + ARC AGI Benchmark")
    print("="*50)
    print("This script demonstrates:")
    print("1. ✅ Custom model serving via LiteLLM")
    print("2. ✅ OpenAI and Anthropic format support") 
    print("3. ✅ ARC AGI benchmark execution")
    print("4. ✅ Complete results and analysis")
    print("="*50)
    print()

    runner = CompleteBenchmarkRunner()
    success = await runner.run_complete_benchmark()

    return 0 if success else 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
