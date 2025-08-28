#!/usr/bin/env python3
"""
Script to start LiteLLM proxy server with our custom model configuration
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['litellm', 'uvicorn', 'httpx']
    missing = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Installing missing packages...")
        for package in missing:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
        print("✅ Dependencies installed")

def start_proxy_server():
    """Start the LiteLLM proxy server"""
    config_path = Path(__file__).parent / "custom_config.yaml"

    print("🚀 Starting LiteLLM Proxy Server...")
    print(f"📁 Config file: {config_path}")

    # Start the proxy server
    cmd = [
        'litellm',
        '--config', str(config_path),
        '--host', '0.0.0.0',
        '--port', '4000',
        '--debug'
    ]

    print(f"🔧 Command: {' '.join(cmd)}")

    try:
        # Start server in background
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(5)

        # Test if server is running (try authenticated health check)
        try:
            # Try without auth first
            response = requests.get("http://localhost:4000/health", timeout=5)
            if response.status_code == 200:
                print("✅ LiteLLM Proxy Server is running at http://localhost:4000")
                print("📚 Swagger docs available at: http://localhost:4000/docs")
                print("🔑 Use 'Bearer sk-1234' for authentication")
                return process
            elif response.status_code == 401:
                # 401 is expected - server is running but requires auth
                print("✅ LiteLLM Proxy Server is running at http://localhost:4000")
                print("📚 Swagger docs available at: http://localhost:4000/docs")
                print("🔑 Use 'Bearer sk-1234' for authentication")
                print("ℹ️  Health endpoint requires authentication (this is normal)")
                return process
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return None
        except requests.RequestException as e:
            print(f"❌ Could not connect to server: {e}")
            return None

    except subprocess.SubprocessError as e:
        print(f"❌ Failed to start server: {e}")
        return None

def test_endpoints():
    """Test the custom model endpoints"""
    print("\n🧪 Testing Custom Model Endpoints...")

    base_url = "http://localhost:4000"
    headers = {
        "Authorization": "Bearer sk-1234",
        "Content-Type": "application/json"
    }

    # Test OpenAI format
    test_data = {
        "model": "our-custom-openai",
        "messages": [{"role": "user", "content": "Hello from OpenAI format!"}],
        "max_tokens": 50
    }

    try:
        response = requests.post(f"{base_url}/v1/chat/completions", 
                               json=test_data, headers=headers, timeout=30)
        if response.status_code == 200:
            print("✅ OpenAI format endpoint working")
            result = response.json()
            print(f"   Response: {result['choices'][0]['message']['content']}")
        else:
            print(f"❌ OpenAI format failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ OpenAI format error: {e}")

    # Test Anthropic format
    test_data["model"] = "our-custom-anthropic"
    test_data["messages"] = [{"role": "user", "content": "Hello from Anthropic format!"}]

    try:
        response = requests.post(f"{base_url}/v1/chat/completions", 
                               json=test_data, headers=headers, timeout=30)
        if response.status_code == 200:
            print("✅ Anthropic format endpoint working")
            result = response.json()
            print(f"   Response: {result['choices'][0]['message']['content']}")
        else:
            print(f"❌ Anthropic format failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Anthropic format error: {e}")

def main():
    """Main function"""
    print("🎯 LiteLLM Custom Model Proxy Setup\n")

    # Check dependencies
    check_dependencies()

    # Start proxy server
    server_process = start_proxy_server()

    if server_process:
        try:
            # Test endpoints
            test_endpoints()

            print("\n📋 Next Steps:")
            print("1. Server is running at http://localhost:4000")
            print("2. Run 'python test_preview_models.py' to test the setup")
            print("3. Install ARC AGI benchmark and run against the endpoints")
            print("4. Press Ctrl+C to stop the server")

            # Keep server running
            server_process.wait()

        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            server_process.terminate()
            server_process.wait()
            print("✅ Server stopped")
    else:
        print("❌ Failed to start proxy server")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
