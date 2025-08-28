"""
Custom Model Handler for LiteLLM
Supports both OpenAI and Anthropic format responses
"""

import litellm
from litellm import CustomLLM
from litellm.types.utils import ModelResponse, Usage, Choices, Message
import time
import json
from typing import Optional, Dict, Any, List
import httpx
import asyncio

class OurCustomLLM(CustomLLM):
    """
    Custom LLM handler that can serve models in both OpenAI and Anthropic formats
    """

    def __init__(self):
        super().__init__()
        # Official fundamental-photon.xyz API endpoints
        self.openai_endpoint = "https://api.fundamental-photon.xyz/api/inference/v1/chat/completions"
        self.anthropic_endpoint = "https://api.fundamental-photon.xyz/api/inference/v1/messages"

        # Static models from model-endpoints.ts configuration
        self.static_model_endpoints = {
            # From model-endpoints.ts configuration
            "FundamentalResearchLabs/qwen2.5_coder_v1": "https://model-7qkzp02q.api.baseten.co/production/predict",
            "FundamentalResearchLabs/ll8-4k-mu-unr-rt1-rel-yes-neg-dm0123456-0089090-b80e337-s1888": "https://fairies--ll8-4k-mu-unr-rt1-rel-yes-neg-dm01-2ebcca-vllm-serve.modal.run/v1",
            "Qwen/Qwen3-Coder-30B-A3B-Instruct": "https://fairies--qwen3-coder-30b-a3b-instruct-vllm-inference-serve.modal.run/v1",
            "Qwen/Qwen3-Coder-480B-A35B-Instruct": "https://fairies--qwen3-coder-480b-a35b-instruct-vllm-inference-serve.modal.run/v1"
        }

        # Custom models from Supabase (loaded dynamically)
        self.custom_models = {}  # Will be populated by fetch_custom_models()

        # Fallback to local for development
        self.local_openai = "http://localhost:3000/api/inference/v1/chat/completions"
        self.local_anthropic = "http://localhost:3000/api/inference/v1/messages"

    async def fetch_custom_models(self):
        """Fetch custom models from Supabase via fundamental-photon API"""
        try:
            print("🔄 Fetching custom models from Supabase...")

            async with httpx.AsyncClient(timeout=30.0) as client:
                # Call fundamental-photon API endpoint to get custom models
                response = await client.get("https://api.fundamental-photon.xyz/api/models")

                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])

                    print(f"📦 Found {len(models)} custom models in Supabase")

                    # Process custom models and extract endpoint information
                    for model in models:
                        if model.get("status") == "Deployed" and model.get("modalEndpoint"):
                            model_source = model.get("modelSource", "")
                            endpoint_url = model["modalEndpoint"].get("url", "")

                            # Store both the model name and modelSource as keys
                            self.custom_models[model.get("name", "")] = {
                                "endpoint": endpoint_url,
                                "modelSource": model_source,
                                "id": model.get("id", ""),
                                "status": model.get("status", ""),
                                "hardware": model["modalEndpoint"].get("hardware", ""),
                                "pricing": model["modalEndpoint"].get("pricing", "")
                            }

                            # Also store by modelSource for compatibility
                            if model_source:
                                self.custom_models[model_source] = self.custom_models[model.get("name", "")]

                            print(f"   ✅ {model.get('name')}: {endpoint_url}")

                    return True
                else:
                    print(f"❌ Failed to fetch models: {response.status_code}")
                    return False

        except Exception as e:
            print(f"❌ Error fetching custom models: {e}")
            return False

    def get_all_available_models(self) -> Dict[str, str]:
        """Get all available models (static + custom)"""
        all_models = {}

        # Add static models
        all_models.update(self.static_model_endpoints)

        # Add custom models
        for name, info in self.custom_models.items():
            all_models[name] = info["endpoint"]

        return all_models

    def _create_openai_response(self, content: str, model: str, usage_info: Optional[Dict] = None) -> ModelResponse:
        """Create OpenAI-format response"""
        if usage_info is None:
            usage_info = {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60}

        return ModelResponse(
            id=f"chatcmpl-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content=content,
                        role="assistant"
                    )
                )
            ],
            usage=Usage(**usage_info)
        )

    def _create_anthropic_response(self, content: str, model: str, usage_info: Optional[Dict] = None) -> ModelResponse:
        """Create Anthropic-format response"""
        if usage_info is None:
            usage_info = {"input_tokens": 10, "output_tokens": 50}

        # Convert to OpenAI format but maintain Anthropic-like structure
        return ModelResponse(
            id=f"msg_{int(time.time())}",
            object="message",
            created=int(time.time()),
            model=model,
            choices=[
                Choices(
                    finish_reason="end_turn",
                    index=0,
                    message=Message(
                        content=content,
                        role="assistant"
                    )
                )
            ],
            usage=Usage(
                prompt_tokens=usage_info.get("input_tokens", 10),
                completion_tokens=usage_info.get("output_tokens", 50),
                total_tokens=usage_info.get("input_tokens", 10) + usage_info.get("output_tokens", 50)
            )
        )

    async def _call_custom_model(self, messages: List[Dict], model: str, **kwargs) -> Dict:
        """
        Call fundamental-photon custom model API
        Returns the full response for proper usage tracking
        """
        # Ensure we have the latest custom models loaded
        if not self.custom_models:
            await self.fetch_custom_models()

        # Map LiteLLM model names to deployed fundamental-photon models
        model_mapping = {
            "our-custom-openai": "FundamentalResearchLabs/ll8-4k-mu-unr-rt1-rel-yes-neg-dm0123456-0089090-b80e337-s1888",
            "our-custom-anthropic": "FundamentalResearchLabs/ll8-4k-mu-unr-rt1-rel-yes-neg-dm0123456-0089090-b80e337-s1888",
            "openai-model": "FundamentalResearchLabs/ll8-4k-mu-unr-rt1-rel-yes-neg-dm0123456-0089090-b80e337-s1888", 
            "anthropic-model": "FundamentalResearchLabs/ll8-4k-mu-unr-rt1-rel-yes-neg-dm0123456-0089090-b80e337-s1888"
        }

        # Get the actual model name - check mapping first, then use as-is
        actual_model = model_mapping.get(model, model)

        # Find the model endpoint (static models, custom models, or via fundamental-photon API)
        model_endpoint = None
        model_info = None

        # Check static models first
        if actual_model in self.static_model_endpoints:
            model_endpoint = self.static_model_endpoints[actual_model]
            print(f"📍 Using static model: {actual_model}")

        # Check custom models from Supabase
        elif actual_model in self.custom_models:
            model_info = self.custom_models[actual_model]
            model_endpoint = model_info["endpoint"]
            print(f"📍 Using custom model from Supabase: {actual_model}")
            print(f"   Hardware: {model_info.get('hardware', 'Unknown')}")
            print(f"   Pricing: {model_info.get('pricing', 'Unknown')}")

        # Also check by model name (not just modelSource)
        else:
            for name, info in self.custom_models.items():
                if name == actual_model or info.get("modelSource") == actual_model:
                    model_endpoint = info["endpoint"]
                    model_info = info
                    print(f"📍 Found custom model by name match: {name} -> {actual_model}")
                    break

        # Determine which endpoint to use based on format preference
        use_anthropic_format = "anthropic" in model.lower() or kwargs.get("anthropic_format", False)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Determine which endpoint to use
                if model_endpoint:
                    # Direct model endpoint (static or custom from Supabase)
                    if "/chat/completions" in model_endpoint:
                        endpoint = model_endpoint
                    else:
                        endpoint = f"{model_endpoint}/chat/completions"

                    payload = {
                        "model": actual_model,
                        "messages": messages,
                        "max_tokens": kwargs.get("max_tokens", 150),
                        "temperature": kwargs.get("temperature", 0.7),
                        "stream": False
                    }
                    print(f"🔗 Calling direct endpoint: {endpoint}")

                elif use_anthropic_format:
                    # Use Anthropic Messages API format via fundamental-photon
                    endpoint = self.anthropic_endpoint
                    payload = {
                        "model": actual_model,
                        "messages": messages,
                        "max_tokens": kwargs.get("max_tokens", 150),
                        "temperature": kwargs.get("temperature", 0.7),
                        "stream": False
                    }
                    # Add system message if provided
                    if kwargs.get("system"):
                        payload["system"] = kwargs["system"]
                    print(f"🔗 Calling Anthropic format via: {endpoint}")

                else:
                    # Use OpenAI Chat Completions format via fundamental-photon
                    endpoint = self.openai_endpoint
                    payload = {
                        "model": actual_model,
                        "messages": messages,
                        "max_tokens": kwargs.get("max_tokens", 150),
                        "temperature": kwargs.get("temperature", 0.7),
                        "stream": False
                    }
                    print(f"🔗 Calling OpenAI format via: {endpoint}")

                print(f"🤖 Using model: {actual_model}")
                if model_info:
                    print(f"📊 Model info: {model_info.get('hardware', 'Unknown')} | {model_info.get('pricing', 'Unknown')}")

                response = await client.post(endpoint, json=payload)

                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"API Error: {response.status_code} - {response.text}")
                    # Return fallback response in expected format
                    if use_anthropic_format:
                        return {
                            "id": f"msg_fallback_{int(time.time())}",
                            "type": "message", 
                            "role": "assistant",
                            "content": [{"type": "text", "text": f"Fallback response (API error {response.status_code})"}],
                            "model": actual_model,
                            "stop_reason": "end_turn",
                            "usage": {"input_tokens": 10, "output_tokens": 20}
                        }
                    else:
                        return {
                            "id": f"chatcmpl_fallback_{int(time.time())}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": actual_model,
                            "choices": [{"message": {"role": "assistant", "content": f"Fallback response (API error {response.status_code})"}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
                        }

        except Exception as e:
            print(f"Custom model call error: {e}")
            # Return fallback response in expected format
            if use_anthropic_format:
                return {
                    "id": f"msg_fallback_{int(time.time())}",
                    "type": "message",
                    "role": "assistant", 
                    "content": [{"type": "text", "text": f"Fallback response (connection error: {str(e)})"}],
                    "model": actual_model,
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 20}
                }
            else:
                return {
                    "id": f"chatcmpl_fallback_{int(time.time())}",
                    "object": "chat.completion", 
                    "created": int(time.time()),
                    "model": actual_model,
                    "choices": [{"message": {"role": "assistant", "content": f"Fallback response (connection error: {str(e)})"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
                }

    def completion(self, *args, **kwargs) -> ModelResponse:
        """Synchronous completion - calls async version"""
        return asyncio.run(self.acompletion(*args, **kwargs))

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        """
        Async completion handler that supports both OpenAI and Anthropic formats
        """
        model = kwargs.get("model", "custom-model")
        messages = kwargs.get("messages", [])
        optional_params = kwargs.get("optional_params", {})

        # Call your custom model API and get full response
        api_response = await self._call_custom_model(messages, model, **optional_params)

        # Convert API response to LiteLLM ModelResponse format
        if "anthropic" in model.lower() or kwargs.get("anthropic_format", False):
            # Handle Anthropic format response
            if "content" in api_response and isinstance(api_response["content"], list):
                content = api_response["content"][0].get("text", "") if api_response["content"] else ""
            else:
                content = str(api_response.get("content", ""))

            usage_info = api_response.get("usage", {})
            return self._create_anthropic_response(content, model, usage_info)
        else:
            # Handle OpenAI format response
            if "choices" in api_response and api_response["choices"]:
                content = api_response["choices"][0].get("message", {}).get("content", "")
            else:
                content = str(api_response.get("content", ""))

            usage_info = api_response.get("usage", {})
            # Convert OpenAI usage format to our internal format
            if "prompt_tokens" in usage_info:
                usage_info = {
                    "prompt_tokens": usage_info.get("prompt_tokens", 10),
                    "completion_tokens": usage_info.get("completion_tokens", 50),
                    "total_tokens": usage_info.get("total_tokens", 60)
                }

            return self._create_openai_response(content, model, usage_info)

# Create instance of custom LLM
our_custom_llm = OurCustomLLM()
