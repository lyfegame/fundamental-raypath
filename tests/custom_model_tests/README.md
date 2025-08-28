# Custom Model Tests

This directory contains comprehensive tests for custom model endpoints and ARC-AGI-2 benchmarking, providing validation for both OpenAI and Anthropic format endpoints through LiteLLM proxy integration.

## Test Files

### `test_custom_model.py`
Tests custom model endpoints via LiteLLM proxy, covering both OpenAI and Anthropic format endpoints.

**Features:**
- ✅ Proxy health checks and endpoint validation
- ✅ Model discovery and listing (6 models configured)
- ✅ OpenAI format endpoint testing (`our-custom-openai`)
- ✅ Anthropic format endpoint testing (`our-custom-anthropic`)
- ✅ Performance metrics: token usage, response times
- ✅ Comprehensive JSON result generation with detailed analytics

**Usage:**
```bash
# Ensure proxy is running first
python start_proxy.py &

# Run custom model tests
python tests/custom_model_tests/test_custom_model.py
```

**Expected Results**: 6/6 tests passing, 100% success rate
**Output:** `custom_model_test_results.json`

### `test_preview_models.py`
Advanced testing for deployed custom models, providing comprehensive evaluation of model capabilities with real-world scenarios.

**Features:**
- ✅ Auto-discovery of deployed models from Supabase (15+ models detected)
- ✅ Multi-scenario testing: conversation, coding, pattern recognition
- ✅ Dual-format validation: OpenAI and Anthropic endpoints
- ✅ ARC-AGI-2 style reasoning challenges with pattern detection
- ✅ Performance analysis: token efficiency, response quality
- ✅ Rate-limited testing (3 models max) for production stability

**Usage:**
```bash
# Auto-discovers models from Supabase, no proxy required
python tests/custom_model_tests/test_preview_models.py
```

**Expected Results**: 3 models tested, pattern recognition validated
**Output:** `preview_models_test_results.json`

## Prerequisites & Setup

### **For `test_custom_model.py`**:
1. ✅ LiteLLM proxy server running (`python start_proxy.py`)
2. ✅ Custom model handler configured (`custom_model_handler.py`)
3. ✅ Master key authentication: `Bearer sk-1234`
4. ✅ Dual-format endpoints configured in `custom_config.yaml`

### **For `test_preview_models.py`**:
1. ✅ Valid Supabase connection for model discovery
2. ✅ Custom model handler with Supabase integration
3. ✅ Deployed custom models with endpoint URLs
4. ✅ No proxy required (direct model testing)

## Result Files & Output

**Auto-generated test results** (properly git-ignored):
- `custom_model_test_results.json` - Custom endpoint test results  
- `preview_models_test_results.json` - Preview model test results
- `*_benchmark_results.json` - Benchmark execution results

**Example Success Output**:
```
✅ 6/6 tests passing (100% success rate)
✅ OpenAI format: 68-129 tokens, 0.53-1.01s response time
✅ Anthropic format: 60 tokens avg, 0.60-1.20s response time  
✅ 15 models discovered via Supabase
```

## Integration & Workflow

**Part of Complete Benchmark System**:
```bash
setup_arc_agi.py          # ⚙️  Environment setup & ARC-AGI-2 dataset download
start_proxy.py            # 🚀 LiteLLM proxy server startup  
test_custom_model.py      # 🧪 Custom endpoint validation
test_preview_models.py    # 🔍 Model discovery & testing
arc_benchmark.py          # 📊 ARC-AGI-2 benchmark execution
run_complete_benchmark.py # 🎯 Complete workflow orchestration
```

**Quick Test Execution**:
```bash
# Individual test runs
python tests/custom_model_tests/test_custom_model.py
python tests/custom_model_tests/test_preview_models.py

# Complete benchmark
python run_complete_benchmark.py
```
