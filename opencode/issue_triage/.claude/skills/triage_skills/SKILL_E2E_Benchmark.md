# E2E Benchmark Expertise Skill

This skill provides expertise for running, debugging, and analyzing E2E benchmark execution for torch-xpu-ops.

## Benchmark Overview

### Supported Benchmarks
| Suite | Path | Models | Script |
|-------|------|--------|--------|
| HuggingFace | `benchmarks/dynamo/huggingface.py` | 65 | `inductor_xpu_test.sh` |
| Timm | `benchmarks/dynamo/timm_models.py` | 88 | `inductor_xpu_test.sh` |
| TorchBench | `benchmarks/dynamo/torchbench.py` | 18 | `inductor_xpu_test.sh` |
| PT2E | `benchmarks/dynamo/pt2e.py` | Various | Separate action |

### Data Types Supported
- `float32` - Standard single precision
- `float16` - Half precision
- `bfloat16` - Brain float precision
- `amp_bf16` - AMP with bfloat16
- `amp_fp16` - AMP with float16

### Modes
- `inference` - Inference only
- `training` - Training with gradients

### Scenarios
- `accuracy` - Accuracy verification
- `performance` - Performance benchmarking

---

## Running E2E Benchmarks

### 1. Basic E2E Test Command
```bash
# Standard E2E test from torch-xpu-ops root
cd ~/pytorch

# Run HuggingFace inference accuracy
bash ./.github/scripts/inductor_xpu_test.sh \
    huggingface float32 inference accuracy \
    xpu 0 static 1 0 ""

# Run Timm models training accuracy
bash ./.github/scripts/inductor_xpu_test.sh \
    timm_models bfloat16 training accuracy \
    xpu 0 static 1 0 ""

# Run TorchBench performance test
bash ./.github/scripts/inductor_xpu_test.sh \
    torchbench float32 inference performance \
    xpu 0 static 1 0 ""
```

### 2. Run Specific Model
```bash
# Using MODEL_ONLY parameter
MODEL_ONLY="BertForMaskedLM" bash ./.github/scripts/inductor_xpu_test.sh \
    huggingface float32 inference accuracy \
    xpu 0 static 1 0 ""

# Using -k filter syntax
MODEL_ONLY="-k GPT2ForSequenceClassification" bash ./.github/scripts/inductor_xpu_test.sh \
    huggingface float32 inference accuracy \
    xpu 0 static 1 0 ""
```

### 3. Parallel Multi-GPU Execution
```bash
# Multi-GPU with numactl
NUMACTL_ARGS='ZE_AFFINITY_MASK=0 OMP_NUM_THREADS=12 numactl -l -C 0-11 ; \
             ZE_AFFINITY_MASK=1 OMP_NUM_THREADS=12 numactl -l -C 12-23' \
bash ./.github/scripts/inductor_xpu_test.sh \
    huggingface bfloat16 training accuracy \
    xpu 0 static 2 0 ""
```

### 4. Dynamic Shapes Testing
```bash
Shape_extra="--dynamic-shapes --dynamic-batch-only" bash ./.github/scripts/inductor_xpu_test.sh \
    huggingface float32 inference accuracy \
    xpu 0 dynamic 1 0 ""
```

---

## E2E Execution Parameters

### Parameter Reference
```bash
SUITE=${1:-huggingface}      # huggingface / timm_models / torchbench
DT=${2:-float32}             # float32 / float16 / amp / amp_bf16 / amp_fp16
MODE=${3:-inference}         # inference / training
SCENARIO=${4:-accuracy}       # accuracy / performance
DEVICE=${5:-xpu}             # xpu / cuda
CARD=${6:-0}                 # 0 / 1 / 2 / 3 (GPU device ID)
SHAPE=${7:-static}           # static / dynamic
NUM_SHARDS=${8}             # Number of parallel shards
SHARD_ID=${9}               # Current shard ID (0-indexed)
MODEL_ONLY=${10}            # Specific model name
```

### Command Flag Summary
| Flag | Description | Example Value |
|------|-------------|---------------|
| `--backend=inductor` | Use Inductor backend | inductor |
| `--cold-start-latency` | Enable cold start measurement | - |
| `--timeout=10800` | 3 hour timeout | 10800 seconds |
| `--disable-cudagraphs` | Disable CUDA graphs | - |
| `--training` | Training mode | required for training |
| `--amp-dtype bfloat16` | AMP with specific dtype | bfloat16 |
| `--dynamic-shapes` | Enable dynamic shapes | - |

---

## E2E Output Logs

### Log Locations
```
inductor_log/
├── huggingface/          # HuggingFace results
│   └── float32/
│       ├── inductor_huggingface_float32_inference_xpu_accuracy_all.log
│       ├── inductor_huggingface_float32_inference_xpu_accuracy_card0.csv
│       └── inductor_huggingface_float32_inference_xpu_accuracy_card0.log
├── timm_models/          # Timm results
│   └── bfloat16/
├── torchbench/            # TorchBench results
│   └── float32/
└── summary_accuracy.csv   # Aggregated accuracy summary
```

### Log Analysis
```python
def parse_accuracy_log(log_file: str) -> dict:
    """
    Parse E2E accuracy test log file.
    
    Returns:
        {
            "passed": [model_list],
            "failed": [model_list], 
            "errors": [error_dict],
            "summary": {
                "total": int,
                "passed": int,
                "failed": int,
                "pass_rate": float
            }
        }
    """
    
    results = {
        "passed": [],
        "failed": [],
        "errors": [],
        "summary": {"total": 0, "passed": 0, "failed": 0}
    }
    
    with open(log_file) as f:
        for line in f:
            if "PASSED" in line:
                # Extract model name
                model = extract_model_name(line)
                results["passed"].append(model)
            elif "FAILED" in line or "ERROR" in line:
                model = extract_model_name(line)
                results["failed"].append(model)
                results["errors"].append({"model": model, "line": line})
    
    results["summary"]["total"] = len(results["passed"]) + len(results["failed"])
    results["summary"]["passed"] = len(results["passed"])
    results["summary"]["failed"] = len(results["failed"])
    results["summary"]["pass_rate"] = (
        results["summary"]["passed"] / results["summary"]["total"] 
        if results["summary"]["total"] > 0 else 0
    )
    
    return results
```

---

## E2E Test Failure Analysis

### 1. Common E2E Error Patterns
```python
E2E_ERROR_PATTERNS = {
    "accuracy_mismatch": {
        "pattern": r"(accuracy|diff).*(failed|error|mismatch)",
        "severity": "P2",
        "action": "Check numerical precision and dtype handling"
    },
    "oom": {
        "pattern": r"(OutOfMemory|OOM|memory.*exceeded)",
        "severity": "P0",
        "action": "Reduce batch size, check memory allocation"
    },
    "timeout": {
        "pattern": r"(timeout|timed.out|taking.too.long)",
        "severity": "P2",
        "action": "Check kernel performance, optimize tiling"
    },
    "not_implemented": {
        "pattern": r"(not implemented|NotImplementedDispatch)",
        "severity": "P1",
        "action": "Implement missing XPU operator"
    },
    "runtime_error": {
        "pattern": r"(RuntimeError|SEGFAULT|Segmentation fault)",
        "severity": "P0",
        "action": "Debug kernel implementation"
    }
}
```

### 2. E2E Failure Debugger
```python
def debug_e2e_failure(model_name: str, suite: str, error_log: str) -> dict:
    """
    Debug E2E test failure.
    
    Steps:
    1. Identify failure type from error pattern
    2. Extract relevant stack trace
    3. Map to operator/implementation
    4. Generate fix suggestions
    """
    
    # Identify failure type
    failure_type = identify_failure_type(error_log)
    
    # Extract stack trace
    stack_trace = extract_stack_trace(error_log)
    
    # Find related operators
    operators = find_related_operators(stack_trace)
    
    # Generate debug report
    return {
        "model": model_name,
        "suite": suite,
        "failure_type": failure_type,
        "stack_trace": stack_trace,
        "related_operators": operators,
        "debug_commands": generate_debug_commands(model_name, suite),
        "fix_suggestions": generate_fix_suggestions(failure_type, operators)
    }
```

### 3. Quick E2E Test Reproduction
```python
def reproduce_e2e_failure(model: str, suite: str, dt: str, mode: str) -> str:
    """
    Generate command to reproduce E2E failure locally.
    """
    return f"""
    # Reproduce E2E failure for {model}
    cd ~/pytorch
    
    # Enable verbose logging
    export TORCH_LOGS="+dynamo,+inductor,+xpu"
    export PYTORCH_DEBUG_XPU_FALLBACK=1
    
    # Run single model test
    MODEL_ONLY={model} bash ./.github/scripts/inductor_xpu_test.sh \\
        {suite} {dt} {mode} accuracy \\
        xpu 0 static 1 0 ""
    
    # Check cached errors
    cat inductor_log/{suite}/{dt}/inductor_{suite}_{dt}_{mode}_xpu_accuracy_card0.log | grep -A 50 {model}
    """
```

---

## E2E Benchmark Skill Integration

### Integration with Triage Workflow
```python
# STEP 1: Check if issue is E2E related
def is_e2e_issue(issue_body: str) -> dict:
    """
    Check if issue relates to E2E benchmarks.
    """
    E2E_KEYWORDS = [
        "Benchmark", "huggingface", "timm", "torchbench",
        "dynamo", "inductor", "torch.compile",
        "bert_pytorch", "resnet18", "gemma", "llama",
        "vllm", "diffusers", "whisper"
    ]
    
    is_e2e = any(kw in issue_body.lower() for kw in E2E_KEYWORDS)
    
    suite = None
    model = None
    
    if is_e2e:
        # Identify which suite
        if "huggingface" in issue_body.lower():
            suite = "huggingface"
        elif "timm" in issue_body.lower():
            suite = "timm_models"
        elif "torchbench" in issue_body.lower():
            suite = "torchbench"
        
        # Extract specific model if mentioned
        model = extract_model_mention(issue_body)
    
    return {
        "is_e2e": is_e2e,
        "suite": suite,
        "model": model,
        "runnable": is_e2e
    }

# STEP 2: Run E2E verification test
def run_e2e_verification(issue_data: dict) -> dict:
    """
    Run E2E test to verify issue.
    """
    is_e2e_check = is_e2e_issue(issue_data["body"])
    
    if not is_e2e_check["runnable"]:
        return {"skipped": True, "reason": "Not E2E related"}
    
    # Determine test parameters from issue
    suite = is_e2e_check.get("suite", "huggingface")
    model = is_e2e_check.get("model", "")
    dt = extract_dtype_from_issue(issue_data["body"]) or "float32"
    mode = extract_mode_from_issue(issue_data["body"]) or "inference"
    
    # Run the test
    cmd = f"""
    cd ~/pytorch
    MODEL_ONLY="{model}" bash ./.github/scripts/inductor_xpu_test.sh \\
        {suite} {dt} {mode} accuracy \\
        xpu 0 static 1 0 ""
    """
    
    return {
        "executed": True,
        "suite": suite,
        "model": model,
        "command": cmd,
        "log_location": f"inductor_log/{suite}/{dt}/"
    }
```

---

## Benchmark Quick Reference

### Model List Files
```
.ci/benchmarks/huggingface_models_list.txt  # 65 models
.ci/benchmarks/timm_models_list.txt           # 88 models
.ci/benchmarks/torchbench_models_list.txt    # 18 models
benchmarks/dynamo/huggingface_models_list.txt  # Synced for PR
```

### Key Models Reference

| Suite | Key Models | Notes |
|-------|------------|-------|
| HuggingFace | BERT, GPT2, T5, LLaMA, Mistral, Gemma, Qwen | NLP models |
| Timm | ResNet, ViT, EfficientNet, Swin | Vision models |
| TorchBench | BERT_pytorch, ResNet18/50, VGG | Core torchbench |

### Performance Baselines
```python
# Expected performance thresholds (cards/sec)
PERFORMANCE_THRESHOLDS = {
    "bert_pytorch": {"float32": 100, "bfloat16": 200},
    "resnet50": {"float32": 500, "bfloat16": 800},
    "gpt_j": {"float32": 10, "bfloat16": 20},
    # Add more as available
}
```

---

## Skill Metadata

- **Version**: 1.0.0
- **Created**: 2026-04-20
- **Related Skills**: SKILL_Triage_Logic.md, SKILL_Priority_Analysis.md
- **Source**: `.github/workflows/_linux_e2e.yml`, `.github/scripts/inductor_xpu_test.sh`