# Issue Category Analysis Skill

## Overview
This skill provides automatic categorization of torch-xpu-ops issues based on deep analysis of issue description, error logs, running output, and code investigation.

**Authoritative Category column values** (used in tracking Excel) are the 8 buckets below. Treat these as an enum — no other values are permitted in the Excel "Category" column.

Older revisions of this doc listed 9 categories including "PT2E", "Inductor/Compilation", "Feature Gap", and "Build/Compilation"/"Test Infrastructure"/"Numerical Accuracy"/"CI/CD" as separate categories. Those names are **deprecated as Category values** and now map into the 8 buckets below. They remain valid as *sub-type descriptions* in `root_cause` / `fix_approach` prose.

---

## Canonical Category Taxonomy (8 buckets)

### Decision Priority Order

When an issue matches multiple categories, apply in this order (first match wins):

1. `Distributed` — anything tagged `[distributed]` or using XCCL/ProcessGroup/DDP/FSDP/DTensor/symm_mem
2. `Flash Attention` — SDPA / flash / efficient attention (unless already claimed by Distributed)
3. `Inductor` — torch.compile / Dynamo / AOTAutograd / Triton codegen / benchmark via inductor
4. `TorchAO` — quantization (int4/int8/fp8/PT2E quant/torchao)
5. `Sparse` — sparse tensor formats/ops
6. `Torch Operations` — ATen/native operator issues (including numerical accuracy on a specific op)
7. `Torch Runtime` — torch.xpu.* runtime, memory/OOM, profiler, RNG, streams, IPC
8. `Others` — CI/infra/tracking/build/doc/test-harness/meta — the catch-all

### 1. `Distributed`

**Trigger**: `[distributed]` title tag OR any of ProcessGroup, XCCL, NCCL, Gloo, DDP, FSDP, DTensor, symm_mem, fused_all_gather_*, fused_matmul_reduce_scatter, collective ops.

**Examples** (from production):
- #1571 `PREMUL_SUM with XCCL` → Distributed
- #1556 `[distributed] _scaled_dot_product_fused_attention DTensor` → Distributed (overrides Flash Attention per priority order)
- #1574 `_grouped_mm` → `Torch Operations` (no distributed tag, pure op gap)

**Related files**: `torch/distributed/`, `torch/csrc/distributed/`, `torch-xpu-ops/src/comm/`, `test/distributed/`.

### 2. `Flash Attention`

**Trigger**: scaled_dot_product_attention, SDPA, `_flash_attention`, `_efficient_attention`, `mha_fwd` / `mha_bwd`, `MultiheadAttention`, TransformerEncoderLayer attention path.

**Related files**: `aten/src/ATen/native/transformers/`, `torch-xpu-ops/src/ATen/native/transformers/sycl/AttentionKernels.cpp`, `mha_fwd.cpp`, `mha_bwd.cpp`, `FlashAttention.cpp`, `AttentionEfficient.cpp`.

### 3. `Inductor`

**Trigger**: torch.compile, Dynamo, AOTAutograd, `_dynamo`, `_inductor`, Triton codegen, FakeTensor/MetadataMismatch, ExportedProgram, AOTInductor, torchbench model failures via inductor path.

**Examples**:
- #2128 speech_transformer accuracy under inductor → Inductor
- #1877 squeezenet1_1 / functorch_dp_cifar10 fail_accuracy under inductor → Inductor
- #1963 FakeTensor MetadataMismatchError → Inductor
- #1505 timm models fail_accuracy on ARC-WSL → Inductor (benchmark is inductor path)

### 4. `TorchAO`

**Trigger**: torchao imports, `quantize_`, `int4_weight_only`, `int8_dynamic_activation`, fp8, NF4, PT2E int8/int4 quantization, `_weight_int4pack_mm`, QLinear/QConv, fake quant, quant observer.

### 5. `Sparse`

**Trigger**: `torch.sparse`, BSR/CSR/CSC/COO formats, `sparse_csr_tensor`, `sparse_coo_tensor`, `SparseTensorImpl`, `SparseCsrTensorMath*`, nnz, sparse_dim.

### 6. `Torch Operations`

**Trigger**: ATen/native operator issue NOT covered above — conv, linalg (non-SDPA), reduce, batchnorm, indexing, pointwise, distributions, pooling, loss, range, upsample, foreach, scatter/gather, compare, binary ops, etc.

**Also covers**: numerical accuracy bugs on a specific operator; missing XPU op implementations (CPU fallback); op-specific precision tuning.

**Examples**:
- #2022 std/var/sum accuracy → Torch Operations
- #1951 BatchNorm test_out → Torch Operations
- #1936 `linalg.cholesky` not implemented → Torch Operations
- #1893 addmv/mv precision → Torch Operations

### 7. `Torch Runtime`

**Trigger**: torch.xpu.* device-management APIs, memory allocation/OOM, profiler, RNG helpers, streams/events, IPC / share_memory, device-query APIs, driver error-message surface, SYCL context init.

**Examples**:
- #2089 torch.xpu.is_available non-initializing probe → Torch Runtime
- #1986 / #1727 `torch.xpu._sleep` missing → Torch Runtime (unless `[distributed]` tag)
- #1784 XPU profiler unreliable → Torch Runtime
- #1324 UR Error on OOM → Torch Runtime
- #1678 `model.share_memory()` (IPC) → Torch Runtime

### 8. `Others`

**Trigger**: catch-all for anything not matching above — CI/CD trackers, release checklists, build-warning housekeeping, path-coverage meta-issues, documentation/term cleanup, upstream benchmark harness bugs, test-infra meta-tracking (`[DONT CLOSE]` ledgers), upstream HuggingFace bugs, ATen utility refactors.

**Examples**:
- #2127 path coverage enhancement → Others
- #2063 out-of-date doc terms → Others
- #1866 vision_maskrcnn benchmark harness → Others (CUDA also fails, not XPU-specific)
- #1729 release validation checklist → Others
- #1159 Deberta HF upstream bug → Others

---

## Quick Mapping: Old Categories → Canonical

| Old label (deprecated as Category value) | Canonical target |
|---|---|
| Test Infrastructure | `Others` (unless it's testing a specific op → that op's bucket) |
| Operator Implementation | `Torch Operations` |
| Build/Compilation | `Others` |
| Performance | same bucket as the affected component (e.g., kernel perf → `Torch Operations`, profiler → `Torch Runtime`) |
| Numerical Accuracy | same bucket as the affected op (usually `Torch Operations`) |
| Feature Gap | same bucket as the feature (Distributed collective gap → `Distributed`; missing linalg op → `Torch Operations`; missing runtime API → `Torch Runtime`) |
| Documentation | `Others` |
| CI/CD | `Others` |
| Environment/Driver | `Torch Runtime` |
| PT2E | `Inductor` (or `TorchAO` if it's PT2E quantization) |
| Inductor/Compilation | `Inductor` |

---

## Category Analysis Tools

### 1. Multi-Keyword Categorizer
```python
def categorize_issue(issue_body: str, error_log: str = "", code_analysis: str = "") -> dict:
    """
    Analyze issue and return category with confidence.
    
    Args:
        issue_body: GitHub issue description
        error_log: Execution error/stack trace
        code_analysis: Code investigation findings
    
    Returns:
        {
            "primary_category": str,
            "secondary_categories": [str],
            "confidence": float,
            "matched_keywords": dict,
            "evidence": str
        }
    """
    
    CATEGORIES = {
        "Distributed": {
            "keywords": [
                "distributed", "XCCL", "NCCL", "Gloo", "ProcessGroup",
                "DDP", "FSDP", "all_reduce", "all_gather", "collective"
            ],
            "weight": 1.0
        },
        "TorchAO": {
            "keywords": [
                "torchao", "quantize_", "int4_weight_only", "int8_dynamic_activation",
                "fp8", "weight_only", "NF4", "int_quant", "uint4"
            ],
            "weight": 1.0
        },
        "PT2E": {
            "keywords": [
                "torch.export", "Dynamo", "fake_tensor", "ExportedProgram",
                "AOT", "dynamo", "export", "_inductor"
            ],
            "weight": 1.0
        },
        "Flash Attention": {
            "keywords": [
                "flash_attention", "sdpa", "scaled_dot_product_attention",
                "attention", "MultiheadAttention", "flashattention"
            ],
            "weight": 1.0
        },
        "Sparse": {
            "keywords": [
                "sparse", "BSR", "CSR", "CSC", "COO", "torch.sparse",
                "sparse_dim", "nnz", "sparse_quantized"
            ],
            "weight": 1.0
        },
        "Inductor/Compilation": {
            "keywords": [
                "torch.compile", "Inductor", "Triton", "codegen",
                "inductor", "TritonKernel", "AOTInductor"
            ],
            "weight": 1.0
        },
        "Torch Runtime": {
            "keywords": [
                "runtime", "OOM", "out of memory", "kernel launch",
                "stream sync", "memory allocation", "page fault"
            ],
            "weight": 1.0
        },
        "Torch Operations": {
            "keywords": [
                "aten::", "native::", "custom op", "operator dispatch",
                "DispatchKey", "NotImplemented"
            ],
            "weight": 1.0
        }
    }
    
    # Combine all text sources
    all_text = f"{issue_body} {error_log} {code_analysis}".lower()
    
    # Score each category
    scores = {}
    for cat_name, cat_info in CATEGORIES.items():
        score = sum(1 for kw in cat_info["keywords"] if kw.lower() in all_text)
        if score > 0:
            scores[cat_name] = {
                "score": score,
                "matched": [kw for kw in cat_info["keywords"] if kw.lower() in all_text]
            }
    
    # Sort by score
    sorted_cats = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
    
    if not sorted_cats:
        return {"primary_category": "Others", "secondary_categories": [], "confidence": 0.0}
    
    return {
        "primary_category": sorted_cats[0][0],
        "secondary_categories": [c[0] for c in sorted_cats[1:3] if c[1]["score"] > 0],
        "confidence": min(sorted_cats[0][1]["score"] / 3.0, 1.0),  # Normalize to 0-1
        "matched_keywords": sorted_cats[0][1]["matched"],
        "evidence": f"Matched {len(sorted_cats[0][1]['matched'])} keywords"
    }
```

### 2. Code Pattern Categorizer
```python
def categorize_by_code_pattern(code_text: str) -> dict:
    """
    Categorize based on code patterns found during investigation.
    
    Uses file paths and function signatures to identify category.
    """
    
    patterns = {
        "Distributed": [
            r"distributed/.*ProcessGroup",
            r"torch/distributed",
            r"collective\(",
            r"all_reduce\(",
        ],
        "TorchAO": [
            r"torch/ao/.*quant",
            r"torchao",
            r"weight_only.*quant",
        ],
        "PT2E": [
            r"torch/export",
            r"_dynamo",
            r"ExportedProgram",
            r"fake_tensor",
        ],
        "Flash Attention": [
            r"flash_attention",
            r"scaled_dot_product_attention",
            r"AttentionKernels",
            r"MultiheadAttention",
        ],
        "Sparse": [
            r"torch/sparse",
            r"sparse.*tensor",
            r"BSR|CSR|COO",
        ],
        "Inductor/Compilation": [
            r"torch/compile",
            r"inductor",
            r"triton.*kernel",
            r"codegen",
        ],
        "Torch Runtime": [
            r"kernel.*launch",
            r"memory.*alloc",
            r"stream.*sync",
            r"drm_neo",
        ],
        "Torch Operations": [
            r"aten::\w+",
            r"native::\w+",
            r"DispatchKey::XPU",
        ]
    }
    
    import re
    scores = {}
    for cat, pats in patterns.items():
        matches = []
        for p in pats:
            found = re.findall(p, code_text, re.IGNORECASE)
            matches.extend(found)
        if matches:
            scores[cat] = {"matches": matches, "count": len(matches)}
    
    return scores
```

### 3. Stack Trace Categorizer
```python
def categorize_by_stack_trace(stack: str) -> dict:
    """
    Categorize based on stack trace patterns.
    
    Prioritizes evidence from actual code paths executed.
    """
    
    # Common stack trace patterns
    stack_categories = {
        "Flash Attention": [
            "scaled_dot_product",
            "AttentionKernels",
            "FlashAttention",
        ],
        "Distributed": [
            "ProcessGroup",
            "NCCL_COMM",
            "XCCL",
            "DistAutogradEngine",
        ],
        "Sparse": [
            "SparseTensorImpl",
            "SparseCsrCPU",
            "SparseCOO",
        ],
        "Torch Runtime": [
            "drm_neo.cpp",
            "os_interface",
            "PageFault",
            "SYCL runtime",
        ],
        "Inductor/Compilation": [
            "TritonKernel",
            "inductor",
            "codegen",
        ],
        "Torch Operations": [
            "RegisterXPU",
            "XPUFallback",
            "atennative",
        ]
    }
    
    results = {}
    for cat, patterns in stack_categories.items():
        found = []
        for pat in patterns:
            if pat.lower() in stack.lower():
                found.append(pat)
        if found:
            results[cat] = found
    
    # Primary category = most pattern matches
    if results:
        primary = max(results.items(), key=lambda x: len(x[1]))
        return {
            "primary": primary[0],
            "all_matches": results,
            "confidence": len(primary[1]) / 5.0
        }
    
    return {"primary": "Others", "all_matches": {}, "confidence": 0.0}
```

---

## Category Analysis Workflow

### Step 1: Text Acquisition
```python
# Collect all text for categorization
all_text_sources = {
    "issue_body": issue_data["body"],
    "error_log": execution_result.get("stderr", ""),
    "stack_trace": extract_stack_trace(execution_result),
    "code_analysis": explore_findings.get("code_paths", ""),
    "comments": issue_data.get("comments", [])
}
```

### Step 2: Keyword Match
```python
# Run multi-keyword categorizer
category_result = categorize_issue(
    issue_body=all_text_sources["issue_body"],
    error_log=all_text_sources["error_log"],
    code_analysis=all_text_sources["code_analysis"]
)
```

### Step 3: Stack Trace Analysis
```python
# Analyze stack trace for category evidence
stack_result = categorize_by_stack_trace(all_text_sources["stack_trace"])
```

### Step 4: Code Pattern Analysis
```python
# Analyze code patterns for category evidence
code_result = categorize_by_code_pattern(all_text_sources["code_analysis"])
```

### Step 5: Combine Results
```python
def combine_category_analysis(primary_result, stack_result, code_result) -> dict:
    """
    Combine multiple analysis methods for robust categorization.
    """
    
    # Weight categories by source
    weights = {
        "stack_trace": 0.4,      # Most reliable
        "code_pattern": 0.3,     # Implementation evidence
        "keyword": 0.3           # Text analysis
    }
    
    combined_scores = {}
    
    for cat in ["Distributed", "TorchAO", "PT2E", "Flash Attention", 
                "Sparse", "Inductor/Compilation", "Torch Runtime", "Torch Operations"]:
        score = 0
        details = []
        
        # Stack trace contribution
        if cat in stack_result.get("all_matches", {}):
            score += weights["stack_trace"]
            details.append(f"Stack: {len(stack_result['all_matches'][cat])} matches")
        
        # Code pattern contribution
        if cat in code_result:
            score += weights["code_pattern"]
            details.append(f"Code: {code_result[cat]['count']} patterns")
        
        # Keyword contribution
        if cat in primary_result.get("matched_keywords", {}):
            score += weights["keyword"] * min(len(primary_result[cat]) / 2.0, 1.0)
            details.append(f"Keywords: {primary_result[cat]}")
        
        if score > 0:
            combined_scores[cat] = {"score": score, "details": details}
    
    # Final categorization
    if combined_scores:
        sorted_cats = sorted(combined_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        return {
            "primary_category": sorted_cats[0][0],
            "confidence": min(sorted_cats[0][1]["score"], 1.0),
            "all_categories": sorted_cats,
            "evidence": sorted_cats[0][1]["details"]
        }
    
    return {"primary_category": "Others", "confidence": 0.0}
```

---

## Category-Specific Investigation Templates

### Distributed Investigation
```python
task(description="distributed_issue_investigation",
     prompt="Investigate Distributed/XCCL issue:\n\n1. Check ProcessGroup implementation\n2. Identify collective communication patterns\n3. Find XCCL/NCCL wrapper code\n4. Look for distributed test cases\n\nSearch: ~/pytorch/third_party/torch-xpu-ops/src/comm/")
```

### Flash Attention Investigation
```python
task(description="attention_issue_investigation",
     prompt="Investigate Flash Attention/SDPA issue:\n\n1. Find Attention kernel implementations\n2. Check SDPA kernel selection\n3. Identify Triton/triton backend usage\n4. Look for memory requirements analysis\n\nSearch: ~/pytorch/third_party/torch-xpu-ops/src/ATen/native/transformers/sycl/")
```

### Torch Runtime Investigation
```python
task(description="runtime_issue_investigation",
     prompt="Investigate Torch Runtime issue:\n\n1. Check memory allocation patterns\n2. Identify device/stream synchronization\n3. Look for IGC/driver interaction\n4. Examine SYCL context management\n\nSearch: ~/pytorch/third_party/torch-xpu-ops/src/comm/")
```

---

## Usage in Triage Report

### Triage Report Category Section
```markdown
## Category Analysis

### Primary Category
**Flash Attention/Transformer** (Confidence: 92%)

### Evidence
- Keywords: "scaled_dot_product_attention", "sdpa", "attention"
- Stack Trace: "AttentionKernels.cpp", "scaled_dot_product"
- Code Patterns: "aten::_scaled_dot_product_efficient_attention"

### Related Categories
- Torch Runtime (15% - memory page fault secondary)
- Inductor/Compilation (8% - Triton kernel involvement)
```

---

## Skill Metadata

- **Version**: 1.0.0
- **Created**: 2026-04-20
- **Requires**: Issue text, execution result, explore findings
- **Related Skills**: SKILL_Triage_Logic.md, SKILL_Deep_Analysis_Patterns.md