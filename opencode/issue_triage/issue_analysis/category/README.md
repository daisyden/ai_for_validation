# Category Analysis Module

## Overview

This module contains LLM-based and rule-based category determination logic for PyTorch XPU issue triaging.

## Usage

```python
from issue_analysis.catetory.category_analyzer import (
    determine_category,
    determine_category_llm
)

# Rule-based category (fast)
category = determine_category(
    title="Issue title",
    summary="Issue summary",
    test_cases_str="...",
    traceback="Error traceback",
    test_module="ut",
    labels=["bug"]
)
# Returns: "6 - Inductor / Compilation Related"

# LLM-based category + reason (accurate)
category, reason = determine_category_llm(
    title="Issue title",
    summary="Issue summary",
    test_cases_info=[{"test_case": "test_xxx", "error_msg": "..."}],
    test_module="ut",
    labels="bug"
)
# Returns: ("6 - Inductor / Compilation", "Detailed reason...")
```

## Categories

| ID | Category | Keywords |
|----|----------|----------|
| 1 | Distributed | distributed, ProcessGroup, DDP, FSDP, all_reduce, all_gather |
| 2 | TorchAO | torchao, quantization, int8, int4, fp8, Adam8bit |
| 3 | PT2E | torch.export, Dynamo, fake_tensor, ExportedProgram |
| 4 | Flash Attention/Transformer | flash_attention, SDPA, attention mask, transformer |
| 5 | Sparse Operations | sparse tensor, CSR, CSC, COO, torch.sparse |
| 6 | Inductor/Compilation | torch.compile, Inductor, Triton, codegen, FX graph |
| 7 | Dtype/Precision | dtype mismatch, bf16, fp16, precision, NaN/inf |
| 8 | Others | None of the above |

## Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `determine_category()` | Rule-based keyword matching | Category string |
| `determine_category_llm()` | LLM-based classification via Qwen3-32B | (category, reason) |

## Classification Rules

1. **Distributed** checked first (clear module identification)
2. **TorchAO** - quantization, int8/int4/fp8, torchao keywords
3. **PT2E** - torch.export, Dynamo, fake tensors
4. **Flash Attention/Transformer** - SDPA, attention, transformer
5. **Sparse** - sparse tensor, CSR/CSC/COO
6. **Inductor/Compilation** - compile, Triton, codegen
7. **Dtype/Precision** - dtype, bf16, fp16, precision issues
8. **Others** - default if no keywords match

## LLM Configuration

Uses Qwen3-32B via internal API:
- Endpoint: `http://10.239.15.43/v1/chat/completions`
- API Key: From `OPENCODE_API_KEY` env var
- Model: `Qwen3-32B`
- Temperature: 0.0
- Max tokens: 400

## Output Format

```
Category Name | detailed_reason
```

Example:
```
6 - Inductor/Compilation | The aten.matmul kernel compilation fails in Inductor due to missing Triton implementation for XPU backend. The fx graph capture encounters unsupported loop pattern.
```

## Constants

```python
CATEGORY_DISTRIBUTED = "1 - Distributed"
CATEGORY_TORCHAO = "2 - TorchAO"
CATEGORY_PT2E = "3 - PT2E"
CATEGORY_FLASH_ATTENTION = "4 - Flash Attention / Transformer Related"
CATEGORY_SPARSE = "5 - Sparse Operations Related"
CATEGORY_INDUCTOR = "6 - Inductor / Compilation Related"
CATEGORY_DTYPE_PRECISION = "7 - Dtype / Precision Related"
CATEGORY_OTHERS = "8 - Others"
```