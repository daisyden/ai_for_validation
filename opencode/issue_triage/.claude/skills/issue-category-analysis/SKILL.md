# Issue Category Analysis

## Overview
Determines issue category using LLM-based analysis with Qwen3-32B model. Populates Category (S/19) and Category Reason (T/20) columns in Issues sheet.

## Usage
```bash
cd /home/daisydeng/ai_for_validation/opencode/issue_triage/issue_analysis/category
python3 issue_category_analyzer.py <excel_file>
python3 issue_category_analyzer.py <excel_file> --issues "3246,3243"
python3 issue_category_analyzer.py <excel_file> --force
```

## Categories
1. Distributed - distributed, XCCL, NCCL, Gloo, ProcessGroup, DDP, FSDP
2. TorchAO - torchao, quantize_, int4_weight_only, int8_dynamic_activation, fp8
3. PT2E - torch.export(), Dynamo, fake_tensor, ExportedProgram, AOT
4. Flash Attention/Transformer - flash_attention, SDPA, attention mask
5. Sparse - sparse tensor, CSR, CSC, COO, torch.sparse
6. Inductor/Compilation - torch.compile(), Inductor, Triton, codegen
7. Torch Runtime - CUDA runtime, OOM, device kernel launch, stream sync
8. Torch Operations - aten::, native::, custom op, operator dispatch
9. Dtype/Precision - float16, bfloat16, mixed precision, autocast
10. Feature Not Supported - unimplemented operator, missing kernel
11. Skip/No Test Exists - missing tests, CI infrastructure problems
12. Others - None of the above

## Key Features
- **LLM Analysis**: Uses Qwen3-32B for accurate categorization
- **Detailed Reasons**: Returns category + detailed explanation
- **Progress Logging**: Logs to result/pipeline.log
- **Auto-save**: Saves every 10 issues

## Output Columns
| Column | Header | Description |
|--------|--------|-------------|
| 19 | Category | Category number and name |
| 20 | Category Reason | Detailed explanation |

## Notes
- Skips issues that already have categories (use --force to re-process)
- Filters out 'skipped'/'skipped_windows' labels from classification
- Matches test cases from Test Cases sheet for context