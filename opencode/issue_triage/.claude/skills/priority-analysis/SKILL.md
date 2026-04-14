# Priority Analysis

## Overview
Determines issue priority (P0/P1/P2/P3) for PyTorch XPU issues using rule-based heuristics. Adds Priority and Priority Reason columns to Issues sheet.

**Important**: Issues about benchmark models (huggingface, timm, torchbench) are distinguished from customer custom models. Benchmark model issues are classified as P2, while custom model issues are P0.

## Workflow
1. Load benchmark models from GitHub (`huggingface_models_list.txt`, `timm_models_list.txt`, `torchbench_models_list.txt`)
2. Load `torch_xpu_ops_issues.xlsx` with Issues and Test Cases sheets
3. Find first blank column for Priority headers
4. For each issue, analyze: title, summary, test_module, labels, failed test count
5. Apply priority rules based on issue type
6. Add columns: Priority, Priority Reason

## Usage
```bash
cd /home/daisydeng/ai_for_validation/opencode/issue_triage/issue_analysis/priority
python3 run_priority.py [--excel EXCEL_FILE] [--limit N] [--force]
```

## Examples
```bash
# Run with default paths
python3 run_priority.py

# Limit to first 10 issues
python3 run_priority.py --limit 10

# Force overwrite existing values
python3 run_priority.py --force
```

## Output
- **Priority column**: P0/P1/P2/P3
- **Priority Reason column**: Detailed explanation

## Priority Rules

| Priority | Condition | Reason |
|----------|-----------|--------|
| **P0** | Build crash/segmentation/segfault | Build crash - critical blocking issue |
| **P0** | Regression (was passing, now fails) | Regression - passed before but failed now |
| **P0** | Custom model/application impact (not benchmark) | Impacts customer custom model/application |
| **P1** | Custom model E2E accuracy/functionality | E2E custom model accuracy/functionality issue |
| **P1** | UT with >20 failed test cases | UT with many failed test cases |
| **P2** | Benchmark model E2E issue (huggingface/timm/torchbench) | E2E benchmark model issue |
| **P2** | E2E benchmark accuracy issue | E2E benchmark accuracy issue |
| **P2** | E2E performance/slow/latency | E2E benchmark performance issue |
| **P2** | UT with few failures | UT issue with few failures |
| **P3** | Minor issues (no failures) | Minor issue - no test failures |

## Custom vs Benchmark Models

- **Benchmark models** (P2): Known models from:
  - `huggingface_models_list.txt`
  - `timm_models_list.txt`
  - `torchbench_models_list.txt`

- **Custom models** (P0/P1): Unknown/custom models, NOT in benchmark lists

This distinction ensures benchmark model issues are properly classified.

## Key Features
- **Rule-based only**: Fast processing, no LLM calls
- **Benchmark model detection**: Distinguishes benchmark from custom models
- **Safe column addition**: Only adds at first blank column (preserves existing data)
- **Force mode**: Can overwrite existing priority values with `--force`
- **Skip existing**: By default skips issues that already have priority

## Related Info
- Benchmark lists: https://github.com/intel/torch-xpu-ops/tree/main/.ci/benchmarks
- Input: Issues sheet (col A=issue_id) and Test Cases sheet
- Output: Priority columns at first blank column