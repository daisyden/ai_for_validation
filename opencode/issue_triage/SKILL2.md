# Intel/torch-xpu-ops Issue Triage Pipeline

## Overview

This skill provides a complete pipeline for triaging intel/torch-xpu-ops GitHub issues. It collects issue data, extracts test cases, gets CI results, and generates analysis reports with priority, category, and root cause classification.

## Pipeline Steps

---

### Step 1: Issue Collection & Excel Generation
**Location**: `issue_analysis/issue_basic_info_extraction/`

Collects all open GitHub issues from intel/torch-xpu-ops repository and generates an Excel file with issue details and test cases.

**Scripts**:
- `generate_excel.py` - Fetches issues via GitHub API, parses test cases from issue body

**Usage**:
```bash
cd ~/ai_for_validation/opencode/issue_triage/issue_analysis/issue_basic_info_extraction
python3 generate_excel.py

# Process specific issues only
python3 generate_excel.py --issues "3246,3243"
```

**Output**: `$RESULT_DIR/torch_xpu_ops_issues.xlsx` with 3 sheets:
- **Issues**: Basic issue info (ID, Title, Status, Assignee, Labels, Type, Module, Test Module, etc.)
- **Test Cases**: Test case details (Test Type, File, Class, Case)
- **E2E Test Cases**: Benchmark test information (huggingface/timm/torchbench)

---

### Step 2: PR Extraction
**Location**: `issue_analysis/pr-extraction/`

Extracts PR references from GitHub issue comments with fix-related keywords and fetches PR status.

**Scripts**:
- `pr_extraction.py` - Extracts PRs from comments, checks merged/closed status

**Usage**:
```bash
cd ~/ai_for_validation/opencode/issue_triage/issue_analysis/pr-extraction
python3 pr_extraction.py $RESULT_DIR/torch_xpu_ops_issues.xlsx

# Process specific issues only
python3 pr_extraction.py $RESULT_DIR/torch_xpu_ops_issues.xlsx --issues "3246,3243"
```

**Output**: Updates Issues sheet with PR columns:
- PR: Links to PRs that fixed this issue
- PR Owner: PR author
- PR Status: merged/closed/open

---

### Step 3: Category Analysis
**Location**: `issue_analysis/category/`

Determines issue category for ALL issues in the Excel file using LLM-based analysis with Qwen3-32B model. Results are saved directly to the **Issues sheet** (Category column S/19 and Category Reason column T/20). Progress and LLM call results are logged to `result/pipeline.log`.

**Scripts**:
- `category_analyzer.py` - LLM-based category determination with progress logging

**Categories**:
| # | Category | Description |
|---|----------|-------------|
| 1 | Distributed | distributed, NCCL, Gloo, ProcessGroup, DDP, FSDP |
| 2 | TorchAO | torchao, quantization, int4/int8/fp8, optimizer |
| 3 | PT2E | torch.export, Dynamo, ExportedProgram, fake_tensor |
| 4 | Flash Attention/Transformer | flash_attention, SDPA, attention mask |
| 5 | Sparse Operations | sparse tensor, CSR, CSC, COO |
| 6 | Inductor/Compilation | torch.compile, Inductor, Triton, codegen |
| 7 | Torch Runtime | CUDA runtime, memory allocation, sync |
| 8 | Torch Operations | aten::, native::, custom op, kernel dispatch |
| 9 | Dtype/Precision | dtype mismatch, float16, bfloat16, amp |
| 10 | Feature Not Supported | unimplemented operator, missing kernel |
| 11 | Skip/No Test Exists | test skipped, missing decorator |
| 12 | Others | None of the above |

**Usage**:
```bash
cd ~/ai_for_validation/opencode/issue_triage/issue_analysis/category
python3 category_analyzer.py $RESULT_DIR/torch_xpu_ops_issues.xlsx

# Process specific issues only
python3 category_analyzer.py $RESULT_DIR/torch_xpu_ops_issues.xlsx --issues "3246,3243"

# Monitor progress in real-time
tail -f $RESULT_DIR/pipeline.log | grep Category
```

**Output to Issues Sheet**:
| Column | Field | Description |
|--------|-------|-------------|
| S (19) | **Category** | Category number and name, e.g., "1 - Distributed" |
| T (20) | **Category Reason** | Detailed explanation for classification |

**Logging to `result/pipeline.log`**:

The script logs detailed status, progress, and LLM call results:

```bash
# Start of processing
[2026-04-11 10:30:45] [Category] Loading Excel: /home/daisydeng/ai_for_validation/opencode/issue_triage/result/torch_xpu_ops_issues.xlsx
[2026-04-11 10:30:45] [Category] Total issues to process: 384

# Each issue analysis
[2026-04-11 10:30:46] [Category] Issue 3246: Starting LLM analysis... (1/384)
[2026-04-11 10:30:48] [Category] Issue 3246: SUCCESS | Category: 9 - Dtype/Precision | Time: 2.3s
[2026-04-11 10:30:48] [Category] Issue 3246: Reason: The test reveals a dtype mismatch in aten.add operation when processing bf16 tensors on XPU device

# Progress updates (every 10 issues)
[2026-04-11 10:31:30] [Category] Progress: 10/384 (2.6%)
[2026-04-11 10:32:15] [Category] Progress: 20/384 (5.2%)
...
[2026-04-11 11:30:00] [Category] Progress: 384/384 (100.0%)

# Error handling
[2026-04-11 10:30:50] [Category] Issue 3247: ERROR | API Error: 500 | Time: 5.2s

# Final summary
[2026-04-11 11:30:05] [Category] ===== Category Analysis Summary =====
[2026-04-11 11:30:05] [Category] Total issues: 384
[2026-04-11 11:30:05] [Category] Category assigned: 382
[2026-04-11 11:30:05] [Category] Errors: 2
[2026-04-11 11:30:05] [Category] Output file: /home/daisydeng/ai_for_validation/opencode/issue_triage/result/torch_xpu_ops_issues.xlsx
[2026-04-11 11:30:05] [Category] Category column: S (19)
[2026-04-11 11:30:05] [Category] Category Reason column: T (20)
[2026-04-11 11:30:05] [Category] Log file: /home/daisydeng/ai_for_validation/opencode/issue_triage/result/pipeline.log
```

**Monitoring Commands**:
```bash
# Watch real-time progress
tail -f $RESULT_DIR/pipeline.log | grep Category

# Count processed issues
grep "SUCCESS |" $RESULT_DIR/pipeline.log | wc -l

# View category distribution
grep "SUCCESS | Category:" $RESULT_DIR/pipeline.log | cut -d'|' -f2 | sort | uniq -c

# Check for errors
grep "ERROR |" $RESULT_DIR/pipeline.log

# View specific issue analysis
grep "Issue 3246:" $RESULT_DIR/pipeline.log
```

**Auto-Save**: The script saves progress every 10 issues, ensuring no data loss if interrupted. Final save occurs at completion.

---

### Step 4: CI Test Results Update
**Location**: `update_test_results/`

Updates test results from CI artifacts (torch-xpu-ops nightly and stock PyTorch XPU CI), populates Error Message/Traceback/torch-ops/dependency.

**Scripts**:
- `update_test_results.py` - Main script to update test results
- `generate_report.py` - Generate markdown report

**Usage**:
```bash
cd ~/ai_for_validation/opencode/issue_triage/update_test_results
python3 update_test_results.py
python3 generate_report.py
```

**Output**:
- Updates status columns (torch-xpu-ops nightly, stock CI)
- Populates Error Message, Traceback, torch-ops, dependency
- Analyzes case existence (CUDA/XPU)
- Adds Priority, Category, Root Cause columns
- Generates `issue_report.md`

---

### Step 4: Report Generation
**Location**: `update_test_results/`

Generates comprehensive markdown report with issue statistics.

**Usage**:
```bash
cd ~/ai_for_validation/opencode/issue_triage/update_test_results
python3 generate_report.py
```

**Output**: `issue_report.md` with:
- Statistics by Priority, Category, Root Cause
- Issue lists by action_TBD type
- Duplicated issues
- Issues without assignee

---

## Run Full Pipeline

```bash
# Set result directory
export RESULT_DIR=~/ai_for_validation/opencode/issue_triage/result

# Step 1: Collect issues
cd ~/ai_for_validation/opencode/issue_triage/issue_analysis/issue_basic_info_extraction
python3 generate_excel.py

# Step 2: Extract PRs
cd ~/ai_for_validation/opencode/issue_triage/issue_analysis/pr-extraction
python3 pr_extraction.py $RESULT_DIR/torch_xpu_ops_issues.xlsx

# Step 3: Category analysis (LLM-based)
cd ~/ai_for_validation/opencode/issue_triage/issue_analysis/category
python3 category_analyzer.py $RESULT_DIR/torch_xpu_ops_issues.xlsx

# Step 4: Update CI results and analysis
cd ~/ai_for_validation/opencode/issue_triage/update_test_results
python3 update_test_results.py

# Step 5: Generate report
cd ~/ai_for_validation/opencode/issue_triage/update_test_results
python3 generate_report.py
```

---

## Related Skills

- **test_result_analysis/torch-ops-extraction**: Extract torch operation information
- **test_result_analysis/check-xpu-test-existence**: Check if XPU test exists
- **test_result_analysis/check-cuda-test-existence**: Check if CUDA test exists
- **test_result_analysis/Test_Cases**: Process UT CI results
- **test_result_analysis/E2E_Test_Cases**: Process E2E benchmark results

---

## CI Data Sources

| Source | Path | Description |
|--------|------|-------------|
| torch-xpu-ops nightly | `/home/daisydeng/issue_traige/ci_results/torch-xpu-ops/` | XML test results |
| Stock PyTorch XPU | `/home/daisydeng/issue_traige/ci_results/stock/` | ZIP test reports |
| Inductor E2E | `Inductor_E2E_Test_Report.xlsx` | Benchmark accuracy results |

---

## Excel Sheets Structure

### Issues Sheet (22 columns after PR extraction and Category analysis)
| Column | Field |
|--------|-------|
| A | Issue ID |
| B | Title |
| C | Status |
| D | Assignee |
| E | Reporter |
| F | Labels |
| G | Created Time |
| H | Updated Time |
| I | Milestone |
| J | Summary |
| K | Type |
| L | Module |
| M | Test Module |
| N | Dependency |
| O | **PR** |
| P | **PR Owner** |
| Q | **PR Status** |
| R | PR Description |
| S | **Category** |
| T | Category Reason |

### Test Cases Sheet (20 columns)
| Column | Field |
|--------|-------|
| A | Issue ID |
| B | Test Reproducer |
| C | Test Type |
| D | Test File |
| E | Origin Test File |
| F | Test Class |
| G | Test Case |
| H | Error Message |
| I | Traceback |
| J | torch-ops |
| K | dependency |
| L | XPU Status |
| M | Stock Status |
| N | Is SKIP |
| O | Is CUDA Skip |
| P | CUDA Case Exist |
| Q | XPU Case Exist |
| R | case_existence_comments |
| S | can_enable_on_xpu |
| T | duplicated_issue |

### E2E Test Cases Sheet
| Column | Field |
|--------|-------|
| A | Issue ID |
| B | Test Reproducer |
| C | Benchmark |
| D | Model |
| E | Phase |
| F | Dtype |
| G | AMP |
| H | Backend |
| I | Test Type |
| J | Cudagraph |
| K | Error Message |
| L | Traceback |

---

## Test Type Classifications

| Type | Pattern | Description |
|------|---------|-------------|
| op_ut | `torch-xpu-ops/test/xpu/..._xpu.py` | XPU unit tests |
| op_extend | `third_party.torch-xpu-ops/...` | Extended operator tests |
| e2e | `benchmarks/dynamo/...` | End-to-end benchmark tests |
| benchmark | `benchmarks/...` | Other benchmark tests |
| ut | `test/test_*.py` | PyTorch unit tests (via XPUPatchForImport) |