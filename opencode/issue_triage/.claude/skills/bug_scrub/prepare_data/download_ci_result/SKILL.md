# Download CI Results Skill

Downloads latest completed CI artifacts from torch-xpu-ops and stock pytorch xpu workflows.

## Base Path Reference

Relative paths from this file location:
```
../../..                          → issue_triage root
../../ci_results/                 → CI artifacts directory
../../result/                     → Excel results directory
../../../<REPO>/<DEST>            → Full download destination
```

## Usage

Use this skill when analyzing XPU test failures or comparing CI results between torch-xpu-ops and pytorch/pytorch main branch.

## Download Process

### 1. Find Latest Completed Run

For torch-xpu-ops:
```bash
gh run list -R intel/torch-xpu-ops --workflow nightly_ondemand.yml -L 1 --json databaseId,name,status,conclusion,createdAt
```

For stock pytorch, find latest completed (not in_progress):
```bash
gh run list -R pytorch/pytorch --workflow xpu.yml -L 10 --json databaseId,name,status,conclusion,createdAt | jq -c '.[] | select(.status=="completed")'
```

### 2. Download Artifacts

```bash
gh run download <RUN_ID> -R <OWNER>/<REPO> -D <DESTINATION_DIR>
```

Example for torch-xpu-ops:
```bash
gh run download 24671332594 -R intel/torch-xpu-ops -D ../../../ci_results/torch-xpu-ops
```

Example for stock pytorch:
```bash
gh run download 24679922053 -R pytorch/pytorch -D ../../../ci_results/stock
```

## Artifact Locations

### torch-xpu-ops Artifacts
Downloaded to: `../../ci_results/torch-xpu-ops/`

Key artifacts include:
- **UT-Data-***: Test data from unit tests (basic, op_ut, xpu_distributed, xpu_profiling, Windows)
- **UT-Failure-List-***: Lists of failed tests per shard
- **E2E-Data-***: End-to-end benchmark results (timm_models, pt2e, torchbench)
- **OP-Benchmark-Data**: Operator benchmark results

### Stock PyTorch XPU Artifacts
Downloaded to: `../../ci_results/stock/`

Key artifacts include:
- **logs-***: Build and test logs per shard
- **test-jsons-***: Test result JSONs per shard
- **test-reports-***: Test reports per shard

## Example Script

```bash
#!/bin/bash
# Download latest CI artifacts from both repos

# Setup directories (relative path from bug_scrub/prepare_data/download_ci_result/)
mkdir -p ../../../ci_results/torch-xpu-ops
mkdir -p ../../../ci_results/stock

# Download torch-xpu-ops artifacts
RUN_ID_XPU=$(gh run list -R intel/torch-xpu-ops --workflow nightly_ondemand.yml -L 1 --json databaseId -q '.[0].databaseId')
gh run download $RUN_ID_XPU -R intel/torch-xpu-ops -D ../../../ci_results/torch-xpu-ops

# Download stock pytorch xpu artifacts
# Find latest completed (not in_progress)
RUN_ID_STOCK=$(gh run list -R pytorch/pytorch --workflow xpu.yml -L 10 --json databaseId,status -q '.[] | select(.status=="completed") | .databaseId' | head -1)
gh run download $RUN_ID_STOCK -R pytorch/pytorch -D ../../../ci_results/stock

echo "Download complete"
```

## Notes

- Required: `gh` CLI authenticated to GitHub
- Artifacts expire after 90 days by default
- Stock pytorch xpu artifacts are downloaded as zip files (may need extraction)
- Total files typically: 500+ for torch-xpu-ops, 2700+ for stock pytorch