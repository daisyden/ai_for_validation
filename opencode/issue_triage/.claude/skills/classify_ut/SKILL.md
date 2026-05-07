# classify_ut

This skill follows agent-guidelines AND extends it with workbook-specific UT classification rules.
Always apply agent-guidelines rules including the mandatory post-write commit protocol.

## Purpose

Maintain and classify the `Non-Inductor XPU Skip` sheet in
`/home/daisyden/opencode/classify/data/Non_inductor_ut_status_ww14_26.xlsx`
without pattern matching.

## Inputs

- Target workbook:
  `/home/daisyden/opencode/classify/data/Non_inductor_ut_status_ww14_26.xlsx`
- Reference workbook:
  `/home/daisyden/opencode/ai_for_validation/opencode/issue_triage/result/torch_xpu_ops_issues.xlsx`
- Deep case-existence workflow:
  `/home/daisyden/opencode/ai_for_validation/opencode/issue_triage/.claude/skills/bug_scrub/analyze_ci_result/check_xpu_case_existence/SKILL.md`

## check_xpu_case_existence reference

- Skill path:
  `/home/daisyden/opencode/ai_for_validation/opencode/issue_triage/.claude/skills/bug_scrub/analyze_ci_result/check_xpu_case_existence/SKILL.md`

### Required tools from check_xpu_case_existence

- `read` - inspect skill docs, wrappers, base tests, and supporting source files
- `bash` - run environment activation, directory listing, and `pytest --collect-only`
- `grep` - inspect exact code locations after semantic analysis identifies targets
- `glob` - enumerate candidate XPU test files and subfolders

### Constraints inherited from check_xpu_case_existence

- Do not use pattern matching alone to classify a row.
- Perform deep analysis through actual file/code inspection and execution evidence.
- Run Python analysis inside `pytorch_opencode_env`.
- Treat `third_party/torch-xpu-ops` as the XPU test location, not a separate workspace root.
- Check all relevant XPU test locations before declaring a case missing, including:
  - `third_party/torch-xpu-ops/test/xpu/`
  - `third_party/torch-xpu-ops/test/xpu/distributed/`
  - `third_party/torch-xpu-ops/test/xpu/extended/`
  - `third_party/torch-xpu-ops/test/xpu/nn/`
  - `third_party/torch-xpu-ops/test/xpu/functorch/`
  - `third_party/torch-xpu-ops/test/xpu/quantization/`
- For distributed/XCCL-style cases, check the torch-xpu-ops distributed folder even when there is no matching upstream `pytorch/test` file.
- When a test is missing locally, verify against the upstream PyTorch release branch before concluding it was removed or renamed.
- Respect `XPUPatchForImport(False)` vs `XPUPatchForImport(True)` semantics when deciding whether a test should be generated.
- Distinguish generated-but-skipped cases from genuinely missing cases.

## Workflow

1. Prepare the requested `pytorch_opencode_env` environment and nightly XPU packages.
2. Open the `Non-Inductor XPU Skip` sheet.
3. Ensure workbook columns `Exaplaination` and `Reason TBD` exist.
4. If XPU test metadata is blank, derive it from CUDA metadata:
   - `classname_cuda` ending with `CUDA` -> replace with `XPU`
   - `name_cuda` ending with `_cuda` -> replace with `_xpu`
   - `testfile_xpu` defaults to `testfile_cuda` when blank
5. Set `Reason TBD` to `True` when `Reason` is blank, otherwise `False`.
6. If both `status_xpu` and `Reason` are blank, do not guess and do not use pattern matching.
   Run the deep-analysis steps from the referenced `check_xpu_case_existence` skill, then fill:
   - `Reason = Not applicable`, `DetailReason = Community Changes`, and `Exaplaination = why the exact XPU case does not exist and what the expected case name is`
   - or `Reason = Not Appliable`, `DetailReason = missing feature/API`
   - or `Reason = To be enabled`
7. Consolidate `Not applicable` / `Not Appliable` features from both workbooks into
   `/home/daisyden/opencode/ai_for_validation/opencode/issue_triage/not_appliable.txt`.

## Files

- Script: `./run_classify_ut.py`

## Usage

```bash
source ~/miniforge3/bin/activate pytorch_opencode_env && \
python /home/daisyden/opencode/ai_for_validation/opencode/issue_triage/.claude/skills/classify_ut/run_classify_ut.py
```

## Script behavior

`run_classify_ut.py` performs deterministic workbook maintenance only:

- ensures the requested columns exist
- fills missing XPU metadata from CUDA metadata
- normalizes `Reason TBD`
- writes `not_appliable.txt`
- reports any rows that still require deep case-existence analysis

The script intentionally does not hardcode semantic classifications and does not use regex-based case classification.
When pending rows exist, follow the deep-analysis workflow from the referenced `check_xpu_case_existence` skill before writing `Reason`, `DetailReason`, and `Exaplaination`.

## Notes

- `Exaplaination` keeps the requested spelling.
- Save a backup before writing the workbook.
- Preserve existing `Reason`, `DetailReason`, and `Exaplaination` unless deep analysis justifies an update.
