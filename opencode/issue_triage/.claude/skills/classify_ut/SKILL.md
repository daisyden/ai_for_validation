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
- Blank `status_xpu` workflow:
  `/home/daisyden/opencode/ai_for_validation/opencode/issue_triage/.claude/skills/classify_ut/blank/SKILL.md`
- Failed `status_xpu` workflow:
  `/home/daisyden/opencode/ai_for_validation/opencode/issue_triage/.claude/skills/classify_ut/failed/SKILL.md`
- Skipped/xfail `status_xpu` workflow:
  `/home/daisyden/opencode/ai_for_validation/opencode/issue_triage/.claude/skills/classify_ut/skipped/SKILL.md`

## Status-specific classification skills

These subskills are authoritative for case-specific `Reason`, `DetailReason`, and
`Exaplaination` decisions. Always read the matching subskill before classifying rows with that
`status_xpu` value.

| `status_xpu` | Skill | Purpose |
|--------------|-------|---------|
| blank / empty | `classify_ut/blank/SKILL.md` | Deep case-existence analysis for rows with no XPU result |
| `failed` | `classify_ut/failed/SKILL.md` | Failure-message, local-run, and known-issue analysis |
| `skipped` / `xfail` | `classify_ut/skipped/SKILL.md` | Skip-message, linked-issue, source, and local-run analysis |

Do not collapse these workflows into a single pattern-matching script. Bulk scripts may prepare
workbook columns, collect candidate rows, or apply already-reviewed decisions, but the actual
classification must follow the status-specific skill.

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
- **Do not classify by filename patterns alone.** Bulk scripts may prepare workbook columns,
  but semantic classification must come from reading the relevant test code, wrapper/direct
  XPU file, parametrization, skip lists, and, for distributed tests only, release-branch source.
- **For distributed tests** (origin_file under `test/distributed/`), the XPU test mechanism
  is NOT a `*_xpu.py` wrapper file. Instead:
  1. Read `third_party/torch-xpu-ops/test/xpu/run_distributed.py` in full to confirm
     which skip dictionaries are imported in the current checkout.
  2. For release/2.12 distributed classification, first read the remote local distributed
     skip list from `intel/torch-xpu-ops` branch `daisyden/distributed_2.12`:
     `test/xpu/skip_list_dist_local.py`. The intended file is sometimes referred to as
     `skip_list_local_dist.py`, but the verified branch filename is `skip_list_dist_local.py`.
  3. Read `third_party/torch-xpu-ops/test/xpu/skip_list_dist.py` in full, or the matching
     `intel/torch-xpu-ops` `daisyden/distributed_2.12:test/xpu/skip_list_dist.py` when
     classifying against release/2.12.
  4. If `third_party/torch-xpu-ops/test/xpu/skip_list_dict_local.py` exists, read it in full.
      Treat it as enabling the listed distributed file(s), and use the skipped-case list in
      that file when deciding whether an individual test case runs or is intentionally skipped.
      If it does not exist in the checkout, explicitly state that it was checked and absent.
   5. Check whether the upstream test file path appears in the active distributed skip dicts.
     - Present, value `None` → all tests in that file run on XPU.
     - Present, value is a tuple/list → tests in the tuple/list are skipped; all others run.
     - Absent from all active distributed skip dicts → the file is not run on XPU by
       `run_distributed.py`; classify as `To be enabled` with a **specific** DetailReason such
       as `Distributed file missing from skip_list_dist.py` or
       `Distributed file missing from skip_list_dict_local.py`, not the generic phrase
       `No XPU wrapper`.
   6. Also check `third_party/torch-xpu-ops/test/xpu/distributed/` for XPU-native
      standalone files (e.g. `test_c10d_xccl.py`) that have no upstream pytorch/test equivalent.
   7. For distributed tests, also check `https://github.com/daisyden/pytorch/tree/release/2.12`
      for the upstream file and test method before concluding a test was removed, renamed, or
      never present.
- Not all XPU tests use `XPUPatchForImport`. Distributed tests run the upstream file
  directly via `run_distributed.py`; many non-distributed XPU files are standalone copies or
  direct XPU implementations without `XPUPatchForImport`. For those files, inspect the actual
  class/function definitions and `instantiate_device_type_tests`/`instantiate_parametrized_tests`
  calls in `third_party/torch-xpu-ops/test/xpu/**` instead of assuming wrapper behavior.
- Respect `XPUPatchForImport(False)` vs `XPUPatchForImport(True)` semantics when
  deciding whether a test should be generated (non-distributed tests only).
- Distinguish generated-but-skipped cases from genuinely missing cases.
- CUDA graph / cudagraph rows must not be treated as `Not Appliable` merely because the CUDA
  name contains `cuda`. XPU graph support is present via `_XPUGraph` and `torch.accelerator.Graph`;
  classify missing/failing CUDA graph coverage as `To be enabled` with a DetailReason mentioning
  the exact XPU graph gap.
- `Not Appliable` with CUDA-specific APIs must name the API in `DetailReason`, e.g.
  `CUDA-specific API: cuBLAS` or `CUDA-specific API: torch.cuda.jiterator`. Do not use only
  `CUDA-specific API`.
- `Not applicable / Community Changes` is only for a CUDA test that no longer exists or was
  renamed/removed in the PyTorch source being compared. If both the CUDA variant and the XPU
  variant exist after parametrization (for example `test_Conv1d_pad2_cuda` and
  `test_Conv1d_pad2_xpu`), it is **not** a community-change case; analyze the XPU status,
  skips, failure, or enablement gap instead.
- If `status_xpu = failed`, use `classify_ut/failed/SKILL.md`. Failed rows must inspect
  `message_xpu`, run locally when the message is missing or not useful, search
  `intel/torch-xpu-ops` issues, include a known issue link when found, and use `[Issue TBD]`
  in `DetailReason` when no known issue exists after analysis.
- If `status_xpu = skipped` or `xfail`, use `classify_ut/skipped/SKILL.md`. Skipped rows are
  not automatically environment limitations. Inspect issue links and issue state, read local skip
  decorators/helpers, and run locally when `message_xpu` is missing or nondiagnostic. In particular,
  `not-support-multithread` is a `Feature gap`, closed PyTorch disabled-test issues are
  `To be enabled`, and open disabled-test issues are classified by issue content.

## Workflow

1. Prepare the requested `pytorch_opencode_env` environment and nightly XPU packages.
2. Open the `Non-Inductor XPU Skip` sheet.
3. Ensure workbook columns `Exaplaination` and `Reason TBD` exist.
4. Before any case-specific analysis or updates, initialize `Reason TBD` from the current `Reason` value:
   - if `Reason` is blank, set `Reason TBD = True`
   - otherwise set `Reason TBD = False`
5. If XPU test metadata is blank, derive it from CUDA metadata:
   - `classname_cuda` ending with `CUDA` -> replace with `XPU`
   - `name_cuda` ending with `_cuda` -> replace with `_xpu`
   - `testfile_xpu` defaults to `testfile_cuda` when blank
6. If `Reason` is blank, do not guess and do not use pattern matching. Choose and read the
   status-specific deep-analysis skill first:
   - blank `status_xpu` -> `classify_ut/blank/SKILL.md`
   - `status_xpu = failed` -> `classify_ut/failed/SKILL.md`
   - `status_xpu = skipped` or `xfail` -> `classify_ut/skipped/SKILL.md`
   Run the selected skill's workflow, plus the referenced `check_xpu_case_existence` workflow
   when the selected skill calls for case-existence analysis.
7. Fill `Reason`, `DetailReason`, and `Exaplaination` according to the selected subskill. The
   complete set of outcomes used by the subskills includes:
   - `Reason = To be enabled`, with a specific `DetailReason` describing the enablement gap,
     stale skip, closed disabled issue, local pass, or existing-but-unreported XPU case.
   - `Reason = Feature gap`, with a specific missing XPU feature/support gap, issue link when
     available, or `[Issue TBD]` when no known issue exists.
   - `Reason = Failures (xpu broken)` or `Reason = Failures (XPU broken)`, with a known issue link
     when found, or `[Issue TBD]` plus the concrete failure when no known issue exists.
   - `Reason = Test Enviroment limitation`, only for true hardware/process/test-run requirements
     such as insufficient GPU count or explicitly disabled slow-test gates. Do not use this merely
     because `status_xpu` is skipped.
   - `Reason = Not Appliable`, `DetailReason = CUDA-specific API: <exact API name>` or another
     precise CUDA-only/backend-specific API or feature name.
   - `Reason = Not applicable`, `DetailReason = Community Changes`, and `Exaplaination` explaining
     why the CUDA case was removed/renamed and which source versions prove it.
8. `Reason TBD` tracks whether the **original** Reason was blank at the time the workbook was processed.
   Do NOT change `Reason TBD` to False after filling in a classification — leave it as initialized in step 4.
   `Reason TBD = True` means "this row originally had no Reason and required deep analysis", regardless of whether analysis is now complete.
9. Consolidate `Not applicable` / `Not Appliable` features from both workbooks into
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
- initializes `Reason TBD` before case-specific work
- fills missing XPU metadata from CUDA metadata
- initializes `Reason TBD`; do not flip it to False after later classification
- writes `not_appliable.txt`
- reports any rows that still require deep case-existence analysis

The script intentionally must not hardcode semantic classifications and must not use regex-based
case classification. When pending rows exist, follow the deep-analysis workflow from the referenced
`check_xpu_case_existence` skill before writing `Reason`, `DetailReason`, and `Exaplaination`.

## Notes

- `Exaplaination` keeps the requested spelling.
- Save a backup before writing the workbook.
- Preserve existing `Reason`, `DetailReason`, and `Exaplaination` unless deep analysis justifies an update.
