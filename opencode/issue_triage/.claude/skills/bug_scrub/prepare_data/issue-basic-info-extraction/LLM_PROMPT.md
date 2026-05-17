# LLM Extraction Prompt (Sub-Agent Contract)

This is the **exact prompt template** used when spawning parallel sub-agents to
extract structured information from torch-xpu-ops issues. Inputs and outputs
are JSON files on disk; the sub-agent does not call GitHub.

## Role in the Pipeline

The LLM is the **fallback** extractor. Deterministic script extractors in
`generate_excel.py` (`parse_test_cases_from_body`, `parse_e2e_info`) run
first; the LLM cache is consulted only when those scripts produce nothing.
The sub-agent does not know which issues will hit the fallback — it extracts
for every issue in the batch, and `generate_excel.py` chooses whether to
consult the cache per-issue. So the contract is: produce a high-quality
record for every input issue, even if it ends up unused.

## Sub-Agent Inputs

| Path | Contents |
|---|---|
| `data/llm_batches/batch_NN.json` | List of `{issue_id, title, labels, body, body_hash}` (10 issues) |
| `data/llm_results/batch_NN.json` | (output) List of schema-conforming records |

## Prompt Template

```
You are extracting structured information from GitHub issues for the
intel/torch-xpu-ops repository. Read EVERY issue in the input batch and
emit ONE JSON object per issue, exactly matching the schema below.

INPUT FILE:   data/llm_batches/batch_<NN>.json
OUTPUT FILE:  data/llm_results/batch_<NN>.json
OUTPUT SHAPE: a JSON LIST of objects, one per input issue, in input order.

Per-issue schema (ASCII only, never invent fields):
{
  "issue_id":      <int, copied from input>,
  "body_hash":     "<string, copied from input>",
  "kind":          "unittest" | "e2e" | "other",
  "test_cases": [
    {"test_file":   "<path as the issue quotes it>",
     "test_class":  "<class name OR benchmark suite>",
     "test_method": "<method name OR model name>"}
  ],
  "reproducer":    "<verbatim block, see Rule 1>",
  "error_message": "<first user-visible error sentence; '' if none>",
  "traceback":     "<full traceback if present; '' if none>",
  "notes":         "<one-sentence semantic summary>"
}

MANDATORY RULES — these are non-negotiable:

1. REPRODUCER IS VERBATIM.
   Every URL that is part of the reproduction instructions is MANDATORY:
   repository links, gist links, branch/tag links, dataset links, docs links,
   instruction links. NEVER paraphrase. NEVER drop URLs. NEVER summarize.
   If the issue says "clone <URL> and run <CMD>", both the URL and CMD belong
   in the reproducer field, exactly as written.

2. test_cases IS EMPTY UNLESS THE ISSUE POINTS AT RUNNABLE TESTS.
   A "[E2E]" prefix in the title is NOT sufficient to populate test_cases.
   You must see an actual test file path (for unittest) or an actual
   benchmark/model name (for e2e) in the body.

3. UNITTEST FORMAT.
   test_file   = path as quoted in the issue (e.g. "test/dynamo/test_x.py")
   test_class  = class name (may be "")
   test_method = method name (may be "" if only the file is identified)

4. E2E FORMAT.
   test_class  = benchmark suite (e.g. "Timm", "Torchbench", "Huggingface")
   test_method = model name (e.g. "convnext_base", "hf_Reformer")
   Each (suite, model) pair is one entry. If the issue lists multiple
   models per suite as prose ("Timm: a, b, c"), produce one entry per model.

5. kind CLASSIFIES THE ISSUE'S SUBJECT, NOT ITS LABELS:
   - "unittest" — fix will land in a test under pytorch/test/** or
                  torch-xpu-ops/test/**
   - "e2e"      — benchmark accuracy / model perf / model run
   - "other"    — infra, build, runtime API, kernel-perf without a test,
                  enhancement requests, design discussion, etc.
   An issue with kind == "e2e" but no enumerable (suite, model) pairs in
   the body MUST still set kind = "e2e" with test_cases = [] (downstream
   logic will route it to Others).

6. NO FABRICATION.
   If the issue contains no reproducer, set reproducer = "".
   If there is no traceback, set traceback = "".
   Never invent a benchmark, test path, model, or error message.

7. ASCII OUTPUT ONLY.
   No smart quotes, em-dashes, arrows, or non-ASCII characters.

8. DEEP READING IS REQUIRED.
   Cross-paragraph references are common. A test path may be quoted in one
   paragraph and the class/method in another. A reproducer may interleave
   prose with a fenced code block. You MUST synthesize across the whole body.

9. COPY issue_id AND body_hash VERBATIM from the input record.

10. OUTPUT IS A JSON LIST in the same order as the input batch. Do not
    drop, reorder, or add issues.

PROCESS:
  - Open data/llm_batches/batch_<NN>.json.
  - For each input record, build the schema object.
  - Write data/llm_results/batch_<NN>.json (JSON list, pretty-printed).
  - Do NOT call any network tool. Everything you need is in the input file.
  - Do NOT write any other file.

VERIFY before finishing:
  - len(output_list) == len(input_list)
  - Every output entry has all 8 schema keys.
  - kind ∈ {"unittest", "e2e", "other"}.
  - test_cases is a list (possibly empty) of dicts with exactly the three
    keys test_file / test_class / test_method.
```

## Example output entry (illustrative — not normative)

```json
{
  "issue_id": 9999,
  "body_hash": "abc123def4567890",
  "kind": "e2e",
  "test_cases": [
    {"test_file": "", "test_class": "Timm",       "test_method": "convnext_base"},
    {"test_file": "", "test_class": "Torchbench", "test_method": "hf_Reformer"}
  ],
  "reproducer": "git clone https://github.com/example/benchmarks ...\npython run.py --device xpu",
  "error_message": "eager_two_runs_differ",
  "traceback": "",
  "notes": "Timm/Torchbench accuracy regression on ARC."
}
```

## Spawning sub-agents (orchestrator side)

For each batch file, fire one background sub-agent. All sub-agents run in
parallel. The orchestrator does NOT poll; it waits for `<system-reminder>`
notifications and then collects results.

```python
# Pseudocode for the orchestrator
for nn in range(num_batches):
    task(
        subagent_type="general",
        run_in_background=True,
        load_skills=[],
        description=f"LLM extract batch {nn:02d}",
        prompt=PROMPT_TEMPLATE.replace("<NN>", f"{nn:02d}"),
    )
# After all batches complete:
#   python merge_llm_results.py
```

## Failure handling

If a sub-agent returns a malformed JSON file, `merge_llm_results.py`:

- prints `SKIP batch_NN.json: bad JSON` to stderr,
- leaves the previous cache entry (if any) untouched, and
- continues with the remaining batches.

Re-run the single failing batch by:

```bash
rm data/llm_results/batch_NN.json
# spawn one sub-agent for batch_NN.json
python merge_llm_results.py
```
