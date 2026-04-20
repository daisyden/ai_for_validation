# Priority Analysis Skill

## Overview
This skill provides automatic priority assessment of torch-xpu-ops issues based on deep analysis of issue content, error messages, traceback, log output, and code investigation.

---

## Priority Definitions

### P0 - Critical

| Condition | Reason | Examples |
|-----------|--------|----------|
| Torch build failure | Critical blocking issue | Compilation failures, linker errors |
| Crash/Segmentation/Segfault | Process termination | GPU hang, kernel panic, SIGSEGV |
| PyTorch fatal error | Process termination | FATAL errors, abort calls |
| Regression (>5% perf drop) | Performance degradation | Was faster, now significantly slower |
| Custom model impact | Blocks customer deployment | Production model failures |

### P1 - High

| Condition | Reason | Examples |
|-----------|--------|----------|
| UT > 20 failed cases | Major test suite failure | Large test class failures |

### P2 - Medium

| Condition | Reason | Examples |
|-----------|--------|----------|
| Benchmark functionality | E2E benchmark errors | Assertion failures, runtime errors |
| Benchmark accuracy | Precision issues | Numerical accuracy degradation |
| Benchmark performance | Performance degradation | <5% slower, acceptable |
| UT few failures | Minor test failures | 1-20 related failures |

### P3 - Low

| Condition | Reason | Examples |
|-----------|--------|----------|
| Minor issues | Cosmetic, no failures | Warning mismatches, documentation |
| Documentation | Missing docs | Unclear API |
| Enhancement | New feature requests | Feature requests |

---

## Priority Analysis Tools

### 1. Error Type Analyzer
```python
def analyze_error_type(error_log: str) -> dict:
    """
    Analyze error log to identify error severity.
    
    Returns:
        {
            "is_critical": bool,
            "error_class": "Fatal|Crash|Error|Warning|Info",
            "evidence": [matched patterns],
            "priority_impact": "P0|P1|P2|P3"
        }
    """
    
    # Critical error patterns
    CRITICAL_PATTERNS = {
        "pattern": r"(Segmentation fault|fat.*crash|Abort.*called|SIGSEGV)",
            "severity": "Fatal",
            "priority": "P0",
            "weight": 1.0
        },
        {
            "pattern": r"(FATAL|panic|critical|abort)",
            "severity": "Fatal", 
            "priority": "P0",
            "weight": 0.9
        },
        "build_failure": {
            "pattern": r"(build.*fail|compilation.*error|linker.*error|make.*fail)",
            "severity": "Fatal",
            "priority": "P0",
            "weight": 1.0
        },
    }
    
    # High error patterns
    HIGH_PATTERNS = {
        "many_tests": {
            "pattern": r"(\d+)\s*failed",
            "severity": "High",
            "priority": "P1",
            "weight": 0.7
        },
    }
    
    # Medium error patterns
    MEDIUM_PATTERNS = {
        "few_tests": {
            "pattern": r"(failed|error|assert).{0,50}(test|Test)",
            "severity": "Medium",
            "priority": "P2",
            "weight": 0.5
        },
        "performance": {
            "pattern": r"(slow|latency|performance|throughput)",
            "severity": "Medium",
            "priority": "P2",
            "weight": 0.4
        },
    }
    
    # Low error patterns
    LOW_PATTERNS = {
        "warning": {
            "pattern": r"(warning|Warning|deprecated)",
            "severity": "Low",
            "priority": "P3",
            "weight": 0.2
        },
        "minor": {
            "pattern": r"(docs?|documentation|enhancement|feature)",
            "severity": "Low",
            "priority": "P3",
            "weight": 0.1
        }
    }
    
    import re
    
    results = {
        "is_critical": False,
        "error_class": "Unknown",
        "matched_patterns": [],
        "priority_scores": {"P0": 0, "P1": 0, "P2": 0, "P3": 0}
    }
    
    all_patterns = {**CRITICAL_PATTERNS, **HIGH_PATTERNS, **MEDIUM_PATTERNS, **LOW_PATTERNS}
    
    for name, info in all_patterns.items():
        matches = re.findall(info["pattern"], error_log, re.IGNORECASE)
        if matches:
            results["matched_patterns"].append({
                "pattern_name": name,
                "matches": matches,
                "severity": info["severity"],
                "priority": info["priority"]
            })
            results["priority_scores"][info["priority"]] += info["weight"]
    
    # Determine final classification
    priority_score = max(results["priority_scores"].items(), key=lambda x: x[1])
    results["suggested_priority"] = priority_score[0]
    
    if results["priority_scores"]["P0"] > 0.5:
        results["is_critical"] = True
        results["error_class"] = "Fatal"
    
    return results
```

### 2. Regression Detector
```python
def detect_regression(issue_data: dict, comments: list = []) -> dict:
    """
    Detect if issue is a regression based on context.
    
    Indicators:
    - Issue mentions "was working" or "was passing"
    - Regression keywords in comments
    - Test case that was passing now failing
    
    Returns:
        {
            "is_regression": bool,
            "confidence": float,
            "evidence": [list of regression indicators],
            "priority_boost": int (1 for regression)
        }
    """
    
    regression_keywords = [
        "was working", "was passing", "was fine",
        "regression", "regressed", "broken",
        "used to work", "no longer", "not working",
        "passed before", "failed after",
        "previously", "prior to",
    ]
    
    all_text = issue_data.get("body", "").lower()
    for comment in comments:
        all_text += " " + comment.lower()
    
    matched = []
    for kw in regression_keywords:
        if kw in all_text:
            matched.append(kw)
    
    is_regression = len(matched) > 0
    
    return {
        "is_regression": is_regression,
        "confidence": min(len(matched) / 3.0, 1.0),
        "matched_indicators": matched,
        "priority_boost": 1 if is_regression else 0,  # Elevate priority
        "reason": "Regression detected - was passing, now failing"
    }
```

### 3. Test Failure Counter
```python
def count_test_failures(error_log: str, issue_body: str) -> dict:
    """
    Count failed test cases from error log and issue body.
    
    Returns:
        {
            "total_failed": int,
            "test_type": str,  # UT/E2E/Mixed
            "severity": str,
            "priority": str
        }
    """
    
    import re
    
    # Extract failure counts
    failure_patterns = [
        r"(\d+)\s*failed",           # "X failed"
        r"failure.*(\d+)",           # "failure X"
        r"(\d+)\s*failure",           # "X failure"
        r"passed.*(\d+)",           # "passed X"
    ]
    
    max_failures = 0
    for pattern in failure_patterns:
        matches = re.findall(pattern, error_log + issue_body, re.IGNORECASE)
        for match in matches:
            try:
                count = int(match) if isinstance(match, str) else 1
                max_failures = max(max_failures, count)
            except:
                pass
    
    # Classify severity
    if max_failures == 0:
        severity = "No failures"
        priority = "P3"
    elif max_failures <= 5:
        severity = "Few failures"
        priority = "P2"
    elif max_failures <= 20:
        severity = "Moderate failures"
        priority = "P2"
    else:
        severity = "Many failures"
        priority = "P1"
    
    # Check if UT or E2E
    test_type = "Unknown"
    if "op_ut" in issue_body or "_xpu" in issue_body:
        test_type = "UT"
    if "benchmark" in issue_body or "huggingface" in issue_body or "timm" in issue_body:
        test_type = "E2E"
    if "pytest" in issue_body or "test_" in issue_body:
        test_type = "UT"
    
    return {
        "total_failed": max_failures,
        "test_type": test_type,
        "severity": severity,
        "priority": priority
    }
```

### 4. Custom Model Detector (with Benchmark Reference)
```python
def detect_custom_model_impact(issue_body: str, comments: list) -> dict:
    """
    Detect if issue impacts custom models (not benchmarks).
    
    Uses BENCHMARK_MODELS.py for accurate benchmark model detection:
    - HuggingFace: gemma, llama, mistral, qwen, opt, gpt, etc.
    - Timm: resnet, vit, efficientnet, convnext, swin, etc.
    - TorchBench: bert_pytorch, resnet18, resnet50, vgg, etc.
    
    Custom model indicators:
    - Production/customer/enterprise mentions
    - "our model", "internal model"
    - NOT matching benchmark patterns
    
    Returns:
        {
            "has_custom_impact": bool,
            "confidence": float,
            "model_type": str,  # Custom/Benchmark/Unknown
            "priority_boost": int
        }
    """
    
    import re
    
    # Load benchmark patterns from external reference
    # Reference: BENCHMARK_MODELS.py
    
    # HuggingFace model patterns
    HF_PATTERNS = [
        r"Albert[A-Z]\w*", r"Bart[A-Z]\w*", r"Bert[A-Z]\w*",
        r"Blenderbot[A-Z]\w*", r"CamemBert[A-Z]\w*", r"Deberta[A-Z]\w*",
        r"Distil[A-Z]\w*", r"Electra[A-Z]\w*", r"GPT2[A-Z]\w*",
        r"GPTJ[A-Z]\w*", r"GPTNeo[A-Z]\w*", r"LayoutLM[A-Z]\w*",
        r"LLama[A-Z]\w*", r"Megatron[A-Z]\w*", r"Mobile[A-Z]\w*",
        r"OPT[A-Z]\w*", r"Roberta[A-Z]\w*", r"T5[A-Z]\w*",
        r"google/gemma", r"meta-llama/Llama", r"mistralai/Mistral",
        r"openai/gpt", r"openai/whisper", r"Qwen/AwQ", r"XGLM",
    ]
    
    # Timm model patterns
    TIMM_PATTERNS = [
        r"adv_inception", r"beit", r"botnet", r"cait", r"coat",
        r"deit", r"dla", r"dm_nfnet", r"eca_", r"ese_vovnet",
        r"fbnet", r"gernet", r"ghostnet", r"gmixer", r"gmlp",
        r"hrnet", r"inception", r"lcnet", r"levit", r"mixer_b",
        r"mixnet", r"mnasnet", r"mobilenet", r"mobilevit", r"nfnet",
        r"pnasnet", r"poolformer", r"regnety", r"repvgg", r"res2net",
        r"sebotnet", r"selecsls", r"spnasnet", r"swin",
        r"tf_efficientnet", r"tinynet", r"tnt_", r"twins",
        r"vit_", r"volo", r"xcit",
    ]
    
    # TorchBench patterns
    TORCHBENCH_PATTERNS = [
        r"BERT_pytorch", r"Background_Matting", r"LearningToPaint",
        r"dcgan", r"densenet", r"mobilenet_v", r"nvidia_deeprecommender",
        r"resnet\d*", r"resnext", r"shufflenet", r"squeezenet",
    ]
    
    # Check if it's a benchmark model
    all_bm_patterns = HF_PATTERNS + TIMM_PATTERNS + TORCHBENCH_PATTERNS
    combined_bm = "|".join(f"({p})" for p in all_bm_patterns)
    
    is_benchmark = bool(re.search(combined_bm, issue_body, re.IGNORECASE))
    
    # Custom indicators
    CUSTOM_INDICATORS = [
        "production", "customer", "custom application",
        "our model", "our application", "internal model",
        "specific model", "enterprise", "deployment"
    ]
    
    # Check for custom impact
    is_custom = any(ind in issue_body.lower() for ind in CUSTOM_INDICATORS)
    
    model_type = "Unknown"
    if is_custom and not is_benchmark:
        model_type = "Custom"
        confidence = 0.9  # High confidence - explicit custom context
    elif is_benchmark:
        model_type = "Benchmark"
        confidence = 0.95  # Very high - matches benchmark patterns
    else:
        model_type = "Unknown"
        confidence = 0.3
    
    return {
        "has_custom_impact": is_custom and not is_benchmark,
        "model_type": model_type,
        "confidence": confidence,
        "priority_boost": 1 if (is_custom and not is_benchmark) else 0,
        "matched_benchmark_patterns": re.findall(combined_bm, issue_body, re.IGNORECASE) if is_benchmark else []
    }
```

### 5. Performance Regression Analyzer
```python
def analyze_performance_regression(error_log: str, issue_body: str) -> dict:
    """
    Analyze if issue indicates performance degradation.
    
    Performance indicators:
    - Percentage slowdown mentioned
    - Latency increase
    - Throughput decrease
    
    Returns:
        {
            "is_performance_issue": bool,
            "perf_category": str,  # slow/latency/throughput/
            "percent_degradation": float,
            "severity": str
        }
    """
    
    import re
    
    perf_patterns = [
        r"(\d+)%\s*slower",           # "X% slower"
        r"(\d+)%\s*degradation",      # "X% degradation"
        r"(\d+)x\s+slower",           # "Xx slower"
        r"(\d+)\s*times\s+slower",    # "X times slower"
        r"slowdown.*(\d+)%",           # "slowdown X%"
    ]
    
    max_perf_drop = 0
    for pattern in perf_patterns:
        matches = re.findall(pattern, (error_log + issue_body).lower())
        for match in matches:
            try:
                pct = int(match)
                if "times" in pattern:
                    pct *= 100  # Convert X times to percentage
                max_perf_drop = max(max_perf_drop, pct)
            except:
                pass
    
    # Classify severity
    if max_perf_drop == 0:
        return {"is_performance_issue": False, "percent_degradation": 0}
    
    if max_perf_drop > 5:
        severity = "Critical (>5% regression)"
        priority = "P0"
    elif max_perf_drop > 1:
        severity = "Moderate (1-5% regression)"
        priority = "P2"
    else:
        severity = "Minor (<1% regression)"
        priority = "P3"
    
    return {
        "is_performance_issue": True,
        "percent_degradation": max_perf_drop,
        "severity": severity,
        "priority": priority
    }
```

---

## Combined Priority Analysis

### Priority Analyzer
```python
def analyze_priority(
    issue_data: dict,
    error_log: str,
    stack_trace: str,
    execution_result: dict = None
) -> dict:
    """
    Combined priority analysis using all analyzers.
    
    Analysis Flow:
    1. Error type analysis
    2. Test failure counting
    3. Regression detection
    4. Performance analysis
    5. Custom model impact
    6. Combine scores for final priority
    
    Returns:
        {
            "final_priority": str,
            "priority_score": float,
            "reason": str,
            "confidence": float,
            "components": {...}
        }
    """
    
    # 1. Error type analysis
    error_analysis = analyze_error_type(error_log)
    
    # 2. Test failure counting
    failure_analysis = count_test_failures(error_log, issue_data.get("body", ""))
    
    # 3. Regression detection
    regression_analysis = detect_regression(
        issue_data,
        issue_data.get("comments", [])
    )
    
    # 4. Performance analysis
    perf_analysis = analyze_performance_regression(
        error_log,
        issue_data.get("body", "")
    )
    
    # 5. Custom model impact
    model_analysis = detect_custom_model_impact(
        issue_data.get("body", ""),
        issue_data.get("comments", [])
    )
    
    # 6. Stack trace severity
    stack_critical = any(term in stack_trace.lower() for term in [
        "segfault", "crash", "abort", "fatal", "panic"
    ])
    
    # Calculate combined priority scores
    priority_scores = {
        "P0": 0.0,
        "P1": 0.0,
        "P2": 0.0,
        "P3": 0.0
    }
    
    # Error type contribution (40%)
    for pri, score in error_analysis["priority_scores"].items():
        priority_scores[pri] += score * 0.4
    
    # Failure count contribution (30%)
    if failure_analysis["total_failed"] > 20:
        priority_scores["P1"] += 0.3
    elif failure_analysis["total_failed"] > 0:
        priority_scores["P2"] += 0.2
    
    # Regression contribution (20%)
    if regression_analysis["is_regression"]:
        # Regression boosts P0 scoring
        if stack_critical:
            priority_scores["P0"] += 0.3
        else:
            priority_scores["P0"] += 0.2
    
    # Custom model impact (40% - for P0)
    if model_analysis["has_custom_impact"]:
        priority_scores["P0"] += 0.4
    
    # Performance analysis contribution
    if perf_analysis["is_performance_issue"]:
        if perf_analysis["percent_degradation"] > 5:
            priority_scores["P0"] += 0.3
        elif perf_analysis["percent_degradation"] > 1:
            priority_scores["P2"] += 0.2
    
    # Determine final priority
    final_priority = max(priority_scores.items(), key=lambda x: x[1])[0]
    
    # Apply priority boost for regression
    if regression_analysis["is_regression"] and final_priority == "P2":
        final_priority = "P1"  # Upgrade regression to P1 if no other critical factors
    
    # Cap at P0 for custom model impact
    if model_analysis["has_custom_impact"]:
        final_priority = "P0"
    
    # Generate reason
    reasons = []
    if error_analysis["is_critical"]:
        reasons.append("Critical error detected in execution")
    if regression_analysis["is_regression"]:
        reasons.append("Regression: was passing, now failing")
    if failure_analysis["total_failed"] > 20:
        reasons.append(f"Many test failures: {failure_analysis['total_failed']}")
    if model_analysis["has_custom_impact"]:
        reasons.append("Impacts custom production model")
    if perf_analysis["is_performance_issue"]:
        reasons.append(f"Performance regression: {perf_analysis['percent_degradation']}%")
    
    return {
        "final_priority": final_priority,
        "priority_scores": priority_scores,
        "reason": "; ".join(reasons) if reasons else "Standard issue classification",
        "confidence": 0.8,
        "components": {
            "error_analysis": error_analysis,
            "failure_analysis": failure_analysis,
            "regression_analysis": regression_analysis,
            "performance_analysis": perf_analysis,
            "model_analysis": model_analysis
        }
    }
```

---

## Priority Classification Templates

### Template 1: Standard Issue Triage
```python
def triage_standard_issue(title: str, body: str, error_log: str) -> dict:
    """
    Standard priority triage for regular issues.
    """
    
    issue_data = {"title": title, "body": body, "comments": []}
    
    result = analyze_priority(issue_data, error_log, stack_trace="")
    
    return {
        "priority": result["final_priority"],
        "reason": result["reason"],
        "confidence": result["confidence"]
    }
```

### Template 2: Crash/Segfault Triage
```python
def triage_crash_issue(title: str, body: str, error_log: str) -> dict:
    """
    Priority triage specifically for crash issues.
    """
    
    crash_indicators = [
        "segfault", "segmentation fault", "crash",
        "abort", "panic", "fatal"
    ]
    
    body_lower = (title + " " + body).lower()
    
    if any(ind in body_lower for ind in crash_indicators):
        # Give strong P0 signal if crash indicators present
        result = {"final_priority": "P0", "reason": "Crash/Segfault detected"}
    else:
        result = {"final_priority": "P2", "reason": "Standard stack trace analysis"}
    
    return result
```

### Template 3: Regression Triage
```python
def triage_regression_issue(title: str, body: str) -> dict:
    """
    Priority triage for regression issues.
    """
    
    regression = detect_regression({"body": body}, [])
    
    if regression["is_regression"]:
        return {
            "priority": "P1",  # Regression is P1 minimum
            "reason": regression["reason"],
            "confidence": regression["confidence"]
        }
    
    return {"priority": "P3", "reason": "Not a regression"}
```

---

## Usage in Triage Report

### Priority Section Template
```markdown
## Priority Assessment

### Primary Priority
**P1 - Regression** (Confidence: 85%)

### Evidence
- Error Type: Crash (Segmentation fault)
- Test Failures: 25+ cases
- Regression: Yes (was working in previous version)
- Custom Impact: Customer production model affected

### Priority Score Breakdown
| Factor | Score | Weight |
|--------|-------|--------|
| Error Type | 0.8 (Critical) | 40% |
| Test Failures | 0.3 (>20 failed) | 30% |
| Regression | 0.2 (Detected) | 20% |
| Custom Impact | 0.0 | 10% |
| **Total** | **0.85** | 100% |

### Recommendation
Immediate attention required - P1 regression affecting customer deployment.
```

---

## Skill Metadata

- **Version**: 1.0.0
- **Created**: 2026-04-20
- **Requires**: Issue text, error log, stack trace
- **Related Skills**: SKILL_Triage_Logic.md, SKILL_Category_Analysis.md