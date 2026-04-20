# Deep Analysis Patterns - Issue Triage

## 1. Verification Environment Logic

### 1.1 Comprehensive Environment Check
```python
# Full environment verification function
def verify_environment() -> dict:
    """
    Comprehensive environment check for XPU triage.
    
    Returns complete verification report with version info,
    compatibility status, and potential issues.
    """
    
    VERIFICATION_CHECKLIST = {
        # Core PyTorch components
        "torch_version": {
            "check": "python -c \"import torch; print(torch.__version__)\"",
            "parser": parse_pytorch_version,
            "required": True
        },
        
        # XPU-specific components
        "xpu_runtime": {
            "check": "python -c \"import torch; print(torch.xpu.is_available())\"",
            "parser": bool,
            "required": True
        },
        "xpu_driver": {
            "check": "python -c \"import torch; print(torch.xpu.get_device_properties(0).driver_version)\"",
            "parser": str,
            "required": True
        },
        "xpu_memory_total": {
            "check": "python -c \"import torch; print(torch.xpu.get_device_properties(0).total_memory)\"",
            "parser": lambda x: int(x) / (1024**3),  # Convert to GB
            "required": True
        },
        "xpu_device_name": {
            "check": "python -c \"import torch; print(torch.xpu.get_device_properties(0).name)\"",
            "parser": str,
            "required": True
        },
        
        # IGC (Intel Graphics Compiler) version
        "igc_version": {
            "check": "bash: ls -d /opt/intel/compiler*/linux/icc/lib/IntelImplicitIGC{*,*/*} 2>/dev/null || echo IGC path check",
            "parser": parse_igc_version_from_path,
            "required": False,  # May not be available
            "note": "IGC version affects kernel compilation compatibility"
        },
        
        # Compiler toolchain
        "dpcpp_version": {
            "check": "conda list | grep dpcpp-cpp-rt",
            "parser": lambda x: x.split()[1] if len(x.split()) > 1 else "unknown",
            "required": True
        },
        "intel_opencl": {
            "check": "conda list | grep intel-opencl",
            "parser": parse_intel_opencl_version,
            "required": True
        },
        
        # Triton version (critical for SDPA)
        "triton_version": {
            "check": "python -c \"import triton; print(triton.__version__)\"",
            "parser": str,
            "required": True,
            "note": "Version determines SDPA kernel selection path"
        },
        "triton_xpu_version": {
            "check": "python -c \"import triton; print(getattr(triton, 'xpu_backend_version', 'N/A'))\"",
            "parser": str,
            "required": False
        },
        
        # oneAPI/MKL components
        "oneapi_version": {
            "check": "conda list | grep -E 'intel-cmplr|onemkl'",
            "parser": parse_oneapi_packages,
            "required": True
        },
        
        # Memory allocator
        "allocator_config": {
            "check": "python -c \"import torch; print(torch.cuda.memory_allocated() if hasattr(torch.cuda, 'memory_allocated') else 'N/A')\"",
            "parser": format_memory,
            "required": False
        }
    }
    
    return run_environment_checks(VERIFICATION_CHECKLIST)
```

### 1.2 Version Parsing Utilities
```python
import re
from packaging.version import Version

def parse_pytorch_version(version_str: str) -> dict:
    """
    Parse PyTorch version string into components.
    
    Examples:
    - "2.13.0.dev20260419+xpu" -> {major: 2, minor: 13, dev: True, date: 20260419, suffix: "+xpu"}
    - "2.12.0a0+gitd0d73b1" -> {major: 2, minor: 12, nightly: True, commit: "d0d73b1"}
    """
    
    # Pattern for development versions
    dev_pattern = r"(\d+)\.(\d+)\.(\d+)\.?(\w*)dev(\d+)(.*)"
    # Pattern for alpha/beta versions
    ab_pattern = r"(\d+)\.(\d+)\.(\d+)(a|b|rc)(\d+)(.*)"
    # Pattern for release versions
    release_pattern = r"(\d+)\.(\d+)\.(\d+)(.*)"
    
    version_info = {
        "original": version_str,
        "major": 0, "minor": 0, "patch": 0,
        "is_dev": False, "is_nightly": False, "is_release": False,
        "suffix": ""
    }
    
    if match := re.search(dev_pattern, version_str):
        version_info["major"] = int(match.group(1))
        version_info["minor"] = int(match.group(2))
        version_info["patch"] = int(match.group(3))
        version_info["is_dev"] = True
        version_info["dev_date"] = match.group(5)
        version_info["suffix"] = match.group(6)
    elif match := re.search(ab_pattern, version_str):
        version_info["major"] = int(match.group(1))
        version_info["minor"] = int(match.group(2))
        version_info["patch"] = int(match.group(3))
        version_info["release_type"] = match.group(4)
        version_info["release_num"] = match.group(5)
        version_info["suffix"] = match.group(6)
    elif match := re.search(release_pattern, version_str):
        version_info["major"] = int(match.group(1))
        version_info["minor"] = int(match.group(2))
        version_info["patch"] = int(match.group(3))
        version_info["is_release"] = True
        version_info["suffix"] = match.group(4)
    
    return version_info


def parse_igc_version_from_path(path: str) -> dict:
    """
    Extract IGC version from installation path.
    
    Example paths:
    - /opt/intel/compiler/2024.2.0/linux/icc/lib/IntelImplicitIGC
    - /usr/lib/x86_64-linux-gnu/intel-opencl/igc/Version/
    """
    
    version_pattern = r"(\d+\.\d+\.\d+\.\d+)"
    
    if match := re.search(version_pattern, path):
        return {"version": match.group(1), "source": path}
    
    return {"version": "unknown", "source": path}


def parse_driver_version(driver_str: str) -> dict:
    """
    Parse XPU driver version string.
    
    Example: "1.14.36300+8" -> {major: 1, minor: 14, build: 36300, rev: 8}
    """
    
    parts = driver_str.replace("+", ".").split(".")
    if len(parts) >= 2:
        return {
            "driver_string": driver_str,
            "major": int(parts[0]) if parts[0].isdigit() else 0,
            "minor": int(parts[1]) if parts[1].isdigit() else 0,
            "build": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        }
    
    return {"driver_string": driver_str, "major": 0, "minor": 0, "build": 0}


def compare_versions(v1: str, v2: str) -> int:
    """
    Compare two version strings.
    
    Returns: -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
    """
    try:
        p1, p2 = Version(v1), Version(v2)
        if p1 < p2: return -1
        if p1 > p2: return 1
        return 0
    except:
        # Fallback to string comparison
        if v1 < v2: return -1
        if v1 > v2: return 1
        return 0
```

### 1.3 Environment Compatibility Assessment
```python
def assess_triage_compatibility(issue_version_info: dict, env_version_info: dict) -> dict:
    """
    Assess whether current environment can properly triage the issue.
    
    Considerations:
    1. PyTorch version compatibility
    2. Driver version requirements
    3. Private/unreleased branch detection
    """
    
    compatibility = {
        "can_reproduce": True,
        "reason": "",
        "limitations": [],
        "confidence_notes": []
    }
    
    # Check PyTorch version compatibility
    pytorch_issue = issue_version_info.get("pytorch", "")
    pytorch_env = env_version_info.get("pytorch", "")
    
    issue_ver = parse_pytorch_version(pytorch_issue)
    env_ver = parse_pytorch_version(pytorch_env)
    
    # Major version check
    if issue_ver.get("major", 0) > env_ver.get("major", 0):
        compatibility["can_reproduce"] = False
        compatibility["reason"] = "Issue is from newer PyTorch major version"
        compatibility["limitations"].append("Cannot verify fix without code update")
    
    # Dev version date check
    if issue_ver.get("dev_date", "") > env_ver.get("dev_date", ""):
        compatibility["can_reproduce"] = False
        compatibility["reason"] = "Issue is from newer PyTorch development build"
        compatibility["confidence_notes"].append("Test results may not apply to current build")
    
    # Check for private/unreleased branch features
    if issue_version_info.get("is_private_branch", False):
        compatibility["can_reproduce"] = False
        compatibility["reason"] = "Issue is from private/unreleased branch"
        compatibility["confidence_notes"].append("Code analysis only, no execution testing")
    
    # Check driver compatibility
    driver_issue = issue_version_info.get("driver", "")
    driver_env = env_version_info.get("driver", "")
    
    if driver_issue and driver_env:
        issue_driver = parse_driver_version(driver_issue)
        env_driver = parse_driver_version(driver_env)
        
        if issue_driver["build"] > env_driver["build"]:
            compatibility["limitations"].append(
                "Driver version mismatch may affect memory/system behavior"
            )
    
    return compatibility
```

---

## 2. Deep Root Cause Analysis Patterns

### 2.1 Multi-Dimension Analysis Framework
```python
# Analysis dimensions for XPU-related issues
ANALYSIS_DIMENSIONS = {
    "memory_pattern": {
        "indicators": [
            "page fault",
            "segmentation fault",
            "out of memory",
            "allocation failed",
            "cannot allocate"
        ],
        "investigation": {
            "buffer_sizes": "Check tensor sizes against allocation limits",
            "tiling": "Examine tiling/blocking in kernel implementation",
            "fragmentation": "Check for memory fragmentation patterns",
            "alignment": "Verify memory alignment requirements"
        },
        "root_causes": [
            "Large contiguous allocation exceeding GPU VA space",
            "Tile size exceeding hardware limits",
            "Memory leak accumulating over iterations",
            "Alignment violation triggering faults"
        ]
    },
    
    "kernel_implementation": {
        "indicators": [
            "kernel crash",
            "invalid memory access",
            "undefined behavior",
            "longjmp",
            "compiler error"
        ],
        "investigation": {
            "syntax": "Check SYCL/Triton kernel syntax",
            "types": "Verify scalar/vector type usage",
            "bounds": "Check array bounds access",
            "synchronization": "Examine SYCL queue barriers"
        },
        "root_causes": [
            "Race condition in concurrent kernels",
            "Pointer arithmetic errors",
            "Missing synchronization points",
            "Type mismatch in kernel arguments"
        ]
    },
    
    "precision_dtype": {
        "indicators": [
            "precision mismatch",
            "numerical error",
            "nan result",
            "inf result",
            "accuracy"
        ],
        "investigation": {
            "input_dtypes": "Extract and catalog all input tensor dtypes",
            "promotion": "Check for implicit dtype promotion",
            "tolerance": "Compare atol/rtol with upstream CUDA",
            "accumulator": "Check accumulate dtype settings"
        },
        "root_causes": [
            "Mixed precision input without proper casting",
            "Accumulator dtype narrower than computation",
            "Precision loss in intermediate operations",
            "Different reduction order affecting results"
        ]
    },
    
    "api_compatibility": {
        "indicators": [
            "not implemented",
            "invalid argument",
            "attribute error",
            "method not found"
        ],
        "investigation": {
            "signature": "Check API function signature",
            "dispatch": "Examine dispatcher logic",
            "fallback": "Verify fallback path exists"
        },
        "root_causes": [
            "Missing operator implementation for XPU",
            "API parameter mismatch between CPU/XPU paths",
            "Unsupported optional parameter combinations"
        ]
    },
    
    "driver_hardware": {
        "indicators": [
            "gpu aborted",
            "device error",
            "runtime abort",
            "dpcpp error"
        ],
        "investigation": {
            "driver_version": "Check against known driver issues",
            "hardware_limits": "Examine against device capabilities",
            "compiler_bugs": "Check IGC release notes"
        },
        "root_causes": [
            "Driver bug in memory management",
            "Hardware limitation reached",
            "IGC compiler optimization bug"
        ]
    }
}


def perform_deep_analysis(issue_data: dict, execution_result: dict = None) -> dict:
    """
    Execute multi-dimension deep analysis.
    
    Args:
        issue_data: Parsed issue information
        execution_result: Result from reproduce command execution
    
    Returns:
        Comprehensive analysis with:
        - Error classification
        - Root cause hypothesis
        - Confidence level
        - Evidence points
    """
    
    analysis_result = {
        "primary_error_type": "unknown",
        "dimensions": {},
        "root_cause_hypothesis": [],
        "confidence": "medium",
        "evidence": [],
        "investigation_steps": []
    }
    
    # Classify primary error type
    error_msg = ""
    if execution_result:
        error_msg = execution_result.get("error_message", "")
    
    # Match error to analysis dimension
    for dim_name, dim_info in ANALYSIS_DIMENSIONS.items():
        for indicator in dim_info["indicators"]:
            if indicator.lower() in error_msg.lower():
                analysis_result["primary_error_type"] = dim_name
                analysis_result["dimensions"][dim_name] = {
                    "matched_indicator": indicator,
                    "evidence_weight": "high"
                }
                break
    
    # Add investigation based on matched dimension
    if analysis_result["primary_error_type"] in ANALYSIS_DIMENSIONS:
        matched_dim = ANALYSIS_DIMENSIONS[analysis_result["primary_error_type"]]
        
        # Generate investigation steps
        for step_name, step_desc in matched_dim.get("investigation", {}).items():
            analysis_result["investigation_steps"].append({
                "type": step_name,
                "description": step_desc,
                "tool": get_investigation_tool(step_name)
            })
        
        # Add potential root causes
        analysis_result["root_cause_hypothesis"].extend(
            matched_dim.get("root_causes", [])
        )
    
    # If no dim matched, perform generic analysis
    if analysis_result["primary_error_type"] == "unknown":
        analysis_result["generic_analysis"] = generic_error_analysis(error_msg)
    
    return analysis_result
```

### 2.2 Error-Specific Deep Analysis Templates

#### 2.2.1 Segmentation Fault Analysis Template
```python
def analyze_memory_failure(issue_data: dict, error_log: str) -> dict:
    """
    Deep analysis for memory-related failures.
    
    Focus patterns:
    1. Page fault patterns (PDE/PDP/PTE)
    2. Buffer allocation sizes
    3. Memory alignment
    """
    
    analysis = {
        "fault_type": "unknown",
        "fault_address": None,
        "memory_pattern": None,
        "root_causes": [],
        "investigation_files": []
    }
    
    # Extract memory address from error
    hex_pattern = r"0x[0-9a-fA-F]+"
    addresses = re.findall(hex_pattern, error_log)
    
    if addresses:
        analysis["fault_address"] = addresses[0]
    
    # Classify page fault type
    if "PDP" in error_log:
        analysis["fault_type"] = "Page Directory not present"
        analysis["memory_pattern"] = "Large tensor exceeding VA space"
    elif "PTE" in error_log:
        analysis["fault_type"] = "Page Table Entry not present"
        analysis["memory_pattern"] = "Tiled operation out of bounds"
    elif "PDE" in error_log:
        analysis["fault_type"] = "Page Directory Entry not present"
        analysis["memory_pattern"] = "Sub-buffer allocation failure"
    
    # Extract sequence/dimension information
    seq_patterns = [
        r"seq[_-]len.*?[\d,]+",
        r"batch.*?[\d,]+",
        r"heads.*?[\d,]+",
        r"shape.*?\[[\d,\s]+\]"
    ]
    
    for pattern in seq_patterns:
        if match := re.search(pattern, error_log):
            analysis["dimension_info"] = match.group(0)
    
    # Identify likely root causes
    if "seq" in analysis.get("dimension_info", "").lower():
        try:
            seq_len = int(re.search(r"(\d+)", analysis["dimension_info"]).group(1))
            if seq_len > 16384:
                analysis["root_causes"].append(
                    f"Sequence length {seq_len} exceeds typical threshold"
                )
        except:
            pass
    
    return analysis
```


#### 2.2.2 Precision Error Analysis Template
```python
def analyze_precision_error(issue_data: dict, reproduce_cmd: str) -> dict:
    """
    Deep analysis for precision/dtype issues.
    
    Focus patterns:
    1. Input tensor dtypes
    2. Intermediate computation precision
    3. Tolerance comparison with CUDA
    """
    
    analysis = {
        "input_dtypes": [],
        "mixed_precision": False,
        "accumulator_config": {},
        "recommended_tolerance": {},
        "root_causes": []
    }
    
    # Extract all dtype mentions
    dtype_pattern = r"(float\d+|bfloat16|half|double|int\d+)"
    dtypes = re.findall(dtype_pattern, reproduce_cmd.lower())
    
    analysis["input_dtypes"] = list(set(dtypes))
    
    # Check for mixed precision
    if len(set(dtypes)) > 1:
        analysis["mixed_precision"] = True
    
    # Identify potential accumulator issues
    if "float32" in analysis["input_dtypes"] and "float16" in analysis["input_dtypes"]:
        analysis["root_causes"].append("Mixed fp32/fp16 input without explicit casting")
    
    # Recommend tolerances based on operation type
    analysis["recommended_tolerance"] = {
        "fp32": {"atol": 1e-3, "rtol": 1e-3},
        "fp16": {"atol": 1e-2, "rtol": 1e-2},
        "bf16": {"atol": 1e-2, "rtol": 1e-2}
    }
    
    return analysis
```


#### 2.2.3 API/Implementation Issue Analysis Template
```python
def analyze_api_implementation(issue_data: dict) -> dict:
    """
    Deep analysis for API/implementation issues.
    
    Focus patterns:
    1. XPU fallback detection
    2. Implementation coverage
    3. Dispatcher behavior
    """
    
    analysis = {
        "d贵妃_fallback": False,
        "implementation_location": None,
        "dispatcher_path": None,
        "root_causes": []
    }
    
    issue_body = issue_data.get("body", "")
    
    # Check for XPU fallback warnings
    if "fallback" in issue_body.lower() and "xpu" in issue_body.lower():
        analysis["xpu_fallback"] = True
        analysis["root_causes"].append(
            "Operation falls back from XPU to CPU, indicating missing implementation"
        )
    
    # Identify implementation file
    body_lines = issue_body.split("\n")
    for line in body_lines:
        if ".cpp" in line or ".h" in line:
            analysis["possible_file"] = line.strip()
    
    return analysis
```

---

## 3. Implementation Investigation Tools

### 3.1 Operator Search Utilities
```python
def find_operator_implementation(operator_name: str, search_paths: list) -> dict:
    """
    Find operator implementation files across search paths.
    
    Args:
        operator_name: e.g., "scaled_dot_product_attention", "linalg.cond"
        search_paths: List of base paths to search
    
    Returns:
        Dict with file paths and relevance scores
    """
    
    results = {
        "native_files": [],
        "xpu_files": [],
        "test_files": [],
        "kernel_files": []
    }
    
    import glob
    
    for base_path in search_paths:
        # Native implementation
        for pattern in [
            f"{base_path}/**/aten/**/native/{operator_name}*.cpp",
            f"{base_path}/**/native/{operator_name}*.cpp"
        ]:
            results["native_files"].extend(glob.glob(pattern, recursive=True))
        
        # XPU-specific implementation
        for pattern in [
            f"{base_path}/**/xpu/**/{operator_name}*.cpp",
            f"{base_path}/**/xpu/**/Attention*.cpp",
            f"{base_path}/**/xpu/**/sycl/*{operator_name}*.cpp"
        ]:
            results["xpu_files"].extend(glob.glob(pattern, recursive=True))
        
        # Test files
        for pattern in [
            f"{base_path}/**/test/**/{operator_name}*.py",
            f"{base_path}/**/test/test_*.py"
        ]:
            test_matches = glob.glob(pattern, recursive=True)
            results["test_files"].extend(
                [f for f in test_matches if operator_name.replace(".", "_") in f]
            )
    
    return results


def extract_kernel_launch_info(kernel_file: str) -> dict:
    """
    Extract kernel launch parameters and configuration.
    """
    
    with open(kernel_file, 'r') as f:
        content = f.read()
    
    info = {
        "parallel_range": re.findall(r"range.*?(\d+)", content),
        "local_size": re.findall(r"local.*?(\d+)", content),
        "memory_flags": re.findall(r"malloc|allocate|sycl::buffer", content)
    }
    
    return info
```

---

## 4. Report Generation Utilities

### 4.1 Structured Report Builder
```python
def build_triage_report(
    issue_number: int,
    issue_data: dict,
    environment_info: dict,
    analysis_result: dict,
    fix_suggestions: list
) -> str:
    """
    Build formatted triage report.
    """
    
    report = f"""# Triage Report - Issue #{issue_number}

## Issue Summary
{issue_data.get('title', 'N/A')}

---

## 1. Version Information
| Component | Issue Version | Environment Version | Compatible |
|-----------|---------------|---------------------|------------|
"""
    
    # Populate version table
    version_mappings = [
        ("PyTorch", "pytorch", "pytorch"),
        ("XPU Driver", "driver", "driver_version"),
        ("Triton", "triton", "triton_version"),
        ("IGC", "igc", "igc_version")
    ]
    
    for display_name, issue_key, env_key in version_mappings:
        issue_ver = issue_data.get(version_key, "N/A")
        env_ver = environment_info.get(env_key, "N/A")
        compatible = "Yes" if _check_compatible(issue_ver, env_ver) else "No"
        
        report += f"| {display_name} | {issue_ver} | {env_ver} | {compatible} |\n"
    
    report += f"""
## 2. Reproduce Information
**Type**: {analysis_result.get('reproduce_type', 'N/A')}
**Typical Cases**: {', '.join(analysis_result.get('typical_cases', ['N/A']))}

## 3. Root Cause Analysis
**Error Type**: {analysis_result.get('primary_error_type', 'N/A')}
**Confidence**: {analysis_result.get('confidence', 'N/A')}

### Analysis Details
"""
    
    for dimension, details in analysis_result.get('dimensions', {}).items():
        report += f"- **{dimension}**: {details.get('matched_indicator', 'N/A')}\n"
    
    report += """
### Potential Root Causes
"""
    
    for cause in analysis_result.get('root_causes', []):
        report += f"- {cause}\n"
    
    report += """
## 4. Fix Suggestions
"""
    
    for i, suggestion in enumerate(fix_suggestions, 1):
        report += f"{i}. {suggestion}\n"
    
    report += f"""
## 5. Priority Assessment
**Recommended Priority**: {analysis_result.get('recommended_priority', 'P3')}
**Labels**: {', '.join(issue_data.get('labels', []))}

---

*Report generated: {datetime.now().isoformat()}*
"""
    
    return report


def _check_compatible(v1: str, v2: str) -> bool:
    # Simplified compatibility check
    return v1 == v2 or v1 == "N/A" or v2 == "N/A"
```

---

## Quality Checklist

Before completing triage, verify all sections:

### Version Section
- [ ] All components listed with versions
- [ ] Compatibility assessed correctly
- [ ] Private branch status noted

### Reproduce Section  
- [ ] Command/code extracted verbatim
- [ ] Test cases standardized format
- [ ] Typical cases selected appropriately

### Analysis Section
- [ ] Multi-dimension analysis performed
- [ ] Evidence cited from error log
- [ ] Root cause hypothesis justified
- [ ] Investigation files identified

### Fix Section
- [ ] Specific code suggestions provided
- [ ] References to implementation files
- [ ] Regression test suggestions included

### Priority Section
- [ ] Priority aligned with issue severity
- [ ] Labels correctly applied
- [ ] Assignee recommended if applicable