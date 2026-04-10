#!/usr/bin/env python3
"""
Guilty Commit Detection Script for PyTorch/XPU Issues

This script finds the guilty commit that caused test failures by analyzing:
1. pytorch/pytorch git history
2. intel/torch-xpu-ops git history

Usage:
    python find_guilty_commit.py --issue-id "2640" --repo "torch-xpu-ops"
    python find_guilty_commit.py --test-file test_ops_xpu.py --test-name "test_compare_cpu_addcmul" --error-message "..."
    python find_guilty_commit.py --excel-file ~/ai_for_validation/opencode/issue_triage/update_test_results/output.xlsx
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# Configuration
PYTORCH_REPO_PATH = os.environ.get("PYTORCH_REPO_PATH", "/home/gta/daisyden/pytorch")
XPU_OPS_REPO_PATH = os.environ.get("XPU_OPS_REPO_PATH", 
    os.path.join(PYTORCH_REPO_PATH, "third_party/torch-xpu-ops"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "~/ai_for_validation/opencode/issue_triage/guilty_commit/output")


class GuiltyCommitFinder:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_info = {}
        self.commit_range = {}
        self.test_type = "eager"  # Default, can be: eager, inductor, dynamo, functorch
        
    def determine_test_type(self, test_file: str, test_name: str) -> str:
        """Determine the test type based on test file and name"""
        test_file_lower = test_file.lower()
        test_name_lower = test_name.lower()
        
        # Check for inductor tests
        if any(k in test_file_lower for k in ["inductor", "compile"]):
            return "inductor"
        
        # Check for dynamo tests
        if any(k in test_file_lower for k in ["dynamo", "compile"]):
            return "dynamo"
        
        # Check for functorch tests
        if any(k in test_file_lower for k in ["functorch", "vmap", "vjp", "grad"]):
            return "functorch"
        
        # Check for eager mode tests (default)
        if any(k in test_file_lower for k in ["test_ops", "test_core", "test_math"]):
            return "eager"
        
        return "eager"  # Default to eager mode
    
    def is_commit_relevant_for_test_type(self, commit_analysis: Dict, test_type: str) -> bool:
        """Check if a commit is relevant for the given test type"""
        msg = commit_analysis.get("message", "").lower()
        files = commit_analysis.get("files_changed", [])
        
        if test_type == "eager":
            # For eager mode tests like test_ops_xpu.py:
            # Include: core operator changes in aten/src/ATen, torch/testing, test/*_ops.py
            # Exclude: purely inductor/dynamo changes (torch/_inductor/, torch/_dynamo/)
            
            # Core file patterns (include these)
            core_patterns = [
                "aten/src/ATen/", "torch/testing/",
                "test/test_ops", "test/test_",
                "PointwiseOps.cpp", "PointwiseOpsKernel.cpp"
            ]
            
            # Dynamo/Inductor only patterns (exclude these if no core files)
            inductor_only_patterns = [
                "torch/_inductor/",
                "torch/_dynamo/decompositions.py",
                "torch/_dynamo/symbolic_trace.py",
                "test/dynamo/",
                "test/inductor/"
            ]
            
            # Check what files are touched
            core_files = [f for f in files if any(p.lower() in f.lower() for p in core_patterns)]
            inductor_only_files = [f for f in files if any(p.lower() in f.lower() for p in inductor_only_patterns)]
            
            # Include if it touches core files
            if core_files:
                return True
            
            # Exclude if only inductor/dynamo files are touched
            if inductor_only_files and not core_files:
                return False
            
            # For commits touching both, make decision based on message
            # If message explicitly mentions inductor/dynamo, might still be relevant for inductor tests
            # But for eager mode, we should be conservative - exclude if unclear
            if inductor_only_files:
                # Check if it's really about core operators vs dynamo decomposition
                if any(p in msg for p in ["decomposition", "lowering", "inductor"]):
                    return False  # Likely dynamo/inductor specific
            
            return True
            
        elif test_type == "inductor":
            # For inductor tests, include all
            return True
            
        elif test_type == "dynamo":
            # For dynamo tests, include dynamo and decomposition changes
            return True
            
        elif test_type == "functorch":
            # For functorch tests, include vmap/vjp related
            return True
            
        return True  # Default include
        
    def extract_info_from_issue(self, issue_id: str, repo: str) -> Dict:
        """Extract test information from issue ID"""
        print(f"Extracting info from {repo} issue #{issue_id}...")
        
        # Fetch issue content using gh CLI
        if repo == "torch-xpu-ops":
            repo_full = "intel/torch-xpu-ops"
        else:
            repo_full = f"pytorch/{repo}"
            
        try:
            result = subprocess.run(
                ["gh", "issue", "view", issue_id, "--repo", repo_full],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                issue_content = result.stdout
                return self._parse_issue_content(issue_content)
            else:
                print(f"Failed to fetch issue: {result.stderr}")
                return {}
        except Exception as e:
            print(f"Error fetching issue: {e}")
            return {}
    
    def _parse_issue_content(self, content: str) -> Dict:
        """Parse issue content to extract test information"""
        info = {
            "test_file": "",
            "test_name": "",
            "test_cases": [],
            "error_message": "",
            "traceback": "",
            "submit_time": datetime.now()
        }
        
        # Extract test cases (pattern: op_ut,third_party.torch-xpu-ops...)
        case_pattern = r'(op_ut|op_extended),(.+?\.\w+),(\w+)'
        matches = re.findall(case_pattern, content)
        for match in matches:
            info["test_cases"].append({
                "type": match[0],
                "test_path": match[1],
                "test_name": match[2]
            })
        
        # Extract error message
        error_pattern = r'(AssertionError|NotImplementedError|RuntimeError|Error):\s*(.+?)(?=\n\n|\n---|$)'
        errors = re.findall(error_pattern, content, re.IGNORECASE | re.DOTALL)
        if errors:
            info["error_message"] = errors[0][1].strip()[:500]
        
        # Extract traceback
        tb_pattern = r'(Traceback|Traceback \(most recent call last\):)(.+?)(?=\n\n|\n===|$)'
        tb_matches = re.findall(tb_pattern, content, re.DOTALL)
        if tb_matches:
            info["traceback"] = tb_matches[0][1].strip()[:1000]
        
        # Try to extract test file from content
        file_pattern = r'test[/_]\w+\.py'
        files = re.findall(file_pattern, content)
        if files:
            info["test_file"] = files[0]
        
        return info
    
    def extract_from_excel(self, excel_path: str) -> List[Dict]:
        """Extract test information from Excel file"""
        try:
            import pandas as pd
            df = pd.read_excel(excel_path)
            # Extract failed tests
            failed_tests = []
            for _, row in df.iterrows():
                if row.get('status_xpu') in ['error', 'failed']:
                    failed_tests.append({
                        "test_path": row.get('name_cuda', ''),
                        "test_name": row.get('name_cuda', '').split(',')[-1].strip(),
                        "error_message": row.get('error_message', ''),
                        "last_status": row.get('last_status_xpu', ''),
                        "current_status": row.get('status_xpu', '')
                    })
            return failed_tests
        except Exception as e:
            print(f"Error reading Excel: {e}")
            return []
    
    def extract_from_manual_input(self, test_file: str, test_name: str, 
                                   error_message: str, traceback: str = "") -> Dict:
        """Parse manual input"""
        traceback_val = traceback[:1000] if traceback else ""
        return {
            "test_file": test_file,
            "test_name": test_name,
            "test_cases": [{"test_path": test_file, "test_name": test_name}],
            "error_message": error_message[:500] if error_message else "",
            "traceback": traceback_val,
            "submit_time": datetime.now()
        }
    
    def determine_commit_range(self, last_good_commit: str = "", 
                                submit_time: str = "", 
                                days_before: int = 3) -> Tuple[str, str]:
        """Determine commit range to search"""
        pytorch_path = Path(PYTORCH_REPO_PATH)
        
        if last_good_commit and last_good_commit != "HEAD":
            # Use last good commit to current
            start_commit = last_good_commit
            end_commit = "HEAD"
        elif submit_time:
            # Use date range
            try:
                submit_date = datetime.strptime(submit_time, "%Y-%m-%d")
                start_date = submit_date - timedelta(days=days_before)
                start_commit = start_date.strftime("%Y-%m-%d")
                end_commit = submit_time
            except:
                start_commit = f"HEAD~100"
                end_commit = "HEAD"
        else:
            # Default: last 2 weeks
            start_commit = "HEAD~50"
            end_commit = "HEAD"
        
        self.commit_range = {
            "start": start_commit,
            "end": end_commit
        }
        print(f"Commit range: {start_commit} to {end_commit}")
        return start_commit, end_commit
    
    def search_pytorch_commits(self, start_commit: str, end_commit: str, 
                                keywords: List[str]) -> List[Dict]:
        """Search pytorch/pytorch git log for relevant commits"""
        print(f"\nSearching pytorch/pytorch commits from {start_commit} to {end_commit}...")
        
        os.chdir(PYTORCH_REPO_PATH)
        
        commits = []
        seen_hashes = set()
        
        # First check current branch, ensure we're on main
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=10
        )
        current_branch = result.stdout.strip()
        if current_branch != "main" and current_branch != "master":
            print(f"Note: Currently on '{current_branch}', searching all branches")
        
        # Method 1: Search by commit message using keywords (on main branch)
        if keywords:
            print(f"Searching for keywords: {keywords}")
            for kw in keywords:
                # Search on main branch specifically
                cmd = ["git", "log", "--oneline", "origin/main", "-30", "--grep", kw]
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    for line in result.stdout.strip().split('\n'):
                        if line and "commit" not in line.lower():
                            parts = line.split()
                            if len(parts) >= 2:
                                commit_hash = parts[0]
                                message = ' '.join(parts[1:])
                                if commit_hash not in seen_hashes:
                                    seen_hashes.add(commit_hash)
                                    commits.append({"hash": commit_hash, "message": message})
                except Exception as e:
                    print(f"Error searching for '{kw}': {e}")
        
        # Method 2: Get recent commits from main branch (for broader search)
        print("Getting recent commits from main branch...")
        cmd = ["git", "log", "--oneline", "origin/main", "-50"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        commit_hash = parts[0]
                        message = ' '.join(parts[1:])
                        if commit_hash not in seen_hashes:
                            seen_hashes.add(commit_hash)
                            commits.append({"hash": commit_hash, "message": message})
        except Exception as e:
            print(f"Error getting recent commits: {e}")
        
        print(f"Found {len(commits)} total commits")
        return commits[:30]  # Limit to 30
    
    def analyze_commit(self, commit_hash: str, repo_path: str) -> Dict:
        """Analyze a specific commit"""
        os.chdir(repo_path)
        
        result = {
            "hash": commit_hash,
            "date": "",
            "author": "",
            "message": "",
            "files_changed": [],
            "addcmul_related": False,
            "test_related": False,
            "opinfo_related": False,
            "inductor_related": False,
            "dynamo_related": False,
            "core_operator_related": False
        }
        
        # Get commit details
        try:
            # Date and author
            date_result = subprocess.run(
                ["git", "log", "-1", "--format=%ci %an", commit_hash],
                capture_output=True, text=True, timeout=10
            )
            if date_result.returncode == 0:
                result["date"] = date_result.stdout.strip()
            
            # Commit message
            msg_result = subprocess.run(
                ["git", "log", "-1", "--format=%s", commit_hash],
                capture_output=True, text=True, timeout=10
            )
            if msg_result.returncode == 0:
                result["message"] = msg_result.stdout.strip()
            
            # Files changed
            files_result = subprocess.run(
                ["git", "show", "--stat", "--name-only", commit_hash],
                capture_output=True, text=True, timeout=10
            )
            if files_result.returncode == 0:
                files = files_result.stdout.strip().split('\n')
                result["files_changed"] = [f for f in files if f and not f.startswith('/')][:20]
                
                # Check for relevant files
                for f in result["files_changed"]:
                    f_lower = f.lower()
                    
                    # Core operator related
                    if any(k in f_lower for k in ["addcmul", "addcdiv", "pointwise", "PointwiseOps"]):
                        result["addcmul_related"] = True
                        result["core_operator_related"] = True
                    
                    # Test related
                    if any(k in f_lower for k in ["test_ops", "test_", "common_methods"]):
                        result["test_related"] = True
                    
                    # OpInfo related
                    if "opinfo" in f_lower or "testing" in f_lower:
                        result["opinfo_related"] = True
                    
                    # Inductor related
                    if any(k in f_lower for k in ["inductor", "triton", " lowering", "lowering.py", "/inductor/"]):
                        result["inductor_related"] = True
                    
                    # Dynamo related
                    if any(k in f_lower for k in ["dynamo", "torch._dynamo", "decomposition"]):
                        result["dynamo_related"] = True
                        
        except Exception as e:
            print(f"Error analyzing commit {commit_hash}: {e}")
        
        return result
    
    def search_torch_xpu_ops(self, start_commit: str, end_commit: str,
                             keywords: List[str]) -> List[Dict]:
        """Search intel/torch-xpu-ops git log"""
        print(f"\nSearching torch-xpu-ops commits...")
        
        xpu_ops_path = Path(XPU_OPS_REPO_PATH)
        if not xpu_ops_path.exists():
            print(f"torch-xpu-ops repo not found at {XPU_OPS_REPO_PATH}")
            return []
            
        os.chdir(XPU_OPS_REPO_PATH)
        
        commits = []
        seen_hashes = set()
        
        # First check if on main branch, if not switch to main
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=10
        )
        current_branch = result.stdout.strip()
        if current_branch != "main":
            print(f"Currently on '{current_branch}', switching to main...")
            subprocess.run(["git", "checkout", "main"], capture_output=True, timeout=30)
        
        # Search by keywords on main branch only
        if keywords:
            print(f"Searching for keywords: {keywords}")
            for kw in keywords:
                cmd = ["git", "log", "--oneline", "origin/main", "-30", "--grep", kw]
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    for line in result.stdout.strip().split('\n'):
                        if line and "commit" not in line.lower():
                            parts = line.split()
                            if len(parts) >= 2:
                                commit_hash = parts[0]
                                message = ' '.join(parts[1:])
                                if commit_hash not in seen_hashes:
                                    seen_hashes.add(commit_hash)
                                    commits.append({"hash": commit_hash, "message": message})
                except Exception as e:
                    print(f"Error searching for '{kw}': {e}")
        
        # Get recent commits on main branch only
        print("Getting recent torch-xpu-ops commits from main...")
        cmd = ["git", "log", "--oneline", "origin/main", "-50"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        commit_hash = parts[0]
                        message = ' '.join(parts[1:])
                        if commit_hash not in seen_hashes:
                            seen_hashes.add(commit_hash)
                            commits.append({"hash": commit_hash, "message": message})
        except Exception as e:
            print(f"Error searching torch-xpu-ops: {e}")
        
        print(f"Found {len(commits)} total commits in torch-xpu-ops main branch")
        return commits[:30]
    
    def generate_report(self, pytorch_commits: List[Dict], 
                        xpu_ops_commits: List[Dict],
                        analysis_results: List[Dict]) -> str:
        """Generate the final report"""
        report = []
        report.append("# Guilty Commit Analysis Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n## Test Information:")
        for key, value in self.test_info.items():
            if value:
                report.append(f"  - {key}: {value}")
        
        # Add test type analysis
        test_type = self.test_info.get("test_type", "eager")
        report.append(f"\n## Test Type Analysis:")
        report.append(f"  - Detected test type: {test_type}")
        if test_type == "eager":
            report.append("  - Note: Excluding purely inductor/dynamo related commits")
            report.append("  - Focus on: core operator implementations, test definitions, OpInfo")
        elif test_type == "inductor":
            report.append("  - Note: Including inductor-specific changes")
        elif test_type == "dynamo":
            report.append("  - Note: Including dynamo and decomposition changes")
        
        report.append(f"\n## Commit Range: {self.commit_range.get('start', 'N/A')} to {self.commit_range.get('end', 'N/A')}")
        
        report.append("\n## pytorch/pytorch Commits")
        if pytorch_commits:
            for c in pytorch_commits[:10]:
                report.append(f"  - {c['hash']}: {c['message']}")
        else:
            report.append("  No relevant commits found")
        
        report.append("\n## intel/torch-xpu-ops Commits")
        if xpu_ops_commits:
            for c in xpu_ops_commits[:10]:
                report.append(f"  - {c['hash']}: {c['message']}")
        else:
            report.append("  No relevant commits found")
        
        report.append("\n## Detailed Analysis")
        for a in analysis_results:
            report.append(f"\n### Commit {a['hash']}")
            report.append(f"Date: {a.get('date', 'N/A')}")
            report.append(f"Message: {a.get('message', 'N/A')}")
            report.append(f"Files changed: {', '.join(a.get('files_changed', [])[:5])}")
            report.append(f"  - core_operator_related: {a.get('core_operator_related', False)}")
            report.append(f"  - test_related: {a.get('test_related', False)}")
            report.append(f"  - opinfo_related: {a.get('opinfo_related', False)}")
            report.append(f"  - inductor_related: {a.get('inductor_related', False)}")
            report.append(f"  - dynamo_related: {a.get('dynamo_related', False)}")
        
        report_text = '\n'.join(report)
        
        # Save report
        report_path = self.output_dir / "guilty_commit_report.md"
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {report_path}")
        
        return report_text


def main():
    parser = argparse.ArgumentParser(description="Find guilty commit for PyTorch/XPU issues")
    
    # Input options
    parser.add_argument("--issue-id", type=str, help="Issue ID from torch-xpu-ops or pytorch")
    parser.add_argument("--repo", type=str, default="torch-xpu-ops", 
                       choices=["torch-xpu-ops", "pytorch"], help="Repository name")
    parser.add_argument("--excel-file", type=str, help="Excel file with test results")
    
    # Manual input
    parser.add_argument("--test-file", type=str, help="Test file name")
    parser.add_argument("--test-name", type=str, help="Test name")
    parser.add_argument("--test-cases", type=str, help="Test cases (comma separated)")
    parser.add_argument("--error-message", type=str, help="Error message")
    parser.add_argument("--traceback", type=str, help="Traceback")
    
    # Commit range options
    parser.add_argument("--last-good-commit", type=str, default="", 
                       help="Last known good commit")
    parser.add_argument("--submit-time", type=str, 
                       help="Issue submit time (YYYY-MM-DD)")
    parser.add_argument("--days-before", type=int, default=3,
                       help="Days before submit time to search")
    
    # Output
    parser.add_argument("--output-dir", type=str, 
                       default=OUTPUT_DIR,
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize finder
    finder = GuiltyCommitFinder(args.output_dir)
    
    # Extract information
    if args.issue_id:
        finder.test_info = finder.extract_info_from_issue(args.issue_id, args.repo)
    elif args.excel_file:
        finder.test_info["failed_tests"] = finder.extract_from_excel(args.excel_file)
    elif args.test_file:
        finder.test_info = finder.extract_from_manual_input(
            args.test_file, args.test_name, args.error_message, args.traceback
        )
    else:
        print("Please provide either --issue-id, --excel-file, or manual input")
        return 1
    
    # Determine test type based on test file and test name
    test_file = finder.test_info.get("test_file", "")
    test_name = finder.test_info.get("test_name", "")
    finder.test_type = finder.determine_test_type(test_file, test_name)
    finder.test_info["test_type"] = finder.test_type
    print(f"\nTest type detected: {finder.test_type}")
    
    # Determine keywords from test info
    keywords = []
    if finder.test_info.get("test_name"):
        # Extract operation name from test name
        test_name = finder.test_info["test_name"]
        if "addcmul" in test_name.lower():
            keywords = ["addcmul", "addcdiv"]
        elif "addcdiv" in test_name.lower():
            keywords = ["addcdiv"]
        elif "conv2d" in test_name.lower():
            keywords = ["conv2d", "conv"]
        elif "pool" in test_name.lower():
            keywords = ["pool", "Pool"]
    
    # Determine commit range
    start_commit, end_commit = finder.determine_commit_range(
        args.last_good_commit, args.submit_time, args.days_before
    )
    
    # Search pytorch/pytorch
    pytorch_commits = finder.search_pytorch_commits(start_commit, end_commit, keywords)
    
    # Search torch-xpu-ops
    xpu_ops_commits = finder.search_torch_xpu_ops(start_commit, end_commit, keywords)
    
    # Analyze top commits and filter by test type relevance
    print(f"\nAnalyzing commits (filtering for {finder.test_type} test type)...")
    analysis_results = []
    filtered_commits = []
    
    for c in (pytorch_commits + xpu_ops_commits)[:10]:
        repo_path = PYTORCH_REPO_PATH if c in pytorch_commits else XPU_OPS_REPO_PATH
        analysis = finder.analyze_commit(c["hash"], repo_path)
        
        # Check relevance for test type
        if finder.is_commit_relevant_for_test_type(analysis, finder.test_type):
            analysis_results.append(analysis)
            filtered_commits.append(c)
        else:
            print(f"  Excluding (not relevant for {finder.test_type}): {c['hash']} {c['message'][:50]}...")
    
    # Generate report
    report = finder.generate_report(pytorch_commits, xpu_ops_commits, analysis_results)
    print("\n" + "="*60)
    print(report)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())