#!/usr/bin/env python3
"""
Combined torch-xpu-ops Issue Collection and Classification Script

This script:
1. Collects open issues from intel/torch-xpu-ops GitHub repository
2. Applies enhanced torch-ops extraction for improved classification
"""

import json
import subprocess
import sys
import os
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))

PHASE1_DIR = os.path.join(BASE_DIR, "torch-xpu-ops-issue-collection")
PHASE2_DIR = os.path.join(BASE_DIR, "torch-ops-extraction")
RESULT_DIR = os.environ.get("RESULT_DIR", os.path.join(BASE_DIR, "result"))
DATA_DIR = os.path.join(ROOT_DIR, "issue_traige", "data")


def run_phase1():
    """Phase 1: Collect issues and generate initial Excel"""
    print("=" * 60)
    print("Phase 1: Collecting torch-xpu-ops issues...")
    print("=" * 60)
    
    script_path = os.path.join(PHASE1_DIR, "generate_excel.py")
    
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, cwd=PHASE1_DIR)
    
    if result.returncode != 0:
        print(f"Error running Phase 1: {result.stderr}")
        return False
    
    print(result.stdout)
    return True


def run_phase2():
    """Phase 2: Improve torch-ops classification"""
    print("\n" + "=" * 60)
    print("Phase 2: Improving torch-ops classification...")
    print("=" * 60)
    
    input_file = os.path.join(DATA_DIR, "torch_xpu_ops_issues.xlsx")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found. Run Phase 1 first.")
        return False
    
    script_path = os.path.join(PHASE2_DIR, "extract_torch_ops.py")
    
    result = subprocess.run(
        [sys.executable, script_path, input_file],
        capture_output=True,
        text=True,
        cwd=PHASE2_DIR
    )
    
    if result.returncode != 0:
        print(f"Error running Phase 2: {result.stderr}")
        return False
    
    print(result.stdout)
    return True


def main():
    print("torch-xpu-ops Issue Collection and Classification")
    print("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--phase1":
        success = run_phase1()
    elif len(sys.argv) > 1 and sys.argv[1] == "--phase2":
        success = run_phase2()
    else:
        success1 = run_phase1()
        if not success1:
            print("\nPhase 1 failed. Exiting.")
            sys.exit(1)
        
        success2 = run_phase2()
        if not success2:
            print("\nPhase 2 failed. Exiting.")
            sys.exit(1)
    
    # Copy result to output directory
    input_file = os.path.join(RESULT_DIR, "torch_xpu_ops_issues.xlsx")
    
    if os.path.exists(input_file):
        print(f"\nResult saved to: {input_file}")
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
