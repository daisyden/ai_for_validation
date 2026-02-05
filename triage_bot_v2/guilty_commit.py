import json
import ast
import re
from typing import Annotated
from typing_extensions import TypedDict, Tuple

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_community.tools.shell.tool import ShellTool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from utils import run_in_docker
from typing import List, Tuple
import os
import traceback


def get_git_log(
        blame_range: str,
        match: str,
        workdir: str,
        container: str,) -> set:
    """
    Use `git log` to find commits that modified files matching `match`
    in the specified `blame_range`.
    
    Args:
        blame_range: The commit range to blame.
        match: File path or pattern to filter commits (e.g., "aten", "test").
        workdir: The working directory where the git command should be run.
        container: The Docker container in which to run the command.
        
    Returns:
        A set of commit SHAs that modified files matching `match` in the blame range.
    """
    try:
        if len(match.strip()) == 0:
            command = f"sh -c \"cd {workdir} && git --no-pager log {blame_range}\""
        else:
            command = f"sh -c \"cd {workdir} && git --no-pager log {blame_range} --name-only {match}\""
        git_log_output = run_in_docker(command, container, workdir)
        return git_log_output
    except Exception as e:
        print(f"[get_blamed_commits_git_log] Error processing git log: {e}")
        traceback.print_exc()
        return {f"Error processing git log: {e}"}


def get_git_show(commit_sha: str, workdir: str, container: str) -> str:
    """
    Use `git show` to get details of a specific commit.
    
    Args:
        commit_sha: The commit SHA to show.
        workdir: The working directory where the git command should be run.
        container: The Docker container in which to run the command.
        
    Returns:
        The git show output as a string, or an error message.
    """
    full_command = f"sh -c \"cd {workdir} && git --no-pager show {commit_sha} -U10\""
    # Get the git show information
    git_show_output = run_in_docker(full_command, container, workdir)
    
    return git_show_output



################################
# Function parse 
################################
def get_function_line_range(file_path: str, function_name: str, inner_lineno: int) -> Tuple[int, int]:
    """
    Return (start_line, end_line) for the Python function named `function_name`
    in `file_path`. `inner_lineno` must be a line number inside that function.
    Raises ValueError if not found or line not inside the function.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    lines = source.splitlines()
    tree = ast.parse(source, filename=file_path)

    candidates = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name != function_name.split("(", 1)[0].strip():  # crude strip of args
                continue
            # Include decorator lines (if any) when determining function start
            if getattr(node, "decorator_list", None):
                start = min(getattr(d, "lineno", node.lineno) for d in node.decorator_list) or node.lineno
            else:
                start = node.lineno
            end = getattr(node, "end_lineno", None)
            if end is None:
                # Fallback: infer end by scanning source indentation
                def_indent = len(lines[start - 1]) - len(lines[start - 1].lstrip())
                end_guess = start
                for i in range(start, len(lines)):
                    line = lines[i]
                    stripped = line.lstrip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    indent = len(line) - len(stripped)
                    if indent < def_indent and not stripped.startswith("@"):
                        end_guess = i
                        break
                else:
                    end_guess = len(lines)
                end = end_guess
            if inner_lineno != 0 and start <= inner_lineno <= end:
                candidates.append((start, end))

    if not candidates:
        print(f"Warning: Function '{function_name}' containing line {inner_lineno} not found in {file_path}")
        return inner_lineno, inner_lineno
    else:
        # If nested, choose the innermost (smallest span)
        start_line, end_line = min(candidates, key=lambda t: (t[1] - t[0], t[0]))
        if inner_lineno != 0 and not (start_line <= inner_lineno <= end_line):
            raise ValueError(f"Line {inner_lineno} not within function '{function_name}' range {start_line}-{end_line}")
        return start_line, end_line

