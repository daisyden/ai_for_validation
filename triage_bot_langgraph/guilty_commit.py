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

from typing import List, Tuple
import os
import traceback


# ################################
# # State
# ################################
# class State(TypedDict):
#     # Messages have the type "list". The `add_messages` function
#     # in the annotation defines how this state key should be updated
#     # (in this case, it appends messages to the list, rather than overwriting them)
#     messages: Annotated[list, add_messages]
#     execution_path: list[str]        

# def get_history(state: State, node: str):
#     executed_nodes = state.get('execution_path')
#     last_node = executed_nodes[-1] if executed_nodes else "START"
    
#     print(f"=== Node {node} executed after {last_node} ====\n")
#     return executed_nodes, last_node 


# ################################
# # Nodes 
# ################################

# def classify(state: State):
#     executed_nodes, last_node = get_history(state, "classify")

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are a helpful assistant that classify pytest faiures."),
#         ("human", "{text}")
#     ])
#     chain = {"text": RunnablePassthrough()} | prompt | llm
#     return {"messages": chain.invoke(state["messages"]),
#             "execution_path": state.get("execution_path", []) + ["classify"]
#     }

# def stream_graph_updates(user_input: str, graph: StateGraph):
#     messages = [HumanMessage(user_input)]

#     for event in graph.stream({"messages": [{"role": "user", "content": user_input}], "execution_path": []}):
#         for value in event.values():
#             if isinstance(value["messages"], list):
#                 json_string = value["messages"][-1].content
#             else:
#                  json_string =value["messages"].content

#             if len(json_string) > 0:
#                 try:
#                     python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))
#                     print(python_object["module_triaged"] + "=========\n" + python_object["reproduce_code"] + "=========\n")
#                     json_string = json.dumps(python_object, indent=4)
#                     print(f"### Assistant output json: {json_string}")
#                     return json_string
#                 except:
#                     print(f"### Assistant output text: {json_string}")
#                     return ""

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


def run_shell_command(command: str) -> str:    
    """
    Execute a shell command and return its output.
    
    Args:
        command: The shell command to execute.
    """
    import subprocess
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e}"

def run_in_tmux_wait(command: str, tmux_session: str) -> str:
    """
    Execute a shell command inside a tmux session, wait for its completion (via tmux wait),
    then capture and return the pane output.
    """
    import subprocess
    try:
        # Send the command to the tmux session:
        # 1. clear the screen
        # 2. run the user command
        # 3. signal completion with 'tmux wait -S my_lock'
        import time
        timestamp = time.time()
        full_command = f"tmux clear-history; clear ; {command} ; tmux wait -S my_lock_{timestamp}"
        subprocess.run(
            ["tmux", "send-keys", "-t", tmux_session, full_command, "C-m"],
            check=True
        )

        # Wait until the in-pane command signals it is done
        subprocess.run(
            ["tmux", "wait", f"my_lock_{timestamp}"],
            check=True
        )

        # Capture the entire pane contents
        result = subprocess.run(
            ["tmux", "capture-pane", "-J", "-S", "-", "-E", "-", "-p", "-t", tmux_session],
            check=True,
            stdout=subprocess.PIPE,
            text=True
        )
        return result.stdout.strip(), timestamp
    except subprocess.CalledProcessError as e:
        return f"Error executing command in tmux: {e}"
    except Exception as e:
        return f"An error occurred: {e}"
    
def get_git_log_by_module_in_tmux(blame_range: str, tmux_session: str, match: str = "aten") -> str:
    """
    Use `git log` to get the commit history since the last good commit.
    
    Args:
        blame_range: The commit range to blame.
        tmux_session: The tmux session name to run the command in.        
    Returns:
        The git log output as a string, or an error message.
    """
    
    full_command = f"git --no-pager log {blame_range} -p --oneline --pretty=format:'%H%nAuthor: %an%nDate: %ad%n%s%n%b%n---END---' {match} "
    return run_in_tmux_wait(full_command, tmux_session)       


###############################
# Collect realted commit:
# 1. commit updated the test function
# 2. commit updated the callee functions of the test function
# 3. commit updated the op_db definition of the test ops
# 4. commit updated the related module
# 5. commit updated the related dependency component
###############################

def retest_with_pytest_hook_in_tmux(
        tmux_session: str,
        original_test_file: str,
        test_class: str,
        test_case: str,
        test_file: str,
        match: str = "test"
    ) -> List[Tuple[str, str, int, int]]:
        """
        Run the specific test inside an existing tmux session (async inside tmux),
        generate called.csv (function call trace), then read and return it as
        a list of (symbol, detail) tuples.

        Returns empty list on failure or if file not produced in time.
        """

        # Pytest command (no redirection needed; we let pytest create log via -s redirected)
        setup_command = (
            f"pushd /tmp/triage_bot_langgraph/call_tracer ; python setup.py develop > /dev/null 2>&1 ; popd ; pip list|grep call-tracer "
        )

        pushd_xpu_ops = ""
        if test_file != original_test_file:
            pushd_xpu_ops = (
                f"pushd ./third_party/torch-xpu-ops ;"
            )

        pytest_command = (
            f"pytest -p call_tracer "
            f"-s {test_file}::{test_class}::{test_case} > log 2>&1 ;"
        )

        greps = ""
        import pdb
        pdb.set_trace
        if "test" in match:
            # Build grep pipelines
            greps = (
                "grep 'at .*\/test\/' log | grep 'CALL:' | grep -v 'CALL: <' | sort | uniq > called.csv ;"
            )
        if "torch" in match:
            # Build grep pipelines
             greps += (
                "grep 'at .*\/torch\/' log | grep 'CALL:' | grep -v 'CALL: <' | sort | uniq >> called.csv ;"
            )

        popd_xpu_ops = ""
        if len(pushd_xpu_ops):
            popd_xpu_ops = (
                f"mv called.csv ../../called.csv ; popd ;"
            )

        full_command = f"{setup_command} ; pwd ; {pushd_xpu_ops} {pytest_command} {greps} {popd_xpu_ops} pwd ; cat called.csv "
        import pdb
        pdb.set_trace()
        output, _ = run_in_tmux_wait(full_command, tmux_session)

        # Parse called.csv
        results: List[Tuple[str, str, int, int]] = []
        try:
            for line in output.split('\n'):
                if not line or not line.startswith('CALL:'):
                    continue
                parts = [p.strip() for p in line.split(' ') if p.strip()]
                if len(parts) >= 4:
                    file = parts[3].split(":")[0]
                    line = int(parts[3].split(":")[1]) if len(parts[3].split(":"))>=2 else 0
                    start, end = get_function_line_range(file, parts[1], line)
                    results.append((parts[1], file, start, end))
        except Exception as e:
            print(f"[retest_with_pytest_hook_in_tmux] Error parsing called.csv (returning partial results): {e}")
            traceback.print_exc()
            return results
        
        print(results)
        return results


def git_blame_in_tmux(file_path: str, function_name: str, start_line: int, end_line: int, blame_range: str, tmux_session: str, since: str) -> list:
    """
    Use `git blame` to find the commit that last modified the specified function.
    
    Args:
        file_path: Path to the Python file.
        function_name: Name of the function.
        start_line: Starting line number of the function.
        end_line: Ending line number of the function.
        blame_range: The commit range to blame.
        tmux: The tmux session name to run the command in.
        
    Returns:
        The commit SHA that last modified the function, or an error message.
    """
    import subprocess
    
    try:
        file_path = file_path.split("site-packages/")[1] if "site-packages" in file_path else file_path
        # Get unique abbreviated (12-char) commit hashes for the blamed lines, then
        # keep only those that are in the blame_range.
        # The previous xargs+grep approach failed when grep found no match (exit codes) or
        # when brace expansion / quoting caused issues; this loop is safer.
        # full_command = (
        #     f"if [ ! -e git_log.csv ]; then git log --no-pager {blame_range} > git_log.csv; fi; "
        #     f"git --no-pager blame HEAD -L {start_line},{end_line} -- {file_path} "
        #     "| awk '{print $1}' "
        #     f"| while read c; do cat git_log.csv | awk -v c=$c '{{if(c ~ $1){{print $1}}}}'; done"
        #     " 2>&1|tee blamed_commits.log"
        # )

        full_command = (
            f"git --no-pager blame {since} HEAD -L {start_line},{end_line} -- {file_path} "
            "| grep -v '^^'"
            "| sort | uniq "
            "| awk '{print $1}' "
        )

        # Get the blame information for the specified line range
        blame_output, timestamp = run_in_tmux_wait(full_command, tmux_session)
        blame_output = blame_output.split(f"tmux wait -S my_lock_{timestamp}")[-1]

        # Parse the output to extract commit SHAs
        commit_list = []
        for line in blame_output.split('\n'):
            line = line.strip()
            if any(c.isalpha() for c in line) and 'Blaming lines' not in line:
                if len(line.split(' ')) != 1:
                    continue

                commit_sha = line.strip()

                def is_commit_hash(commit_str):
                    """Check if string is longer than 12 characters SHA-1 commit hash"""
                    if not commit_str or len(commit_str) < 12:
                        return False
                    import re
                    return bool(re.match(r'^[a-f0-9]{12}$', commit_str, re.IGNORECASE))

                if is_commit_hash(commit_sha) and commit_sha not in commit_list:
                    print(full_command)
                    print("blamed commit_sha: " + commit_sha)
                    commit_list.append(commit_sha)
        return commit_list
    
    except subprocess.CalledProcessError as e:
        return f"Error executing git command: {e}"
    except Exception as e:
        return f"An error occurred: {e}"
    
def git_blame_commit(
    results: list[Tuple[str, str, int, int]],
    blame_range: str,
    tmux_session: str,
    since: str = ""
) -> set:
    """
    For each (symbol, file_path, start_line, end_line) tuple in `results`,
    run `git blame` (via get_blamed_commit) to find the commit that last modified
    that function span.

    Returns:
        Dict mapping commit_sha -> list of tuples:
            [
                (file_path, symbol, start_line, end_line),
                ...
            ]

        Any error or 'No relevant commits' messages are also used as keys so the
        caller can inspect/problem-handle them.
    """
    blamed_commit_list = []

    try:
        for symbol, file_path, start_line, end_line in results:
            commit_sha_list = git_blame_in_tmux(
                file_path=file_path,
                function_name=symbol,
                start_line=start_line,
                end_line=end_line,
                blame_range=blame_range,
                tmux_session=tmux_session,
                since=since
            )

            blamed_commit_list.extend(commit_sha_list)
        
        blamed_commit_set = set(blamed_commit_list)
    except Exception as e:
        blamed_commit_set = {f"Error processing git blame: {e}"}
        traceback.print_exc()
    return blamed_commit_set


def get_blamed_commits_case_update(tmux_session: str,
        failure_info: dict,
        blame_range: str,
        match: str) -> set:
    
    run_shell_command("mkdir -p /tmp/triage_bot_langgraph ; cp -rf call_tracer /tmp/triage_bot_langgraph/.")

    called_functions = retest_with_pytest_hook_in_tmux(
        tmux_session=failure_info["tmux"],
        original_test_file=failure_info["original_test_file"],
        test_class=failure_info["test_class"],
        test_case=failure_info["test_case"],
        test_file=failure_info["test_file"],
        match=match)

    since = ""
    if re.search(r'--since (\d{4}-\d{2}-\d{2})', blame_range) is not None:
        since = re.search(r'--since (\d{4}-\d{2}-\d{2})', blame_range).group(1)

    commits_sha_set = git_blame_commit(called_functions, blame_range, failure_info["tmux"], since=f"--since {since}")

    import pdb
    pdb.set_trace()
    return commits_sha_set, called_functions

def git_show_in_tmux(commit_sha: str, tmux_session: str) -> str:
    """
    Use `git show` to get details of a specific commit.
    
    Args:
        commit_sha: The commit SHA to show.
        tmux: The tmux session name to run the command in.
        
    Returns:
        The git show output as a string, or an error message.
    """
    full_command = f"git --no-pager show {commit_sha} -U10"
    # Get the git show information
    git_show_output, _ = run_in_tmux_wait(full_command, tmux_session)
    
    return git_show_output
    
