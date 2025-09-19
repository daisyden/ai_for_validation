import json
import ast
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


################################
# State
################################
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    execution_path: list[str]        

def get_history(state: State, node: str):
    executed_nodes = state.get('execution_path')
    last_node = executed_nodes[-1] if executed_nodes else "START"
    
    print(f"=== Node {node} executed after {last_node} ====\n")
    return executed_nodes, last_node 


################################
# Nodes 
################################

def classify(state: State):
    executed_nodes, last_node = get_history(state, "classify")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that classify pytest faiures."),
        ("human", "{text}")
    ])
    chain = {"text": RunnablePassthrough()} | prompt | llm
    return {"messages": chain.invoke(state["messages"]),
            "execution_path": state.get("execution_path", []) + ["classify"]
    }

def stream_graph_updates(user_input: str, graph: StateGraph):
    messages = [HumanMessage(user_input)]

    for event in graph.stream({"messages": [{"role": "user", "content": user_input}], "execution_path": []}):
        for value in event.values():
            if isinstance(value["messages"], list):
                json_string = value["messages"][-1].content
            else:
                 json_string =value["messages"].content

            if len(json_string) > 0:
                try:
                    python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))
                    print(python_object["module_triaged"] + "=========\n" + python_object["reproduce_code"] + "=========\n")
                    json_string = json.dumps(python_object, indent=4)
                    print(f"### Assistant output json: {json_string}")
                    return json_string
                except:
                    print(f"### Assistant output text: {json_string}")
                    return ""

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
            if node.name != function_name:
                continue
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
            if start <= inner_lineno <= end:
                candidates.append((start, end))

    if not candidates:
        raise ValueError(f"Function '{function_name}' containing line {inner_lineno} not found in {file_path}")

    # If nested, choose the innermost (smallest span)
    start_line, end_line = min(candidates, key=lambda t: (t[1] - t[0], t[0]))
    if not (start_line <= inner_lineno <= end_line):
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
    Execute a shell command inside a tmux session and wait for it to complete.
    
    Args:
        command: The shell command to execute.
        tmux_session: The tmux session name to run the command in.
    """
    import subprocess
    try:
        full_command = f"{command} ; tmux wait -S my_lock"
        subprocess.run(
            ["tmux", "send-keys", "-t", tmux_session, full_command, "C-m"],
            check=True
        )
        # Wait for the command to complete
        subprocess.run(
            ["tmux", "wait", "my_lock"],
            check=True
        )
        return "Command executed successfully."
    except subprocess.CalledProcessError as e:
        return f"Error executing command in tmux: {e}"
    except Exception as e:
        return f"An error occurred: {e}"
    
def get_git_log_in_tmux(last_good_commit: str, tmux_session: str) -> str:
    """
    Use `git log` to get the commit history since the last good commit.
    
    Args:
        last_good_commit: The last known good commit (SHA or ref).
        tmux_session: The tmux session name to run the command in.
        
    Returns:
        The git log output as a string, or an error message.
    """
    
    full_command = f"git --no-pager log {last_good_commit}..HEAD --oneline --pretty=format:'%H%nAuthor: %an%nDate: %ad%n%s%n%b%n---END---'"
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
        test_file: str,
        test_class: str,
        test_case: str,
    ) -> List[Tuple[str, str, int, int]]:
        """
        Run the specific test inside an existing tmux session (async inside tmux),
        generate called.csv (function call trace), then read and return it as
        a list of (symbol, detail) tuples.

        Returns empty list on failure or if file not produced in time.
        """

        # Pytest command (no redirection needed; we let pytest create log via -s redirected)
        pytest_command = (
            f"pytest -p triage_bot_langgraph.tools.call_tracer "
            f"-s {test_file}::{test_class}::{test_case} > log 2>&1"
        )

        # Build grep pipelines (fix awk usage; original had quoted $2 / $4)
        grep_test_called = (
            f"grep -F '{test_file}' log | grep 'CALL:' | "
            "awk '{print $2\",\"$4}' | sort | uniq | sed 's/:/,/' > called.csv"
        )
        grep_torch_called = (
            "grep -F 'torch' log | grep 'CALL:' | "
            "awk '{print $2\",\"$4}' | sort | uniq | sed 's/:/,/' >> called.csv"
        )

        full_command = f"{pytest_command} ; {grep_test_called} ; {grep_torch_called}"
        run_in_tmux_wait(full_command, tmux_session)

        target_path = os.path.abspath("called.csv")

        # Parse called.csv
        results: List[Tuple[str, str, int, int]] = []
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                    if len(parts) == 3:
                        start, end = get_function_line_range(parts[1,parts[0], parts[2]])
                        results.append((parts[0], parts[1], start, end))
        except Exception:
            return []

        return results


def git_blame_in_tmux(file_path: str, function_name: str, start_line: int, end_line: int, last_good_commit: str, tmux_session: str) -> set:
    """
    Use `git blame` to find the commit that last modified the specified function.
    
    Args:
        file_path: Path to the Python file.
        function_name: Name of the function.
        start_line: Starting line number of the function.
        end_line: Ending line number of the function.
        last_good_commit: The last known good commit (SHA or ref).
        tmux: The tmux session name to run the command in.
        
    Returns:
        The commit SHA that last modified the function, or an error message.
    """
    import subprocess
    
    try:
        full_command = f"git blame {last_good_commit} -L {start_line},{end_line} -- {file_path}"
        # Get the blame information for the specified line range
        blame_output = run_in_tmux_wait(full_command, tmux_session)
        
        # Parse the output to extract commit SHAs
        commits = set()
        for line in blame_output.splitlines():
            if line:
                commit_sha = line.split()[0]
                commits.add(commit_sha)        
        return commits
    
    except subprocess.CalledProcessError as e:
        return f"Error executing git command: {e}"
    except Exception as e:
        return f"An error occurred: {e}"
    
def git_blame_commit(
    results: list[Tuple[str, str, int, int]],
    last_good_commit: str,
    tmux_session: str
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
    blamed_commit_set: set = ()

    for symbol, file_path, start_line, end_line in results:
        commit_sha_set = git_blame_in_tmux(
            file_path=file_path,
            function_name=symbol,
            start_line=start_line,
            end_line=end_line,
            last_good_commit=last_good_commit,
            tmux_session=tmux_session
        )
        blamed_commit_set.update(commit_sha_set)

    return blamed_commit_set

def get_blamed_commit_of_module(depency: str, module: str, last_good_commit: str, tmux_session: str) -> set:
    """
    Use `git blame` to find commits that modified files related to the dependency or module.
    Args:
        dependency: The related dependency component.
        module: The related module.
        last_good_commit: The last known good commit (SHA or ref).
        tmux: The tmux session name to run the command in.
    Returns:
        A set of commit SHAs that modified files related to the dependency or module, or an error message since last_good_commit.
    """
    import subprocess
    
    try:
        
        full_command = "find ./aten ./torch ./third_party -name '*{depency}*|*{module}*' -exec sh -c 'git blame {last_good_commit} \"{{}}\"' \;"
        # Get the blame information for the specified line range
        blame_output = subprocess.check_output(
            ["tmux", "send-keys", "-t", tmux_session, full_command, "tmux wait -S my_lock", "C-m", "tmux wait my_lock", f"tmux capture-pane -S - -E - -p -t {tmux_session}"],
            text=True
        )
        
        # Parse the output to extract commit SHAs
        commits = set()
        for line in blame_output.splitlines():
            if line:
                commit_sha = line.split()[0]
                commits.add(commit_sha)        
        return commits
    
    except subprocess.CalledProcessError as e:
        return f"Error executing git command: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

def get_blamed_commits_case_update(tmux_session: str,
        failure_info: dict,
        last_good_commit: str) -> set:
    import pdb
    pdb.set_trace()
    called_functions = retest_with_pytest_hook_in_tmux(
        tmux_session=failure_info["tmux"],
        test_file=failure_info["original_test_file"],
        test_class=failure_info["original_test_class"],
        test_case=failure_info["original_test_case"],)
    
    commits_sha_set = git_blame_commit(called_functions, last_good_commit, failure_info["tmux"])
    return commits_sha_set, called_functions
    
def get_blamed_commits_module_update(tmux_session: str,
        failure_info: dict,
        last_good_commit: str) -> set:
    commits_sha_set = get_blamed_commit_of_module(failure_info["dependency"], failure_info["module"], last_good_commit, failure_info["tmux"])
    return commits_sha_set
 


def git_show_in_tmux(commit_sha: str, tmux_session: str) -> str:
    """
    Use `git show` to get details of a specific commit.
    
    Args:
        commit_sha: The commit SHA to show.
        tmux: The tmux session name to run the command in.
        
    Returns:
        The git show output as a string, or an error message.
    """
    full_command = f"git show {commit_sha} --pretty=format:'%H%nAuthor: %an%nDate: %ad%n%s%n%b%n---END---' --stat"
    # Get the git show information
    git_show_output = run_in_tmux_wait(full_command, tmux_session)
    
    return git_show_output
    
   