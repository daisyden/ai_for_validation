import argparse
import json
from langgraph.graph import StateGraph, START, END
from jinja2 import Environment, FileSystemLoader

from github_issue import Github_Issue
from guilty_commit import run_in_tmux_wait, retest_with_pytest_hook_in_tmux
from common_nodes import stream_graph_updates, classify_depsrag_graph, classify_graph
import subprocess

######## create tmux session if not exists ########
def tmux_session_exists(session_name):
    """
    Checks if a tmux session with the given name exists.
    """
    try:
        subprocess.run(['tmux', 'has-session', '-t', session_name], 
                       check=True,  # Raise CalledProcessError for non-zero exit codes
                       capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False

def tmux_session_creation(session_name):
    """
    Creates a new tmux session with the given name.
    """
    try:
        subprocess.run(['tmux', 'new-session', '-d', '-s', session_name], 
                       check=True,  # Raise CalledProcessError for non-zero exit codes
                       capture_output=True)
        print(f"Tmux session '{session_name}' created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create tmux session '{session_name}': {e.stderr.decode().strip()}")
        exit(1)

def get_tmux_sessions(tmux: str):
    """
    Returns a list of existing tmux session names.
    """    
    if not tmux_session_exists(tmux):
        print(f"Tmux session '{tmux}' does not exist. Creating it now...")
        tmux_session_creation(tmux)

    run_in_tmux_wait("echo 'Tmux session is ready'", tmux)
    return True


###### Process the issue ##########
def extract_issue_information(rep: str, issue_number: int, token: str):
    if issue_number is None or rep is None or token is None:
        print("Please provide input, repo and token.")
        return None

    github_issue = Github_Issue(rep, token)
    issue = github_issue.get_issue(issue_number)
    link = issue.html_url

    if issue.pull_request != None:
        print("\n### Drop the issue becuase it is pull request : " + str(issue.number))
        exit(0)
    else: 
        title = issue.title if issue.title != None else ""
        labels = [ label.name for label in issue.labels ]
        date_created = issue.created_at 
        date_created = date_created.strftime("%Y-%m-%d")

        if rep == "pytorch/pytorch" and ( "skipped" not in labels or "triaged" not in labels ):
            exit(0)
        
        state = issue.state

        # Disable temperarily
        # if state != "open":
        #     print("\n### Drop the issue becuase it is not open : " + str(issue.number))
        #     exit(0) 

        # request issue contents
        issue_contents = issue.body if issue.body != None else ""
        issue_contents = github_issue.parse_github_issue_attachment(issue_contents, "./attachment")
        issue_contents = "Content of #" + str(issue.number) + " is : " + issue_contents
        body = issue_contents


        ###### Define the graph ##########
        graph = classify_depsrag_graph()

        def generate_prompt(link: str, title: str, body: str, date_created: str) -> str:
            env = Environment(loader=FileSystemLoader('prompts'))
            template = env.get_template('extraction__DISABLED_issue.j2')
            return template.render(link=link, title=title, body=body, date_created=date_created)

        prompt = generate_prompt(link, title, body, date_created)
        user_input = prompt
        # collect the failure information
        json_string = stream_graph_updates(user_input, graph)

        return json_string


def build_pytorch_enviroment(build: str, tmux: str):
    run_in_tmux_wait("export http_proxy=http://proxy.ims.intel.com:911", tmux)
    run_in_tmux_wait("export https_proxy=http://proxy.ims.intel.com:911", tmux)
    run_in_tmux_wait("conda create -n pytorch_guilty_commit python=3.10 -y", tmux)
    run_in_tmux_wait("source ~/miniforge3/bin/activate pytorch_guilty_commit", tmux)
    run_in_tmux_wait("if [ -d ~/pytorch ]; then rm -rf ~/pytorch ; fi", tmux)
    run_in_tmux_wait("git clone https://github.com/pytorch/pytorch.git ~/pytorch", tmux)
    run_in_tmux_wait("if [ -d triton_whl ]; then rm -rf triton_whl ; fi", tmux)
    run_in_tmux_wait("pip download --no-deps --index-url https://download.pytorch.org/whl/nightly/xpu --pre pytorch-triton-xpu --dest tritone_whl", tmux)
    run_in_tmux_wait("pip install tritone_whl/pytorch_triton_xpu-*.whl", tmux)
    if build == "source":
        print("### Start to build pytorch from source in tmux session : " + tmux)
        run_in_tmux_wait("cd ~/pytorch && git pull", tmux)
        run_in_tmux_wait("source ~/env.sh", tmux)
        run_in_tmux_wait("pip install cmake, ninja, pybind11", tmux)
        run_in_tmux_wait("conda install -y libuv", tmux)
        run_in_tmux_wait("pip install -r requirements.txt", tmux)
        run_in_tmux_wait("cd ~/pytorch && python3 setup.py clean", tmux)
        run_in_tmux_wait("cd ~/pytorch && python3 setup.py develop", tmux)
    elif build == "nightly":
        print("### Start to install pytorch nightly in tmux session : " + tmux)
        run_in_tmux_wait("cd ~/pytorch && git pull", tmux)
        run_in_tmux_wait("pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/xpu", tmux)
    else:
        print("Use existing pytorch installation.")

    torch_avail = run_in_tmux_wait("cd ~ ; source ~/env.sh; python3 -c 'import torch; print(torch.__version__); print(torch.xpu.is_available());'", tmux)
    torch_avail = [ line.strip() for line in torch_avail[0].split('\n') if len(line.strip()) > 0 ]
    if "True" in torch_avail:
        return True
    else:
        print("Failed to build pytorch.")
        return False

def reproduce_issue_in_tmux_calltracer(python_object: str, match: str = "test", blame_callee: bool = False):
    called_funcs, log, casecode = retest_with_pytest_hook_in_tmux(python_object,
                                    match=match,
                                    blame_callee=blame_callee)
    return called_funcs, log, casecode

def reproduce_issue_in_tmux(python_object: str, match: str = "test", blame_callee: bool = False):
    issue_inforamtion = python_object
    tmux = issue_inforamtion["tmux"]
    reproduce_command = issue_inforamtion["reproduce_command"]
    if reproduce_command is None or len(reproduce_command) == 0:
        print("No reproduce command found.")
        exit(1)

    called_funcs = [f"{python_object['test_file']}::{python_object['test_class']}::{python_object['test_case']}",]
    casecode = called_funcs[0]
    
    called_funcs, output, casecode = reproduce_issue_in_tmux_calltracer(python_object, match=match, blame_callee=blame_callee)
    # else:
    #     print("### Start to reproduce the issue in tmux session : " + tmux)
    #     run_in_tmux_wait("source ~/env.sh", tmux)
    #     run_in_tmux_wait("pip install -r .ci/docker/requirements-ci.txt", tmux)
    #     if "pytorch/pytorch" not in issue_inforamtion["link"]:
    #         output = run_in_tmux_wait("cd ~/pytorch/third_party/torch-xpu-ops/ && " + reproduce_command + " && cd ~/pytorch", tmux)
    #     else:
    #         output = run_in_tmux_wait("cd ~/pytorch && " + reproduce_command, tmux)

    #if issue_inforamtion["error_message"] is not None and len(issue_inforamtion["error_message"]) > 0:
    def generate_prompt(test_case_name: str, test_case_output: str, error_message: str, called_funcs: list, casecode: str) -> str:
        env = Environment(loader=FileSystemLoader('prompts'))
        template = env.get_template('reproduce.j2')
        return template.render(test_case_name=test_case_name, test_log=test_case_output, error_message=error_message, called_funcs=called_funcs, casecode=casecode)

    prompt = generate_prompt(python_object["test_case"], output, issue_inforamtion["error_message"], called_funcs=called_funcs, casecode=casecode)


    # else:
    #     def generate_prompt(output: str, called_funcs: list, casecode: str) -> str:
    #         env = Environment(loader=FileSystemLoader('prompts'))
    #         template = env.get_template('check_result.j2')
    #         return template.render(test_log=output, called_funcs=called_funcs, casecode=casecode)

    #     prompt = generate_prompt(output, called_funcs=called_funcs, casecode=casecode)

    ###### Define the graph ##########
    user_input = prompt

    graph = classify_graph()
    
    # collect the failure information
    json_string = stream_graph_updates(user_input, graph)

    if json_string is None or len(json_string) == 0:
        print("No response from the model.")
        exit(1)
    
    python_object = json.loads(json_string)
    error_message = python_object.get("reproduced_error_message", "") + '\n' + python_object.get("traceback" , "") + '\n' + python_object.get("captured_stdout", "")
    python_object["error_message"] = error_message
    if python_object['reproduced'] == True:
        print("### The issue is reproduced.")
        return True, python_object
    else:
        print("### The issue is not reproduced.")
        return False, python_object
