import argparse
import json
from langgraph.graph import StateGraph, START, END
from jinja2 import Environment, FileSystemLoader

from github_issue import Github_Issue
from triage_bot_langgraph.guilty_commit import run_in_tmux_wait, stream_graph_updates
from common_nodes import State, classify, stream_graph_updates, classify_depsrag_graph
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

        if "module:xpu" not in labels or "skipped" not in labels or "triaged" not in labels :
            exit(0)
        
        state = issue.state

        if state != "open":
            print("\n### Drop the issue becuase it is not open : " + str(issue.number))
            exit(0) 

        # request issue contents
        issue_contents = issue.body if issue.body != None else ""
        issue_contents = github_issue.parse_github_issue_attachment(issue_contents, "./attachment")
        issue_contents = "Content of #" + str(issue.number) + " is : " + issue_contents
        body = issue_contents


        ###### Define the graph ##########
        graph = classify_depsrag_graph()

        def generate_prompt(link: str, title: str, body: str) -> str:
            env = Environment(loader=FileSystemLoader('prompts'))
            template = env.get_template('extraction__DISABLED_issue.j2')
            return template.render(link=link, title=title, body=body)

        prompt = generate_prompt(link, title, body)
        user_input = prompt
        # collect the failure information
        json_string = stream_graph_updates(user_input, graph)

        return json_string


def build_pytorch_enviroment(build: str, tmux: str):
    run_in_tmux_wait("if [ -d ~/pytorch ]; then echo 'pytorch exists'; else git clone --recursive https://github.com/pytorch/pytorch.git ~/pytorch; fi", tmux)
    if build == "source":
        print("### Start to build pytorch from source in tmux session : " + tmux)
        run_in_tmux_wait("cd ~/pytorch && git pull", tmux)
        run_in_tmux_wait("source ~/env.sh", tmux)
        run_in_tmux_wait("pip install cmake, ninja, pybind11", tmux)
        run_in_tmux_wait("pip install -r requirements.txt", tmux)
        run_in_tmux_wait("cd ~/pytorch && python3 setup.py clean", tmux)
        run_in_tmux_wait("cd ~/pytorch && python3 setup.py develop", tmux)
    elif build == "nightly":
        print("### Start to install pytorch nightly in tmux session : " + tmux)
        run_in_tmux_wait("cd ~/pytorch && git pull", tmux)
        run_in_tmux_wait("pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/xpu", tmux)
    else:
        print("Please provide correct build method, source or nightly.")
        exit(1)
    run_in_tmux_wait("python3 -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.backends.mps.is_available())'", tmux)

def reproduce_issue_in_tmux(input_json: str, tmux: str):
    try:
        python_object = json.loads(input_json)
    except:
        print("Please provide correct input json.")
        exit(1)
    
    reproduce_command = python_object["reproduce_command"]
    if reproduce_command is None or len(reproduce_command) == 0:
        print("No reproduce command found.")
        exit(1)
    print("### Start to reproduce the issue in tmux session : " + tmux)
    run_in_tmux_wait("source ~/env.sh", tmux)
    output = run_in_tmux_wait("cd ~/pytorch && " + reproduce_command, tmux)

    def generate_prompt(output: str, error_message: str) -> str:
        env = Environment(loader=FileSystemLoader('prompts'))
        template = env.get_template('reproduce.j2')
        return template.render(output=output, error_message=error_message)

    prompt = generate_prompt(output, python_object["error_message"])
    user_input = prompt

    ###### Define the graph ##########
    graph_builder = StateGraph(State)
    graph_builder.add_node("classify", classify)
    graph_builder.add_edge(START, "classify")
    graph_builder.add_edge("classify", END)
    graph = graph_builder.compile()
    
    # collect the failure information
    json_string = stream_graph_updates(user_input, graph)

    if json_string is None and len(json_string) == 0:
        print("No response from the model.")
        exit(1)
    
    python_object = json.loads(json_string)
    if python_object['Reproduced'] == True:
        print("### The issue is reproduced.")
        return True
    else:
        print("### The issue is not reproduced.")
        return False
    