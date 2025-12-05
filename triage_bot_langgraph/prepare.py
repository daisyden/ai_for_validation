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
def extract_issue_information(issue_number: int, token: str, repo: str):
    if issue_number is None or repo is None or token is None:
        print("Please provide input, repo and token.")
        return None

    github_issue = Github_Issue(repo, token)
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

        if repo == "pytorch/pytorch" and ( "skipped" not in labels or "triaged" not in labels ):
            exit(0)
        
        state = issue.state

        # Disable temperarily
        if state != "open":
            print("\n### Drop the issue becuase it is not open : " + str(issue.number))
            exit(0) 

        # request issue contents
        issue_contents = issue.body if issue.body != None else ""
        issue_contents = github_issue.parse_github_issue_attachment(issue_contents, "./attachment")
        issue_contents = issue_contents.split("### Versions")[0]
        issue_contents = "Content of #" + str(issue.number) + " is : " + issue_contents
        body = issue_contents


        ###### Define the graph ##########
        graph = classify_depsrag_graph()

        def generate_prompt(link: str, title: str, body: str, date_created: str, repo: str) -> str:
            env = Environment(loader=FileSystemLoader('prompts'))
            template = env.get_template('extraction__DISABLED_issue.j2') if repo == "pytorch/pytorch" else env.get_template('extraction_skipped_issue.j2')
            return template.render(link=link, title=title, body=body, date_created=date_created)

        import pdb
        pdb.set_trace()
        prompt = generate_prompt(link, title, body, date_created, repo)
        user_input = prompt
        # collect the failure information
        json_string = stream_graph_updates(user_input, graph)

        return json_string

def extract_reproduce_commands(issue_number: int, token: str, repo: str):
    if issue_number is None or repo is None or token is None:
        print("Please provide input, repo and token.")
        return None

    github_issue = Github_Issue(repo, token)
    issue = github_issue.get_issue(issue_number)
    link = issue.html_url

    if issue.pull_request != None:
        print("\n### Drop the issue becuase it is pull request : " + str(issue.number))
        exit(0)
    else: 
        labels = [ label.name for label in issue.labels ]
        if "skipped" not in labels:
            print("\n### Drop the issue becuase it is not a skipped issue : " + str(issue.number))
            exit(0)
        
        state = issue.state
        if state != "open":
            print("\n### Drop the issue becuase it is not open : " + str(issue.number))
            exit(0) 

        # request issue contents
        issue_contents = issue.body if issue.body != None else ""
        issue_contents = github_issue.parse_github_issue_attachment(issue_contents, "./attachment")
        issue_contents = issue_contents.split("### Versions")[0]
        issue_contents = "Content of #" + str(issue.number) + " at " + link + " is : " + issue_contents
        body = issue_contents

        import re

        if repo == "intel/torch-xpu-ops":
            # Parse multiple lines of format: op_ut,<test_class>,<test_name>
            test_case_pattern = r'\nop_ut,([^,\n]+),([^,\n]+)$'
            matches = re.findall(test_case_pattern, body, re.MULTILINE)

            if not matches:
                print("No valid test cases found in the expected format")
                return None
    
            commands = []
            for i, match in enumerate(matches, 1):
                module_path, test_method = match
                
                # Could need to handle more corner cases for module path
                module_path = module_path.replace('third_party.torch-xpu-ops.test.xpu.', '')
                test_class = module_path.split('.')[-1]
                module_path = "/".join(module_path.split('.')[:-1]) + ".py"
            
                commands.append([module_path, test_class, test_method, "pytest --junit-xml={test_method}.xml -v -s {module_path} -k {test_method}".format(module_path=module_path, test_method=test_method)])
                print(f"Test Case {i}:")
                print(f"  Module: {module_path}")
                print(f"  Class: {test_class}")
                print(f"  Method: {test_method}")
                print()
            return commands

        if repo == "pytorch/pytorch":
            # Parse URL format: failureCaptures=["module/test_file.py::TestClass::test_method"]
            url_pattern = r'failureCaptures=%5B%22([^%]+(?:%2F[^%]+)*?)%3A%3A([^%]+)%3A%3A([^%]+)%22%5D'
            match = re.search(url_pattern, body)

            if match:
                module_path = match.group(1).replace('%2F', '/')
                test_class = match.group(2)
                test_method = match.group(3)
                
                print(f"Parsed from URL:")
                print(f"  Module: {module_path}")
                print(f"  Class: {test_class}")
                print(f"  Method: {test_method}")
                
                command = f"pytest --junit-xml={test_method}.xml -v -s {module_path} -k {test_method}"
                return [[module_path, test_class, test_method, command],]
            else:
                print("No valid test case found in URL format")
                return None


def build_pytorch_enviroment(download: bool=False, build: str="existing", tmux: str="default", workdir: str="./pytorch"):
    import pdb
    pdb.set_trace()
    import os
    if not os.path.exists(workdir):
        download = True
    
    if download == True:
        run_in_tmux_wait("export http_proxy=http://proxy.ims.intel.com:911", tmux)
        run_in_tmux_wait("export https_proxy=http://proxy.ims.intel.com:911", tmux)
        run_in_tmux_wait("conda remove --name pytorch_guilty_commit --all -y && conda create -n pytorch_guilty_commit python=3.10 -y", tmux)
        run_in_tmux_wait("source ~/miniforge3/bin/activate pytorch_guilty_commit", tmux)
        run_in_tmux_wait(f"if [ -d {workdir} ]; then rm -rf {workdir} ; fi", tmux)
        run_in_tmux_wait(f"git clone https://github.com/pytorch/pytorch.git {workdir}", tmux)        
        run_in_tmux_wait(f"cd {workdir}", tmux)
        run_in_tmux_wait("cd third_party && git clone https://github.com/intel/torch-xpu-ops.git && cd torch-xpu-ops && git rev-parse HEAD >../xpu.txt", tmux)

    run_in_tmux_wait(f"cd {workdir} && pip install -r .ci/docker/requirements-ci.txt && pip install pytest-timeout", tmux)
    run_in_tmux_wait("export PYTORCH_TEST_WITH_SLOW=1", tmux)
    run_in_tmux_wait("source ~/env.sh", tmux)
    run_in_tmux_wait("export PYTEST_ADDOPTS=' -n 1 --timeout 30 --timeout_method=thread '", tmux)

    torch_avail = run_in_tmux_wait("cd ~ ; source ~/env.sh; python3 -c 'import torch; print(torch.__version__); print(torch.xpu.is_available());'", tmux)
    torch_avail = [ line.strip() for line in torch_avail[0].split('\n') if len(line.strip()) > 0 ]

    if "True" in torch_avail:
        print("### Pytorch is already available in tmux session : " + tmux)
    elif build == "existing":
        build = "nightly"
    
    # If build is not source or nightly, use existing pytorch installation and code under tmux    
    if build == "source":
        print("### Start to build pytorch from source in tmux session : " + tmux)
        run_in_tmux_wait(f"cd {workdir} && git pull", tmux)
        run_in_tmux_wait("source ~/env.sh", tmux)
        run_in_tmux_wait("pip install cmake, ninja, pybind11", tmux)
        run_in_tmux_wait("conda install -y libuv", tmux)
        run_in_tmux_wait("pip install -r requirements.txt", tmux)
        run_in_tmux_wait(f"cd {workdir} && python3 setup.py clean", tmux)
        run_in_tmux_wait(f"cd {workdir} && python3 setup.py develop", tmux)
    elif build == "nightly":
        print("### Start to install pytorch nightly in tmux session : " + tmux)
        run_in_tmux_wait(f"cd {workdir} && git pull", tmux)
        run_in_tmux_wait("pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/xpu", tmux)
    
    
    run_in_tmux_wait("if [ -d triton_whl ]; then rm -rf triton_whl ; fi", tmux)
    run_in_tmux_wait("pip download --no-deps --index-url https://download.pytorch.org/whl/nightly/xpu --pre pytorch-triton-xpu --dest tritone_whl", tmux)
    run_in_tmux_wait("pip install tritone_whl/pytorch_triton_xpu-*.whl", tmux)    

    run_in_tmux_wait(f"pushd && cd {workdir} && git log -1 > {workdir}/enviroments.txt && popd", tmux)
    run_in_tmux_wait(f"pushd && cd {workdir}/third_party/torch-xpu-ops && git log -1 >> {workdir}/enviroments.txt && popd", tmux)
    run_in_tmux_wait(f"pip list|grep torch >> {workdir}/enviroments.txt", tmux)
    run_in_tmux_wait(f"dpkg -l |grep intel >> {workdir}/enviroments.txt", tmux)
    run_in_tmux_wait(f"cd ~ ; source ~/env.sh; python3 -c 'import torch; print(torch.__version__); print(torch.xpu.is_available());' >> {workdir}/enviroments.txt", tmux)
    enviroments = run_in_tmux_wait(f"cat {workdir}/enviroments.txt", tmux)
    enviroments = '\n'.join(enviroments[0].replace('\n', '; ').split('my_lock_.*;')[-1].strip().split('; '))   

    return enviroments
    
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
    
    if issue_inforamtion["error_message"] is not None and len(issue_inforamtion["error_message"]) > 0:
        def generate_prompt(test_case_name: str, test_case_output: str, error_message: str, called_funcs: list, casecode: str) -> str:
            env = Environment(loader=FileSystemLoader('prompts'))
            template = env.get_template('reproduce.j2')
            return template.render(test_case_name=test_case_name, test_log=test_case_output, error_message=error_message, called_funcs=called_funcs, casecode=casecode)

        prompt = generate_prompt(python_object["test_case"], output, issue_inforamtion["error_message"], called_funcs=called_funcs, casecode=casecode)
    else:
        def generate_prompt(output: str, called_funcs: list, casecode: str) -> str:
            env = Environment(loader=FileSystemLoader('prompts'))
            template = env.get_template('check_result.j2')
            return template.render(test_log=output, called_funcs=called_funcs, casecode=casecode)

        prompt = generate_prompt(output, called_funcs=called_funcs, casecode=casecode)

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

def reproduce_issues_with_commands(folder: str, commands: list, tmux: str):
    if folder is None or len(folder) == 0:
        print("Please provide working folder.")
        return None
    print("### Start to reproduce the issue in tmux session : " + tmux)
    for test_file, test_class, test_method, command in commands:
        import pdb
        pdb.set_trace()
        print("### Testing command: " + command + " in folder: " + folder)
        run_in_tmux_wait(f"cd {folder} && " + command, tmux)

        import os
        output_xml = folder + '/' + test_method + ".xml"
        with open(output_xml, 'r') as f:
            output = f.read()
            # Parse the pytest XML output to extract test results
            import xml.etree.ElementTree as ET
            try:
                # Extract XML content from output
                xml_start = output.find('<?xml')
                if xml_start == -1:
                    xml_start = output.find('<testsuites')
                
                if xml_start != -1:
                    xml_content = output[xml_start:]
                    # Find the end of XML
                    xml_end = xml_content.rfind('</testsuites>') + len('</testsuites>')
                    xml_content = xml_content[:xml_end]
                    
                    # Parse XML
                    root = ET.fromstring(xml_content)
                    
                    # Find the testcase element
                    testcase = root.find('.//testcase')
                    if testcase is not None:
                        # Check for failure or error
                        failure = testcase.find('failure')
                        error = testcase.find('error')
                        
                        if failure is not None:
                            error_message = failure.get('message', '')
                            error_trace = failure.text or ''
                            print(f"### Test failed: {error_message}")
                            print(f"### Traceback: {error_trace}")
                            return {"status": "failed", 
                                    "test_file": test_file, 
                                    "test_class": test_class,
                                    "test_case": test_method, 
                                    "command": command, 
                                    "error_message": error_message, 
                                    "trace": error_trace, 
                                    }
                        elif error is not None:
                            error_message = error.get('message', '')
                            error_trace = error.text or ''
                            print(f"### Test error: {error_message}")
                            print(f"### Traceback: {error_trace}")
                            return {"status": "failed",
                                    "test_file": test_file,
                                    "test_class": test_class,
                                    "test_case": test_method,
                                    "command": command,
                                    "error_message": error_message,
                                    "trace": error_trace,
                                    }
                        else:
                            print(f"### Passed: {command}")
            except ET.ParseError as e:
                print(f"### Failed to parse XML output: {e}")
            except Exception as e:
                print(f"### Error processing test results: {e}")
        
    return {"status": "passed", "message": "Could not reproduce any error or failure."}

def extracte_information(test_category: str, json_object: object, prompt: str):
    graph = classify_depsrag_graph()

    def generate_prompt(test_category: str, test_file: str, test_case: str, test_class: str, command: str, logs: str, prompt: str) -> str:
        env = Environment(loader=FileSystemLoader('prompts'))
        template = env.get_template(prompt)
        return template.render(test_category=test_category, test_file=test_file, test_case=test_case, test_class=test_class, command=command, logs=logs, prompt=prompt)

    print(json_object)
    prompt = generate_prompt(test_category=test_category,
                             test_file=json_object["test_file"],
                             test_case=json_object["test_case"],
                             test_class=json_object["test_class"],
                             command=json_object["command"],
                             logs=json_object["error_message"] + '\n' + json_object.get("trace", ""),
                             prompt=prompt)
    user_input = prompt
    # collect the failure information
    json_string = stream_graph_updates(user_input, graph)
    import json
    try:
        import pdb
        pdb.set_trace()
        json_object = json.loads(json_string.split("```")[-1])
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        return None
    return json_object