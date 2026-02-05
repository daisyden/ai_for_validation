import argparse
import json
from langgraph.graph import StateGraph, START, END
from jinja2 import Environment, FileSystemLoader

from github_issue import Github_Issue
from guilty_commit import run_in_docker
from common_nodes import stream_graph_updates, depsrag_graph, document_analysis_graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage
from vllm_service import llm
import subprocess

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
        graph = depsrag_graph()

        def generate_prompt(link: str, title: str, body: str, date_created: str, repo: str) -> str:
            env = Environment(loader=FileSystemLoader('prompts'))
            template = env.get_template('extraction__DISABLED_issue.j2') if repo == "pytorch/pytorch" else env.get_template('extraction_skipped_issue.j2')
            return template.render(link=link, title=title, body=body, date_created=date_created)

        prompt = generate_prompt(link, title, body, date_created, repo)
        user_input = prompt
        # collect the failure information
        
        json_string = stream_graph_updates(user_input, graph)

        python_object = json.loads(json_string)
        return python_object

def get_issue_body(issue_number: int, token: str, repo: str):
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

        return body

def extract_commit_range(user_input: str):
    def gen_prompt() -> str:
        env = Environment(loader=FileSystemLoader('prompts'))
        template = env.get_template('extract_commit_range.j2')
        return template.render()
    prompt_text = gen_prompt()
    prompt = ChatPromptTemplate.from_messages([
                                            ("system", prompt_text),
                                            ("human", "{text}")
                                            ])
    chain = {"text": RunnablePassthrough()} | prompt | llm
    json_string = chain.invoke(user_input).content

    try:
        python_object = json.loads(json_string)
        since = python_object.get("since", None)
        until = python_object.get("until", None)
    except:
        print(f"### Failed to parse commit range output: {json_string}")
        return "", ""
    
    return since, until
    
    
    
def extract_reproduce_commands(user_input: str):    
    body = user_input

    import re        
    # Parse multiple lines of format: op_ut,<test_class>,<test_name>
    start = False
    matches = []
    
    
    for line in body.splitlines():
        print(line)
        if start == False and line.strip().endswith("Cases:"):
            start = True
        elif start == True:
            test_case_pattern = r'(op_ut|op_extended),([^,\n]+),([^,\n]+)'
            _matches = re.findall(test_case_pattern, line)
            if not _matches:
                test_case_pattern = r'(op_ut|op_extended),([^,\n]+)'
                _matches = re.findall(test_case_pattern, line)
                if not _matches:
                    start = False
                    break
                else:
                    _matches.append("")
            matches.extend(_matches)

    commands = []
    if not matches:
        def generate_prompt() -> str:
            env = Environment(loader=FileSystemLoader('prompts'))
            template = env.get_template('extract_issue_list.j2')
            return template.render()
        generate_prompt_text = generate_prompt()
        prompt = ChatPromptTemplate.from_messages([
                                            ("system", generate_prompt_text),
                                            ("human", "{text}")
                                            ])
        chain = {"text": RunnablePassthrough()} | prompt | llm
    
        message = chain.invoke(AIMessage(content=body))
        print("### LLM output for test case extraction: " + message.content)

        matches = [ [line.split(',')[0], line.split(',')[1], line.split(',')[2]] for line in message.content.split('\n') if line.strip()]
        for i, match in enumerate(matches, 1):
            commands.append([match[0], match[1], match[2], "pytest --junit-xml={test_method}.xml -v -s {module_path} -k {test_method}".format(module_path=match[0], test_method=match[2])])
            print(f"LLM Extracted Test Case {i}: {match}")
            print(f" Module: {match[0]}")
            print(f" Class: {match[1]}")
            print(f" Method: {match[2]}")
        return commands
    
    for i, match in enumerate(matches, 1):
        _, module_path, test_method = match
        
        if "third_party.torch-xpu-ops.test.xpu." not in module_path:
            print(f"Skipping invalid module path in test case {i}: {module_path}")
            continue
        
        # Could need to handle more corner cases for module path
        module_path = module_path.replace('third_party.torch-xpu-ops.test.xpu.', '')
        test_class = module_path.split('.')[-1]
        module_path = "/".join(module_path.split('.')[:-1]) + ".py"            
        
        commands.append([module_path, test_class, test_method, "pytest --junit-xml={test_method}.xml -v -s {module_path} -k {test_method}".format(module_path=module_path, test_method=test_method)])
        
        print(f"Test Case {i}:")
        print(f"  Module: {module_path}")
        print(f"  Class: {test_class}")
        print(f"  Method: {test_method}")
    
    import pdb
    pdb.set_trace()
    return commands


def group_failed_test_cases(issue_list: list):
    def generate_prompt():
        env = Environment(loader=FileSystemLoader('.'))
        prompt_template = "./prompts/group_failed_test_cases.j2"
        template = env.get_template(prompt_template)
        prompt = template.render()
        return prompt

    issue_info_str = "\n".join([json.dumps(issue_info) for issue_info in issue_list])      
    
    prompt_text = generate_prompt()
    
    prompt = ChatPromptTemplate.from_messages([
                                            ("system", prompt_text),
                                            ("human", "{text}")
                                            ])
    chain = {"text": RunnablePassthrough()} | prompt | llm
    
    message = chain.invoke(AIMessage(content=issue_info_str))

    return message.content



def build_pytorch_docker_enviroment(download: bool=False, build: str="nightly", container: str="guilty_commit"):
    workdir = "/workdir"
    import os
    run_in_docker(f"""
                     bash /tools/build.sh {download} {build}
                     """, container, workdir)
    
    enviroments = run_in_docker(f"cat {workdir}/enviroments.txt", container, workdir)  

    return enviroments


def reproduce_issues_with_docker_commands(commands: list, container: str, onlyone=False):    
    status_info = []
    
    print("### Start to reproduce the issue in docker : " + container)
    for test_file, test_class, test_method, command in commands:
        workdir = "/workdir"
        folder = f"{workdir}/pytorch/third_party/torch-xpu-ops/test/xpu" if "_xpu.py" in test_file else f"{workdir}/pytorch/test"
        if folder is None or len(folder) == 0:
            print("Please provide working folder.")
            return None
    
        print("### Testing command: " + command + " in folder: " + folder)
        
        run_in_docker(f"bash /tools/run_one_test.sh {folder} \'{command}\'", container, workdir=folder)
        xml_content = run_in_docker(f"cat {folder}/{test_method}.xml", container, workdir=folder)
        import os
        if len(xml_content) != 0:
            # Parse the pytest XML output to extract test results
            import xml.etree.ElementTree as ET
            try:
                # Extract XML content from output
                xml_start = xml_content.find('<?xml')
                if xml_start == -1:
                    xml_start = xml_content.find('<testsuites')
                
                if xml_start != -1:
                    xml_content = xml_content[xml_start:]
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
                        skipped = testcase.find('skipped')
                        
                        if failure is not None:
                            error_message = failure.get('message', '')
                            error_trace = failure.text or ''
                            print(f"### Test failed: {error_message}")
                            print(f"### Traceback: {error_trace}")
                            status_info.append({"status": "failed", 
                                    "test_file": test_file, 
                                    "test_class": test_class,
                                    "test_case": test_method, 
                                    "reproduce_command": command, 
                                    "error_message": error_message, 
                                    "traceback": error_trace, 
                                    "workdir": workdir
                                    })
                        elif error is not None:
                            error_message = error.get('message', '')
                            error_trace = error.text or ''
                            print(f"### Test error: {error_message}")
                            print(f"### Traceback: {error_trace}")
                            
                            status_info.append({"status": "failed",
                                    "test_file": test_file,
                                    "test_class": test_class,
                                    "test_case": test_method,
                                    "reproduce_command": command,
                                    "error_message": error_message,
                                    "traceback": error_trace,
                                    "workdir": workdir
                                    })
                        elif skipped is not None:
                            error_message = skipped.get('message', '')
                            print(f"### Test skipped.")
                            status_info.append({"status": "skipped",
                                    "test_file": test_file,
                                    "test_class": test_class,
                                    "test_case": test_method,
                                    "reproduce_command": command,
                                    "error_message": error_message,
                                    "traceback": "",
                                    "workdir": workdir
                                    })
                        else:
                            status_info.append({"status": "passed",
                                    "test_file": test_file,
                                    "test_class": test_class,
                                    "test_case": test_method,
                                    "reproduce_command": command,
                                    "error_message": "passed",
                                    "traceback": "",
                                    "workdir": workdir
                                    })
                        if onlyone:
                            break
                    else:
                        status_info.append({"status": "non-detected",
                                    "test_file": test_file,
                                    "test_class": test_class,
                                    "test_case": test_method,
                                    "reproduce_command": command,
                                    "error_message": "non-detected",
                                    "traceback": "",
                                    "workdir": workdir
                                    })
                        print("### No testcase element found in XML output.")
            except ET.ParseError as e:
                print(f"### Failed to parse XML output: {e}")
            except Exception as e:
                print(f"### Error processing test results: {e}")
    
    return status_info
    

def extract_torch_test_details(json_object: object, prompt: str):
    graph = depsrag_graph()

    def generate_prompt(json_object: dict, prompt: str) -> str:
        env = Environment(loader=FileSystemLoader('prompts'))
        template = env.get_template(prompt)
        return template.render(test_category=json_object["test_category"],
                               test_file=json_object["test_file"],
                               test_case=json_object["test_case"],
                               test_class=json_object["test_class"],
                               error_message=json_object["error_message"],
                               traceback=json_object.get("traceback", ""),
                               reproduce_command=json_object['reproduce_command'],
                               )

    prompt = generate_prompt(json_object=json_object,
                             prompt=prompt)
    user_input = prompt

    # collect the failure information
    print("input: " + user_input) 
    import pdb
    pdb.set_trace()   
    json_string = stream_graph_updates(user_input, graph)
    
    return json.loads(json_string)