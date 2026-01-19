import argparse
import json
from langgraph.graph import StateGraph, START, END
from jinja2 import Environment, FileSystemLoader

from github_issue import Github_Issue
from guilty_commit import run_in_docker
from common_nodes import stream_graph_updates, classify_depsrag_graph, classify_graph
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
        graph = classify_depsrag_graph()

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
            
            test_case_pattern = r'\n(op_ut|op_extended),([^,\n]+),([^,\n]+)$'
            matches = re.findall(test_case_pattern, body, re.MULTILINE)

            if not matches:
                print("No valid test cases found in the expected format")
                return None
    
            commands = []
            for i, match in enumerate(matches, 1):
                _, module_path, test_method = match
                
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


def build_pytorch_docker_enviroment(download: bool=False, build: str="existing", container: str=""):
    workdir = "/workdir"
    import os
    run_in_docker(f"""
                     bash tools/build.sh {download} {build}
                     """, container)
    
    enviroments = run_in_docker(f"cat {workdir}/enviroments.txt", container)  

    return enviroments


def reproduce_issues_with_docker_commands(folder: str, commands: list, container: str):
    if folder is None or len(folder) == 0:
        print("Please provide working folder.")
        return None
    
    print("### Start to reproduce the issue in docker : " + container)
    for test_file, test_class, test_method, command in commands:
        print("### Testing command: " + command + " in folder: " + folder)
        
        run_in_docker(f"bash /tools/run_one_test.sh {folder} \'{command}\'", container)
        xml_content = run_in_docker(f"cat {folder}/{test_method}.xml", container)
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


def extracte_information_from_log(test_category: str, json_object: object, prompt: str):
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