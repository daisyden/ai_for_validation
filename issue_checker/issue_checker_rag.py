import requests
import os
import csv
import sys
from pathlib import Path

# Get the parent directory (one level up)
parent_dir = Path(__file__).resolve().parent.parent

# Add it to Python path
sys.path.append(str(parent_dir))

from github_issue import Github_Issue

DEFAULT_HOST_IP = "10.7.180.119"


def QnA(request, host_ip=DEFAULT_HOST_IP):
    url = f"http://{host_ip}:8888/v1/chatqna"

    headers = {"Content-Type": "application/json"}

    proxy = os.environ["http_proxy"]
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""

    response = requests.post(url, headers=headers, json=request)

    os.environ["http_proxy"] = proxy 
    os.environ["https_proxy"] = proxy 

    return response

def delete_rag_file(rag_file, host_ip=DEFAULT_HOST_IP):
    url = f"http://{host_ip}:6007/v1/dataprep/delete"
    headers = {"Content-Type": "application/json"}
    payload = {"file_path": rag_file}

    proxy = os.environ["http_proxy"]
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""

    response = requests.post(url, headers=headers, json=payload)

    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy

    return response

def upload_rag_file(rag_file, host_ip=DEFAULT_HOST_IP):
    url = f"http://{host_ip}:6007/v1/dataprep/ingest"

    proxy = os.environ["http_proxy"]
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    with open(rag_file, "rb") as f:
        files = {"files": f}
        response = requests.post(url, files=files)

    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy

    return response


def submit_issue_check_request(file_list_path, issue_number):
    delete_rag_file("all")
    upload_rag_file(file_list_path)

    message = f"Check the information of github issue {issue_number} including issue number, reporter, assignee, state, last update time, issue descriptions, error message, test cases, dtype, hardware, OS, dependency, impact, reproduce steps, root cause and resolution"
    request = {
            "messages": message,
            "stream": False,
            "top_n": 5,
            "max_tokens": 5000,
        }
    
    QnA_response=QnA(request)
    
    print("Status Code:", QnA_response.status_code)
    print("Response Body:", QnA_response.text)

    if QnA_response.status_code==200:
        result=QnA_response.json()["choices"][0]["message"]["content"]
        answer = result.split("</think>")[-1].strip()
        print("### PR " +  str(issue.number) + " Result: " + message + "\n\n" + answer)
        return answer
    return ""



repo = "intel/torch-xpu-ops"
token = ""
github_issue = Github_Issue(repo, token)
issues = github_issue.get_issues("open")
latencies = []

for issue in issues:
    if issue.pull_request != None:
       print("\n### Drop the issue becuase it is pull request : " + str(issue.number))
       continue

    labels = [ label.name for label in issue.labels ]
    label = ".".join(labels)

    if "skipped" in labels or "module: infra" in labels or "enhancement" in labels:
        continue

    last_update = issue.updated_at

    # request issue contents
    issue_contents = issue.body if issue.body != None else ""
    issue_contents = github_issue.parse_github_issue_attachment(issue_contents, "./attachment")

    # Remove version information that is with less information and could impact model result
    # where_version = issue_contents.find("### Versions")
    # if where_version != -1:
    #     issue_contents = issue_contents[:where_version]
    issue_contents = "Content of #" + str(issue.number) + " is : " + issue_contents

    # request comments content
    comments_page_content = github_issue.get_comments(issue.number)
    comments_page_content = github_issue.parse_github_issue_attachment(comments_page_content, "./attachment")
    comments_contents = ""
    if comments_page_content == "":
        print("Issue {} has no comments\n".format(issue.number))
        comments_contents = "No comments"
    else:
        comments_contents = "Content of #" + str(issue.number) + " comments are: " + "{ " + comments_page_content + " }"

    # collect other issue information
    user = str(issue.user.login) if issue.user != None else ""
    assignee = str(issue.assignee.login) if issue.assignee != None else ""
    issue_number = issue.number
    state = issue.state

    with open(f"{issue.number}.txt", "w") as file:
        file.write(f"issue_number: {issue.number}\rreporter: {user}\rassignee: {assignee}\rstate: {state}\rlabel: {label}\rlast_update: {last_update}\rissue_body: {issue_contents}\rcomments: {comments_contents}\r")
        file.close()

    answer = submit_issue_check_request(f"{issue.number}.txt", issue.number)
    if answer != None and answer != "":
        with open(f"rag_report.csv", "a") as file:
            file.write(answer)
            file.write("\n")
        file.close()
