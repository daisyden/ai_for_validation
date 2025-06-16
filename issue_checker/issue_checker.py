from openai import OpenAI
import os
import httpx
import requests

from enum import Enum
from datetime import date, datetime, time, timedelta
from pydantic import BaseModel

#from vllm import LLM, SamplingParams
#from vllm.sampling_params import GuidedDecodingParams
import sys
from pathlib import Path

# Get the parent directory (one level up)
parent_dir = Path(__file__).resolve().parent.parent

# Add it to Python path
sys.path.append(str(parent_dir))

from github_issue import Github_Issue 
from enum import Enum, IntEnum

#DEFAULT_HOST_IP = "10.112.100.138"
DEFAULT_HOST_IP = "skyrex.jf.intel.com"
#DEFAULT_HOST_IP = "skytrain.jf.intel.com"

# initialize client
client = OpenAI(
    base_url=f"http://{DEFAULT_HOST_IP}:8000/v1",
    api_key="EMPTY",
    http_client=httpx.Client(trust_env=False)
)

class scoreEnum(IntEnum):
    concise = 2
    simple = 1
    nothing = 0

class score(BaseModel):
    score: scoreEnum
    evidence: str

class typeEnum(str, Enum):
    bug = "bug"
    performance = "performance"
    feature = "feature"

class moduleEnum(str, Enum):
    ut = "UT"
    distributions = "distributed"
    quantization = "quant"
    transformers = "transformers"
    core = "Core"
    op_imp = "OP impl"
    dependency = "dependency bug"
    infra = "infra"
    inductor = "inductor"
    torchbench = "torchbench"
    build = "build"
    na = "N/A"


class ReproduceSteps(BaseModel):
    steps:  score
    software_version: score
    platform: score

class module(BaseModel):
    module: moduleEnum
    evidence: str

class updated(BaseModel):
    update:  date
    evidence: str

# Data schema
class IssueDescription(BaseModel):
    issue_number: int
    issue_description:  score
    error_message: score
    reproduce_steps: ReproduceSteps
    reporter: str
    assignee: str
    resolution: score
    root_cause: score
    impact: score
    state: str
    milestone: str
    labeled_module: module
    predicted_module: module
    report_date: updated
    last_update: updated
    issue_type: typeEnum 

json_schema = IssueDescription.model_json_schema()

#guided_decoding_params = GuidedDecodingParams(json=json_schema, backend="xgrammar:disable-any-whitespace")
#sampling_params = SamplingParams(guided_decoding=guided_decoding_params, max_tokens=1000, temperature=0, seed=1234)

# Collect all the issues
repo = "intel/torch-xpu-ops"
token = ""
github_issue = Github_Issue(repo, token)
issues = github_issue.get_issues("open")
latencies = [] 

for issue in issues:
    try:
        import time
        extraction_start = time.time()

        #import pdb
        #pdb.set_trace()

        # skip pull requests 
        if issue.pull_request != None:
            print("\n### Drop the issue becuase it is pull request : " + str(issue.number))
            continue

        labels = [ label.name for label in issue.labels ]
        label = ".".join(labels)

        #import pdb
        #pdb.set_trace()
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
        else:
            comments_contents = "Content of #" + str(issue.number) + " comments are: " + "{ " + comments_page_content + " }"

        # collect other issue information
        user = str(issue.user.login) if issue.user != None else ""
        assignee = str(issue.assignee.login) if issue.assignee != None else ""
        issue_number = issue.number
        state = issue.state
        milestone = "not set"
        try:
            milestone = issue.milestone.title
        except:
            print("No milstone")


        # In case content of issue or comments are very long
        from langchain_text_splitters import TokenTextSplitter

        text_splitter = TokenTextSplitter( # Controls the size of each chunk chunk_size=4500,
            # Controls overlap between chunks
            chunk_overlap=100,
        )

        texts = text_splitter.split_text(issue_contents)
        comments_texts = text_splitter.split_text(comments_contents) if comments_contents != "" else None

        import json
        def merge_json(json1, json2):
            merged = {}
            for item1 in json1.items():
                key = item1[0]
                value = item1[1]

                if key in json2:
                    if key in [ "reproduce_steps" ]:
                        if value['steps']['score'] == 0 and json1["reproduce_steps"]['steps']['score'] != 0:
                            value['steps'] = json1["reproduce_steps"]['steps']
                        if value['software_version']['score'] == 0 and json1["reproduce_steps"]['software_version']['score'] != 0:
                            value['software_version'] = json1["reproduce_steps"]['software_version']
                        if value['platform']['score'] == 0 and json1["reproduce_steps"]['platform']['score'] != 0:
                            value['platform'] = json1["reproduce_steps"]['platform']
                    elif key in [ "issue_description", "error_message", "impact", "resolution", "root_cause" ]:
                        if value['score'] == 0 and json2[key]['score'] != 0:
                            value['score'] = json2[key]['score']
                            value['evidence'] = json2[key]['evidence']
                    elif key in ["last_update"]:
                        date_format = "%Y-%m-%d" 
                        date1 = datetime.strptime(value["update"], date_format)
                        date2 = datetime.strptime(json2[key]["update"], date_format)
                        if date1 < date2:
                            value["update"] = json2[key]["update"]

                    elif (value == None and json2[key] != None and json2[key] != "None" and json2[key] != ""):
                        value = json2[key]

                if key in merged.keys():
                    merged[key] = value
                else:
                    merged.update({key: value})

            return merged

        #################################################################################
        def extract_description(texts):
            output_json = None

            max_split = min(20, len(texts))
            for text in texts[0:max_split]:
                prompt = f""" 
                    This is a github issue link https://github.com/{repo}/issues/{issue_number}. 
                    The reporter of the issue is {user}, 
                    and the assignee is {assignee},
                    and the state of the issue is {state},
                    and the issue is created at {issue.created_at},
                    and the issue is last updated at {last_update},
                    and the milestone is {milestone},
                    and the labels of the issue are {label}.

                    \nThis is the github issue title {issue.title},
                    and issue body {text}. 
                    As an expert of software quality control, please help to extract the issue number, reproter, assignee and state, and check whether the issue have concise information about issue_description, error message, impact of the issue, reproduce steps of the issue including python, pytest or shell commands (for example pytest test*.py or python *.py or shell commands), pytorch version and platform information, provide a score and evidence for each information, 2 is with the information and concise and clear, 1 is with the information but not so clear, and 0 is missing the information. Please also extract the module from issue label and also predict a module for the issue base on the issue description espcially the test case classifications, possible modules include core, transformers, UT, distributions, op_imp, and quantization, dependency, build, transformers, torchbench, inductor, also please provide evidence, if no module can be predicted return na. If possible also score the resolution and root cause information and provide evidence. In order to identify the issue without response for a long time, please also extract the date the issue is created. If the issue is a functionality bug return issue_type as bug, otherwise if it is a performacne bug return performance and if it is a feature request return feature. If the milestone is set return the milestone, otherwise return NA. 
                    \nPlease generate a valid formatted json for the information collected in English only. Please provide details and don't generate unrelated informations not addressed in the prompt. If the information is not collected succussfully, just return 0 for integer dtype or "" for string dtype as the json value. Please ensure the generated output is a valid json and without repeated information.
                    """
                print(prompt)

                completion = client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    extra_body={"guided_json": json_schema, "guided_decoding_backend": "xgrammar:disable-any-whitespace"},
                )

                #outputs[0].outputs[0].text = outputs[0].outputs[0].text.encode('utf-8', 'replace').decode()
                output_text = completion.choices[0].message.content
                print("### Result of each chunck:" + str(issue.number) + output_text)
                if output_json == None:
                    output_json = json.loads(output_text)
                else:
                    json2 = json.loads(output_text)
                    output_json = merge_json(output_json, json2)

            return output_json

        output_json = extract_description(texts)

        print("\n#### Results: " + str(issue.number) + json.dumps(output_json)) 

        #################################################################################
        def extract_comments(comments_texts):
            output_json = None

            max_split = min(20, len(texts))
            for text in comments_texts[0:max_split]:
                print("\n********* comments {}\n".format(text))

                prompt = f""" 
                    This is a github issue link https://github.com/{repo}/issues/{issue_number}. 
                    The reporter of the issue is {user}, 
                    and the assignee is {assignee},
                    and the state of the issue is {state},
                    and the issue is last updated at {last_update}.
                    \nAnd this is the comments for this github issue {text}, 
                    As an expert of software quanlity control, please check whether the comments provided concise information about resolution and root cause. If the information is concise and clear return 2, if has the inforamtion but not so concise return 1, if no information return 0, please also provide evidence. In order to identify the issue without response for a long time, please also extract the last updated date.
                    \nPlease generate a valid formatted json for the information collected in English only. Please provide details and don't generate unrelated informations not addressed in the prompt. If the information is not collected succussfully, just return 0 for integer dtype or "" for string dtype as the json value. Please ensure the generated output is a valid json and without repeated information.
                    """

                completion = client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    extra_body={"guided_json": json_schema, "guided_decoding_backend": "xgrammar:disable-any-whitespace"},
                )

                output_text = completion.choices[0].message.content

                print("### Result of each chunck in comments:" + str(issue.number) + output_text)
                if output_json == None:
                    output_json = json.loads(output_text)
                else:
                    json2 = json.loads(output_text)
                    output_json = merge_json(output_json, json2)

            print("\n#### Results of comments: " + str(issue.number) + json.dumps(output_json))
            return output_json

        if comments_texts != None:
            comments_json = extract_comments(comments_texts)
            output_json = merge_json(output_json, comments_json)
        #################################################################################

        print("\n#### Merged Results: " + str(issue.number) + " " + json.dumps(output_json))

        with open("issue_checker_results.txt", 'a') as f:
            f.write("### Merged Result:" + str(issue.number) + json.dumps(output_json) + "\n")

        extraction_end = time.time()
        latencies.append(extraction_end - extraction_start)
        print("\n\n*** Latency: {} s , avg: {} s\n\n".format(extraction_end - extraction_start, sum(latencies) / len(latencies)))
 
    except Exception as e:
        print("\n### Result:" + str(issue.number) + " failed to extract") 
        print(repr(e))
        with open("issue_checker_results.txt", 'a') as f:
            f.write("\n### Result:" + str(issue.number) + f" failed to extract\n    {repr(e)}\n")

