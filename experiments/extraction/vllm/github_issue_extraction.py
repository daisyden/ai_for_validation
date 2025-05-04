from openai import OpenAI
import os
import httpx
import requests

from enum import Enum
from pydantic import BaseModel

#from vllm import LLM, SamplingParams
#from vllm.sampling_params import GuidedDecodingParams
from github_issue import Github_Issue 

DEFAULT_HOST_IP = "10.112.100.138"

# initialize client 
client = OpenAI(
    base_url=f"http://{DEFAULT_HOST_IP}:9009/v1",
    api_key="-",
    http_client=httpx.Client(trust_env=False)
)


# Data schema
class IssueDescription(BaseModel):
    issue_number: int 
    issue_description: str
    reporter: str
    assignee: str
    resolution: str
    root_cause: str
    state: str


json_schema = IssueDescription.model_json_schema()

#guided_decoding_params = GuidedDecodingParams(json=json_schema, backend="xgrammar:disable-any-whitespace")
#sampling_params = SamplingParams(guided_decoding=guided_decoding_params, max_tokens=1000, temperature=0, seed=1234)

# Collect all the issues
repo = "intel/torch-xpu-ops"
token = ""
github_issue = Github_Issue(repo, token)
issues = github_issue.get_issues("all")

for issue in issues:
    try:
        if issue.number != 717:
            continue
        # skip pull requests 
        if issue.pull_request != None:
            print("\n### Drop the issue becuase it is pull request : " + str(issue.number))
            continue

        # request issue contents
        issue_contents = issue.body if issue.body != None else ""
        # Remove version information that is with less information and could impact model result
        where_version = issue_contents.find("### Versions")
        if where_version != -1:
            issue_contents = issue_contents[:where_version]
        issue_contents = "Content of #" + str(issue.number) + " is : " + issue_contents

        # request comments content
        comments_page_content = github_issue.get_comments(issue.number)
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

        # In case content of issue or comments are very long
        from langchain_text_splitters import TokenTextSplitter
     
        text_splitter = TokenTextSplitter(
            # Controls the size of each chunk
            chunk_size=4500,
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
                    
                if key in json2.keys():
                    if key in [ "issue_description", "resolution", "root_cuase" ]:
                        value = value + "\n" + json2[key]

                    if (value == "None" or value == "") and \
                    (json2[key] != None and json2[key] != "None" and json2[key] != ""):
                        value = json2[key]

                    merged[key] = value
                else:
                    merged[key] = value 
            return merged

        #################################################################################
        def extract_description(texts):
            output_json = None

            for text in texts:
                prompt = f""" 
                    This is a github issue link https://github.com/{repo}/issues/{issue_number}. 
                    The reporter of the issue is {user}, 
                    and the assignee is {assignee},
                    and the state of the issue is {state}.
                    \nThis is the github issue title {issue.title},
                    and issue body {text}, 
                    Extract the github issue description with error message information from issue tile and issue body,
                    if possible also extract the resolution and root cause information. 
                    \nnPlease generate a valid json for the information collected in English only. Please provide details and don't generate unrelated informations not addressed in the prompt. If the information is not collected succussfully, just return 0 for integer dtype or "" for string dtype as the json value. Please ensure the generated output is a valid json and without repeated information. 
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

            for text in comments_texts:
                print("\n********* comments {}\n".format(text))

                prompt = f""" 
                    This is a github issue link https://github.com/{repo}/issues/{issue_number}. 
                    The reporter of the issue is {user}, 
                    and the assignee is {assignee},
                    and the state of the issue is {state}.
                    \nAnd this is the comments for this github issue {text}, 
                    Extract the resolution and root cause information from it. 
                    \nnPlease generate a json for the information collected in English only. Please provide details and don't generate unrelated informations not addressed in the prompt. If the information is not collected succussfully, just return 0 for integer dtype or "" for string dtype as the json value. Please ensure the generated output is a valid json and without repeated information. 
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

        print("\n#### Merged Results: " + str(issue.number) + json.dumps(output_json))

        with open("results.txt", 'a') as f:
            f.write("### Merged Result:" + str(issue.number) + json.dumps(output_json) + "\n")
 
    except:
        print("\n### Result:" + str(issue.number) + " failed to extract") 
        with open("results.txt", 'a') as f:
            f.write("\n### Result:" + str(issue.number) + " failed to extract\n")

