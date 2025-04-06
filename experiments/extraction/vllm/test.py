# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from pydantic import BaseModel

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from github_issue import Github_Issue 

llm = LLM(model="Qwen/Qwen2.5-14B-Instruct", max_model_len=5000)
class IssueDescription(BaseModel):
    issue_number: int 
    issue_descrption: str
    reporter: str
    assignee: str
    resolution: str
    root_cause: str
    state: str


json_schema = IssueDescription.model_json_schema()

guided_decoding_params = GuidedDecodingParams(json=json_schema, backend="xgrammar:disable-any-whitespace")
sampling_params = SamplingParams(guided_decoding=guided_decoding_params, max_tokens=500, temperature=0)

repo = "intel/torch-xpu-ops"
token = ""
github_issue = Github_Issue(repo, token)
issues = github_issue.get_issues()

for issue in issues:
    try:
        # request issue and comments contents
        if issue.pull_request != None:
            print("\n### Drop the issue becuase it is pull request : " + str(issue.number))
            continue

        issue_contents = issue.body if issue.body != None else "None"
        issue_contents = "Content of #" + str(issue.number) + ": " + "{ " + issue_contents + " }"
        comments_page_content = github_issue.get_comments(issue.number)
        if comments_page_content == None:
            comments_page_content = "None"
        comments_contents = "Content of #" + str(issue.number) + " comments : " + "{ " + comments_page_content + " }"
        
        user = str(issue.user.name) if issue.user != None else "None"
        assignee = str(issue.assignee.name) if issue.assignee != None else "None"
        issue_number = issue.number

                      
        from langchain_text_splitters import TokenTextSplitter

        text_splitter = TokenTextSplitter(
            # Controls the size of each chunk
            chunk_size= (5000 - len(comments_contents) - 500),
            # Controls overlap between chunks
            chunk_overlap=100,
        )

        texts = text_splitter.split_text(issue_contents)

        for text in texts:
            prompt = f""" 
                This is a github issue link https://github.com/{repo}/{issue_number}. 
                The reporter of the issue is {user}, 
                and the assignee is {assignee},
                This is the github issue title {issue.title},
                and issue body {text}, 
                \nExtract the github issue description from issue tile and issue body. 
                And this is the comments for this github issue {comments_contents}, 
                \nExtract the resolution and root cause information from it. 
                \nPlease generate a json for the information collected in English only. Please provide details and don't generate unrelated informations not addressed in the prompt.
                """
            outputs = llm.generate(
                    prompts=prompt,
                    sampling_params=sampling_params,
            )

            print("### Result of :" + str(issue.number) + outputs[0].outputs[0].text)
            with open("results.txt", 'a') as f:
                f.write("### Result:" + str(issue.number) + outputs[0].outputs[0].text)
    except:
        print("### Result:" + str(issue.number) + " failed to extract") 
        with open("results.txt", 'a') as f:
            f.write("### Result:" + str(issue.number) + " failed to extract")

