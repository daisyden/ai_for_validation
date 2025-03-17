import os
import getpass
import re
import requests
from langchain_community.document_loaders import BSHTMLLoader
from github_issue import Github_Issue
from github_issue_prompts import IssueKeys, IssueExtractionData, issue_extraction_prompt
from github_issue_prompts import CommentsKeys, CommentsExtractionData, comments_extraction_prompt
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from llm_model import llm_model 
##from groq_llm_model import llm_model



ticket = 1437
repo = "intel/torch-xpu-ops"
token = "" 
link = "https://github.com/intel/" + repo + "/issues/" + str(ticket)

# request issue and comments contents
github_issue = Github_Issue(repo, token)
page_content = github_issue.get(ticket)
issue_contents = "Content of " + link + ": " + "{ " + page_content + " }"
comments_page_content = github_issue.get_comments(ticket)
comments_contents = "Content of " + link + " comments : " + "{ " + comments_page_content + " }"

chat = ChatHuggingFace(llm=llm_model, verbose=True)
chat_with_tools = chat.bind_tools([IssueExtractionData,])
extractor = issue_extraction_prompt | chat_with_tools
#extractor = issue_extraction_prompt | chat.with_structured_output(
#    schema=IssueExtractionData,
#    include_raw=False,
#    #method="json_schema"
#)

###
from langchain_text_splitters import TokenTextSplitter

text_splitter = TokenTextSplitter(
    # Controls the size of each chunk
    chunk_size=200,
    # Controls overlap between chunks
    chunk_overlap=20,
)

texts = text_splitter.split_text(issue_contents)

# Limit just to the first 3 chunks
# so the code can be re-run quickly
first_few = texts[:1]

extractions = extractor.invoke(
    [{"text": text} for text in first_few],
    {"max_concurrency": 5},  # limit the concurrency by passing max concurrency!
)

key_developments = []

print(extractions.tool_calls)

for extraction in extractions:
    key_developments.extend(extraction)

print(key_developments[:10])


#comments_key_developments = []
#for extraction in comments_extractions:
#    comments_key_developments.extend(extraction.comments_keys)
#
#comments_key_developments[:10]
#
#
#print("================\nissue {}: [ owner: '{}', error_message: '{}', evidence: '{}', dependency: '{}', root_cause: '{}', PR: '{}' ]\n\n".format(link, key_developments[0].reporter, key_developments[0].error_msg, key_developments[0].evidence, comments_key_developments[0].dependency, comments_key_developments[0].root_cause, comments_key_developments[0].pr))
#
#
#
