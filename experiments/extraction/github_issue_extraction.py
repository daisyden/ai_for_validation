import re

import requests
from langchain_community.document_loaders import BSHTMLLoader
from github_issue import Github_Issue
from github_issue_prompts import IssueKeys, IssueExtractionData, issue_extraction_prompt
from github_issue_prompts import CommentsKeys, CommentsExtractionData, comments_extraction_prompt

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


import getpass
import os

if not os.environ.get("GROQ_API_KEY"):
  #os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")
  os.environ["GROQ_API_KEY"] = ""

from langchain.chat_models import init_chat_model

llm = init_chat_model("llama3-8b-8192", model_provider="groq")
#llm = init_chat_model("Deepseek-R1-Distill-Qwen-32b", model_provider="groq")

extractor = issue_extraction_prompt | llm.with_structured_output(
    schema=IssueExtractionData,
    include_raw=False,
)

comments_extractor = comments_extraction_prompt | llm.with_structured_output(
    schema=CommentsExtractionData,
    include_raw=False,
)

###
from langchain_text_splitters import TokenTextSplitter

text_splitter = TokenTextSplitter(
    # Controls the size of each chunk
    chunk_size=2000,
    # Controls overlap between chunks
    chunk_overlap=200,
)

#texts = text_splitter.split_text(document.page_content)
texts = text_splitter.split_text(issue_contents)
comments_texts = text_splitter.split_text(comments_contents)

# Limit just to the first 3 chunks
# so the code can be re-run quickly
first_few = texts[:5]
comments_first_few = comments_texts[:5]

#print("#### ", first_few)
extractions = extractor.batch(
    [{"text": text} for text in first_few],
    {"max_concurrency": 5},  # limit the concurrency by passing max concurrency!
)

#print("#### ", comments_first_few)
comments_extractions = comments_extractor.batch(
    [{"text": text} for text in comments_first_few],
    {"max_concurrency": 5},  # limit the concurrency by passing max concurrency!
)
####

key_developments = []

#for extraction in extractions:
for extraction in extractions:
    key_developments.extend(extraction.issue_keys)

key_developments[:10]


comments_key_developments = []
for extraction in comments_extractions:
    comments_key_developments.extend(extraction.comments_keys)

comments_key_developments[:10]


print("================\nissue {}: [ owner: '{}', error_message: '{}', evidence: '{}', dependency: '{}', root_cause: '{}', PR: '{}' ]\n\n".format(link, key_developments[0].reporter, key_developments[0].error_msg, key_developments[0].evidence, comments_key_developments[0].dependency, comments_key_developments[0].root_cause, comments_key_developments[0].pr))



