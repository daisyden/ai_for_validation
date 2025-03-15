../../../RAG/langchain_extract_test2.pyimport re

import requests
from langchain_community.document_loaders import BSHTMLLoader

issue_id = 1437
link = "https://github.com/intel/torch-xpu-ops/issues/" + str(issue_id)
# Download the content
response = requests.get(link)

# Write it to a file
with open("issue.html", "w", encoding="utf-8") as f:
    f.write(response.text)

# Load it with an HTML parser
loader = BSHTMLLoader("issue.html")
document = loader.load()[0]
# Clean up code
# Replace consecutive new lines with a single new line
document.page_content = re.sub("\n\n+", "\n", document.page_content)

from github import Github

# Replace with your GitHub token
GITHUB_TOKEN = ""
# Replace with the repository name
REPO_NAME = "intel/torch-xpu-ops"
# Replace with the issue number
ISSUE_NUMBER = issue_id 

# Authenticate with GitHub
github = Github(GITHUB_TOKEN)

# Get the repository
repo = github.get_repo(REPO_NAME)

# Get the issue
issue = repo.get_issue(number=ISSUE_NUMBER)

# Get the comments
comments = issue.get_comments()


issue_contents = "Content of https://github.com/intel/torch-xpu-ops/issues/1437: " + "{ " + document.page_content + " }"

comments_contents = "Comments of https://github.com/intel/torch-xpu-ops/issues/1437: {\n"

# Print or process the comments
for comment in comments:
    comments_contents = comments_contents + "{"
    comments_contents = comments_contents + f"Author: {comment.user.login}, "
    comments_contents = comments_contents + f" Date: {comment.created_at}, "
    comments_contents = comments_contents + f" Comment: {comment.body}\n"
    comments_contents = comments_contents + "},"
comments_contents = comments_contents + " }"

#print(issue_contents)
#print(comments_contents)

from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field


class KeyDevelopment(BaseModel):
    """Information about the reporter and error message of github issue."""

    reporter: str = Field(
        ..., description="Who is the reporter of the github issue? If none return TBD."
    )
    error_msg: str = Field(
        ..., description="What is the error message for the github issue?"
    )
    evidence: str = Field(
        ...,
        description="Repeat in verbatim the sentence(s) from which the reporter, dependent_components and error message information were extracted",
    )


class ExtractionData(BaseModel):
    """Extracted information about the reporter and error message of github issue."""

    key_developments: List[KeyDevelopment]


# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at identifying github issue reporter and error message in text. "
            "Only extract important information about the issue https://github.com/intel/torch-xpu-ops/issues/1437. Extract nothing if no important information can be found in the text.",
        ),
        ("human", "{text}"),
    ]
)

class CommentsDevelopmemt(BaseModel):
    """Information about the comments of the github issue."""

    root_cause: str = Field(
        ..., description="What is the in-depth reason of this issue? If cannot find the information return NA."
    )
    dependency: str = Field(
        ..., description="What is the dependent component or person on the github issue? If cannot find the information return NA"
    )
    pr: str = Field(
        ..., description="What is the link of pr to fix the issue? If cannot find the information return NA"
    )

    evidence: str = Field(
        ...,
        description="Repeat in verbatim the sentence(s) from which the root cause, dependency and PR information were extracted",
    )


class ExtractionCommentsData(BaseModel):
    """Extracted information about the root cause, dependency and PR of github issue."""

    key_developments: List[CommentsDevelopmemt]


# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
comments_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at identifying github issue root cause and dependnecy and PR information in text. "
            "Only extract important information about the issue https://github.com/intel/torch-xpu-ops/issues/1437. Extract nothing if no important information can be found in the text.",
        ),
        ("human", "{text}"),
    ]
)


import getpass
import os

if not os.environ.get("GROQ_API_KEY"):
  #os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")
  os.environ["GROQ_API_KEY"] = "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"

from langchain.chat_models import init_chat_model

llm = init_chat_model("llama3-8b-8192", model_provider="groq")
#llm = init_chat_model("Deepseek-R1-Distill-Qwen-32b", model_provider="groq")

extractor = prompt | llm.with_structured_output(
    schema=ExtractionData,
    include_raw=False,
)

comments_extractor = comments_prompt | llm.with_structured_output(
    schema=ExtractionCommentsData,
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
    key_developments.extend(extraction.key_developments)

key_developments[:10]


comments_key_developments = []
for extraction in comments_extractions:
    comments_key_developments.extend(extraction.key_developments)

comments_key_developments[:10]


print("================\nissue {}: [ owner: '{}', error_message: '{}', evidence: '{}', dependency: '{}', root_cause: '{}', PR: '{}' ]\n\n".format(link, key_developments[0].reporter, key_developments[0].error_msg, key_developments[0].evidence, comments_key_developments[0].dependency, comments_key_developments[0].root_cause, comments_key_developments[0].pr))



