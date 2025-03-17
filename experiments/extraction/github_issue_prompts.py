
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


class IssueKeys(BaseModel):
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


class IssueExtractionData(BaseModel):
    """Extracted information about the reporter and error message of github issue."""

    issue_keys: List[IssueKeys]


#issue_prompt_parser = PydanticOutputParser(pydantic_object=IssueKeys)

# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
issue_extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at identifying github issue reporter and error message in text. "
            "Only extract important information about the issue. Extract nothing if no important information can be found in the text."
        ),
        ("human", "{text}"),
    ]
)

class CommentsKeys(BaseModel):
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


class CommentsExtractionData(BaseModel):
    """Extracted information about the root cause, dependency and PR of github issue."""

    comments_keys: List[CommentsKeys]


# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
comments_extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at identifying github issue root cause and dependnecy and PR information in text. "
            "Only extract important information about the issue. Extract nothing if no important information can be found in the text.",
        ),
        ("human", "{text}"),
    ]
)


