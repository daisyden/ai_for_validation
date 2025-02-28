import os

os.environ["GITHUB_TOKEN"] = ""
github_token = os.environ.get("GITHUB_TOKEN") 

from llama_index.readers.github import (
    GitHubRepositoryIssuesReader,
    GitHubIssuesClient,
)

github_client = GitHubIssuesClient(github_token=github_token)
loader = GitHubRepositoryIssuesReader(
    github_client,
    owner="intel",
    repo="torch-xpu-ops",
    verbose=True,
)

docs = loader.load_data(
    state=GitHubRepositoryIssuesReader.IssueState.ALL,
    labelFilters=[("bug", GitHubRepositoryIssuesReader.FilterType.INCLUDE)]
)

ticket = docs[1].text
import pdb
pdb.set_trace()

prompt_template_str = """\
Here is a Github Issue ticket.

{ticket}

Please extract central themes and output a list of tags.\
"""

from transformers import pipeline

generator = pipeline('text-generation', model='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B')

input_text = f"Here is a Github Issue ticket\n {ticket}\nPlease extract central themes and output it."
print(input_text)

response = generator(input_text[0:900], max_length=1200, num_return_sequences=1)

print(response[0]['generated_text'])
