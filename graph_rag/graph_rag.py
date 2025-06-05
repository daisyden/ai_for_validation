import argparse
import os
import logging
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from nano_graphrag._storage import Neo4jStorage
from sentence_transformers import SentenceTransformer
from github_issue import Github_Issue

import numpy as np

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

system_prompt = """### Task description: You are a helpful, respectful, and honest assistant tasked with determining whether a failed unit test case is a known issue. \
Please refer to the search results from the local knowledge base. All issues retrieved by RAG, regardless of status, if matched, they should be treated as known issues \
If yes, provide the similar issue id, similar issue state, similar issue owner, issue description, issue root cause and solution information. \
If no, return the similar issue id, similar issue state and similar issue owner as N/A and give some insights in issue description, issue root cause and solution. \
Only include information relevant to the question. Do not provide false information if unsure. \
Please generate a valid json for the information collected in English only. \
Please provide details and don't generate unrelated informations not addressed in the prompt. \
Please ensure the generated output is a valid json and without repeated information.
"""


DEEPSEEK_API_KEY = ""
LLAMA_CLOUD_KEY = ""
MODEL = "deepseek-chat"
# DEFAULT_HOST_IP = "http://10.7.180.119:9009/v1"
DEFAULT_HOST_IP = "https://api.deepseek.com"
# DEFAULT_HOST_IP = "http://10.112.100.138:9009/v1"
# MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# os.environ["OPENAI_API_KEY"] = ""


neo4j_config = {
  "neo4j_url": os.environ.get("NEO4J_URL", "neo4j://localhost:7687"),
  "neo4j_auth": (
      os.environ.get("NEO4J_USER", "neo4j"),
      os.environ.get("NEO4J_PASSWORD", "77777777"),
  )
}

async def deepseepk_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY, base_url=DEFAULT_HOST_IP,
    )
    # openai_async_client = OpenAI(
    #     base_url=DEFAULT_HOST_IP,
    #     api_key=DEEPSEEK_API_KEY,
    #     http_client=httpx.Client(trust_env=False)
    # )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
    # response = openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


WORKING_DIR = "./regtest"
DATA_DIR = WORKING_DIR + "/data"
os.makedirs(DATA_DIR, exist_ok=True)

EMBED_MODEL = SentenceTransformer(
    "TencentBAC/Conan-embedding-v1", cache_folder=WORKING_DIR, device="cpu"
)
# We're using Sentence Transformers to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)

def get_issue_report(issue, github_issue):
    context = None
    try:
        import time
        extraction_start = time.time()

        # skip pull requests 
        if issue.pull_request != None:
            print("\n### Drop the issue becuase it is pull request : " + str(issue.number))
            return context

        issue_id = issue.number
        if os.path.isfile(f"{DATA_DIR}/issue_{issue_id}.txt"):
            with open(f"{DATA_DIR}/issue_{issue_id}.txt", 'r') as f:
                context = f.read()
        else:
            # request issue contents
            issue_contents = issue.body if issue.body != None else ""
            issue_contents = github_issue.parse_github_issue_attachment(issue_contents, "./attachment")

            issue_contents = "Content of #" + str(issue.number) + " is : " + issue_contents


            labels = issue.labels
            if len(labels) > 0:
                label = [l.name for l in labels]
            else:
                label = "N/A"

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
            if issue.title is not None:
                issue_info = f"This issue titled '{issue.title}', "
            elif assignee != "":
                issue_info = f"issue id is #{issue_number}, assigned to {assignee}, and has state {state}. "
            elif label != "N/A":
                issue_info += f"Labels are: {label}. \nThe issue description is: "
            context = issue_info + "\n\nThe issue description is: " + issue_contents + "\n\n" + comments_contents
            with open(f"{DATA_DIR}/issue_{issue.number}.txt", 'w') as f:
                f.write(context)
    except Exception as e:
        print("\n### Result:" + str(issue.number) + " failed to extract") 
        print(repr(e))
    
    return context

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphRAG Github Issue Processor")
    parser.add_argument(
        "--output", "-o", type=str, default="results.json",
        help="output file"
    )
    parser.add_argument(
        "--insert", action="store_true",
        help="Insert mode: insert data into the knowledge base"
    )
    args = parser.parse_args()

    repo = "intel/torch-xpu-ops"
    token = ""
    github_issue = Github_Issue(repo, token)
    issues = github_issue.get_issues("all")
    latencies = []

    if args.insert:
        remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
        remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
        remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
        remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
        remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")
    

    rag = GraphRAG(
        graph_storage_cls=Neo4jStorage,
        addon_params=neo4j_config,
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=deepseepk_model_if_cache,
        cheap_model_func=deepseepk_model_if_cache,
        embedding_func=local_embedding,
    )

    with open(args.output, "w") as f:
        for issue in issues:
            context = get_issue_report(issue, github_issue)
            if context is not None:
                print(f"Processing issue #{issue.number} with insert {args.insert}...")
                if args.insert:
                    rag.insert(context)
                else:
                    prompt = system_prompt + "\n\n" + "### the unit test context is:\n" + context
                    result = rag.query(prompt, param=QueryParam(mode="local"))
                    f.write(f"issue_id#{issue.number}: {result}\n")
                    print(result)
