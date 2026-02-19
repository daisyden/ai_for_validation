import argparse
import json
from jinja2 import Environment, FileSystemLoader
from guilty_commit import get_git_show, get_git_log
from common_nodes import State, stream_graph_updates, document_analysis_graph
import time


parser = argparse.ArgumentParser()
parser.add_argument("--issue_json", type=str, help="Path to JSON file containing issue information (alternative to --issue/--repo/--token)", required=False)
parser.add_argument("--container", type=str, help="The container name where the enviroment is located", default="guilty_commit")
parser.add_argument("--match", type=str, help="The pattern to match in git log", default="test")
parser.add_argument("--since", type=str, help="The start time for git blame, e.g., '2023-10-01'", required=False)
parser.add_argument("--until", type=str, help="The end time for git blame, e.g., '2023-10-31'", required=False) 
parser.add_argument("--blame_range", type=str, help="The git blame range, e.g., 'commit1..commit2'", required=False)
parser.add_argument("--blame_callee", action="store_true", default=False, help="If set, use call tracer to trace the function calls and git blame callee functions.", required=False)
parser.add_argument("--add-ops", type=str, help="Additional ops to consider", required=False)
parser.add_argument("--rag", type=bool, help="Whether to use git show rag", required=False)

args = parser.parse_args()

start = time.time()


try:
    issue_information = json.load(open(args.issue_json, 'r'))
except:
    print(f"Failed to load issue information from {args.issue_json}")
    exit(1)

container = issue_information.get("container", "guilty_commit")
workdir = issue_information.get("workdir", "/workdir/pytorch")
test_file = issue_information.get("test_file", None)
original_test_file = issue_information.get("original_test_file", None)
test_class = issue_information.get("test_class", None)
test_case = issue_information.get("test_case", None)
error_message = issue_information.get("error_message", None)
trace = issue_information.get("trace", None)
last_good_commit = issue_information.get("last_good_commit", None)
since = issue_information.get("since", None) if args.since is None else args.since
until = issue_information.get("until", None) if args.until is None else args.until
match = args.match

if args.blame_range is not None:
    blame_range = args.blame_range
else:
    import re
    import datetime
    def is_date(s):
        """Check if a string is a valid date in common formats."""
        if s is None:
            return False
        
        # Common date formats to check
        date_formats = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{4}/\d{2}/\d{2}$',  # YYYY/MM/DD
            r'^\d{2}-\d{2}-\d{4}$',  # DD-MM-YYYY
            r'^\d{2}/\d{2}/\d{4}$',  # DD/MM/YYYY
        ]
        
        # Check format match
        for pattern in date_formats:
            if re.match(pattern, s):
                try:
                    # Try parsing with common formats
                    for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y']:
                        try:
                            datetime.strptime(s, fmt)
                            return True
                        except ValueError:
                            continue
                except:
                    pass
        
        return False
    
    blame_range = f" --since {since}" if is_date(since) else " -n 100"
    if '-n' in blame_range:
        blame_range = f" --until {until}" if is_date(until) else ""
    else:
        blame_range = f" --until {until}" if is_date(until) else " -n 100"

    if blame_range == "":
        if since is not None and len(since) > 0 and since != "unknown" and until is not None and len(until) > 0 and until != "unknown":
            blame_range = f"{since}..{until}"
        elif until is not None and len(until) > 0 and until != "unknown":
            blame_range = f"{until}~200..{until}"
        else:
            blame_range = "HEAD~200..HEAD"


######## Find guilty commit ###########
# Prase git log to get possible guilty commits
# Parse the git show of each candidate commit to confirm if it is the guilty commit
# Git blame each callee function to get more candidate commits if needed
#######################################
graph = document_analysis_graph()


if "third_party" in workdir:    
    workdir = f"{workdir}".replace("third_party/torch-xpu-ops/test/xpu", "")


def git_log_analysis(): 
    import re
    from datetime import datetime

    git_log_output = get_git_log(blame_range, match, f"{workdir}", container)
    def generate_prompt_git_log(git_log: str, python_object: dict) -> str:
        env = Environment(loader=FileSystemLoader('prompts'))
        template = env.get_template('guilty_commit_gitlog.j2')
        # use original test file because we want to check pytorch git log
        return template.render(git_log=git_log,
                            test_file=python_object["original_test_file"],
                            test_case=python_object["original_test_case"],
                            torch_op=python_object["torch_op"],
                            module=python_object["module"],
                            dependency=python_object["dependency"],
                            error_message=python_object["error_message"])
    prompt = generate_prompt_git_log(git_log=git_log_output, python_object=failure_info)
    user_input = prompt
    json_string = stream_graph_updates(user_input, graph)

    python_object = json.loads(json_string.replace("```json\n","").replace("```", ""))

    blamed_commits_git_log = []
    for key in ["test_case_update", "related_components_update", "error_message_related_update"]:
        if python_object.get(key) not in [False, "false"]:
            blamed_commits_git_log += [ v.strip() for v in python_object.get(key).split(",")]
    blamed_commits_git_log = list(set(blamed_commits_git_log))
    print(f"Potential blamed commits from git log: {blamed_commits_git_log}")

    with open(f"./results/guilty_commit.log", "w") as f:
        f.write(f"\nPotential blamed commits from git log in {workdir}: {blamed_commits_git_log}\n The git log used:\n {git_log_output}\n")

    blamed_commits = []
    called_functions = []
    # if args.blame_callee == True:
    #     blamed_commits, called_functions = get_blamed_commits_case_update(failure_info, blame_range, match)
    #     called_functions = list(set([func[0] for func in called_functions]))
    #     print(f"Potential blamed commits from git blame and call tracer: {blamed_commits}.")            

    def generate_prompt_gitshow(commit_sha: str, git_show: str, git_log: str, python_object: dict, called_functions) -> str:
        env = Environment(loader=FileSystemLoader('prompts'))
        template = env.get_template('guilty_commit_gitshow.j2')
        # use original test file because we want to check pytorch git log
        return template.render(commit_sha=commit_sha, 
                            git_show=git_show,
                            git_log=git_log,
                            original_test_file=python_object["original_test_file"],
                            original_test_case=python_object["original_test_case"],
                            original_test_class=python_object["original_test_class"],
                            called_functions=called_functions,
                            torch_op=python_object["torch_op"],
                            module=python_object["module"],
                            dependency=python_object["dependency"],
                            error_message=python_object["error_message"])
    
    for commit in set(list(blamed_commits) + list(blamed_commits_git_log)):
        print(f"Check blamed commit: {commit}.")

        try:
            show_results = get_git_show(commit, f"{workdir}", issue_information["container"])
            
            prompt = generate_prompt_gitshow(commit_sha=commit, 
                                    git_show=show_results,
                                    git_log=git_log_output,
                                    python_object=failure_info,
                                    called_functions=",".join(called_functions) if len(called_functions) > 0 else "")
            user_input = prompt
            
            json_string = stream_graph_updates(user_input, graph)

            index0 = json_string.find("```json\n")
            json_string = json_string[index0 + len("```json\n"):] if index0 > 0 else json_string
            index1 = json_string.rfind("}\n```")
            json_string = json_string[:index1+1] if index1 > 0 else json_string

            python_object = json.loads(json_string)    
            print(f"Potential guilty commit is found from git show of commit {commit}. {python_object}")
            with open(f"./results/guilty_commit.log", "a") as f:
                f.write(f"\nPotential guilty commit is found from git show of commit {commit} in {workdir}.\n {python_object}\n")

        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON from git show of commit {commit}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing commit {commit}: {e}")
    
    
def git_show_rag_analysis():
    import re
    from datetime import datetime
    gitshow_folder = "./gitshow_results"
    git_log_output = get_git_log(blame_range, match, f"{workdir}", container)

    def parse_commits_from_git_log(git_log: str) -> list:
        """Parse commit SHAs from git log output."""
        commits = []    
        for line in git_log.split('\n'):
            line = line.strip()
            if line.startswith('commit '):
                commit_sha = line.split()[1]
                commits.append(commit_sha)
            elif re.match(r'^[0-9a-f]{40}$', line) or re.match(r'^[0-9a-f]{7,}$', line):
                commits.append(line)
        return commits

    commits = parse_commits_from_git_log(git_log_output)
    
    # Initialize embeddings and vector store
    from langchain_huggingface import HuggingFaceEmbeddings
    #from langchain_chroma import Chroma
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    
    # Use LangChain compatible embeddings
    langchain_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    client = chromadb.PersistentClient(
        path="./chroma_gitshow_db",  # persist_directory parameter is now 'path'
        settings=Settings(anonymized_telemetry=False)
    )
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # Lightweight, good for general use
    )
    
    collection_name = "gitshow_collection"
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef,
    )

    def add_commit_to_collection(commit_sha: str, collection) -> str:
        try:
            show_results = get_git_show(commit_sha, f"{workdir}", issue_information["container"])
            collection.add(
                documents=[show_results],
                metadatas=[{"commit_sha": commit_sha}],
                ids=[commit_sha]
            )
            print(f"Added commit {commit_sha} to collection.")
        except Exception as e:
            print(f"Failed to add commit {commit_sha} to collection: {e}")
        return collection
    
    collection = add_commit_to_collection(gitshow_folder, collection)

    from langchain_chroma import Chroma
    db = Chroma(client=client, collection_name=collection_name, embedding_function=langchain_embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template(
        """Given the following queston and context, find the guilty commit of the problem described in question. 
        Question: {question}
        Context: {context}

        Steps:
        1) Identify the commits that related to the bug based on the question and context. The question contains the information about the test failure, such as the test file, test case, error message, and trace. The context contains the git show information of candidate commits.
        2) If there are mulitple commits identified in step 1, summary the update history of the related code, for example, the code was added in commit A, then updated in commit B and C, and the bug is likely introduced in commit B or C.
        3) Identify the guilty commit based on the update history and the information in question and context.

        Only inference the result based on the context, do not use any other information. 
        
        Return the result in a beutified json without explanation, just the json with the following format:
        ```json
        {{
        "guilty commit": "the commit sha of the guilty commit, if no guilty commit is found, return 'No guilty commit found'",
        "update_history": "the update history of the related code, if no update history is found, return 'No update history found'",    
        }}
        ```
        """
    )

    from langchain_core.output_parsers import JsonOutputParser
    import llm
    # Define the chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | JsonOutputParser()
    )
    
    question = f"Find the guilty commit for the test failure with test file: {test_file}, test class: {test_class}, test case: {test_case}, error message: {error_message}, and trace: {trace}."
    answer = rag_chain.invoke(question)
    print(f"Guilty commit analysis result: {answer}")

if container is not None and workdir is not None: 
    try:
        failure_info = issue_information
        print(f"Failure information: {failure_info}")
    
        if not args.rag:
            git_log_analysis()
        else:
            git_show_rag_analysis()
        end = time.time()
        print(f"Time used: {end - start} seconds.")

    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
