import os
from github_issue import Github_Issue
from langchain_core.runnables import RunnablePassthrough 
from vllm_service import llm
from langchain_core.prompts import ChatPromptTemplate


def add_issue_to_collection(issue_folder: str, collection: object, issue_id: str = None):
    documents = []
    ids = []
    metadatas = []
    
    for filename in os.listdir(issue_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(issue_folder, filename)
            
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                _issue_id = filepath.split('/')[-1].split('.')[0]
                
                if issue_id is not None and _issue_id == issue_id:
                    continue
                documents.append(content)
                ids.append(_issue_id)
                metadatas.append({"issue_id": _issue_id})
                
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )

    return collection

# def get_duplicated_issues_with_rag(issue_info: dict, issue_folder:str, repo:str, token:str):
#     # TBD, use RAG to find duplicated issues based on the extracted information
#     # for example, we can use the issue title, description, error message and test case information to find similar issues in the existing issue list

#     # Here we just return the issues with the same test case as an example
#     test_file = issue_info.get("test_file", "unknown")
#     test_class  = issue_info.get("test_class", "unknown")
#     test_case = issue_info.get("test_case", "unknown")
#     error_message = issue_info.get("error_message", "unknown")
#     trace = issue_info.get("trace", "unknown")

#     skipped_issues = download_all_open_issues_and_get_skiplist(repo, token, issue_folder)
#     # Use metadata filtering to find similar issues
#     # Filter by test file, test class, test case, or error message similarity
    
#     # Initialize embeddings and vector store
#     from langchain_huggingface import HuggingFaceEmbeddings
#     #from langchain_chroma import Chroma
#     import chromadb
#     from chromadb.config import Settings
#     # embeddings = HuggingFaceEmbeddings()
#     # vectorstore = Chroma(persist_directory=issue_folder, embedding_function=embeddings)
#     from chromadb.utils import embedding_functions
#     client = chromadb.PersistentClient(
#         path=issue_folder,  # persist_directory parameter is now 'path'
#         settings=Settings(anonymized_telemetry=False)
#     )
#     sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name="all-MiniLM-L6-v2"  # Lightweight, good for general use
#     )
    
#     collection_name = "issues_collection"
#     collection = client.get_or_create_collection(
#         name="issues_collection",
#         embedding_function=sentence_transformer_ef,
#     )

    

#     add_issue_to_collection(issue_folder, collection)

#     results = collection.query(
#         query_texts=[f"Test: {test_file}::{test_class}::{test_case}\nError: {error_message}\nTrace: {trace}"],
#         n_results=2,
#     )
    
#     if results is not None:
#         print(f"Found similar issues: {results['ids']} with distances {results['distances']}")
#         print(f"Check the similarity of the first issue with the problem...")

#         from common_nodes import document_analysis_graph
#         input = f""" Please help to check whether the issue with content 
#         <% raw %>
#         {results['documents'][0]} 
#         <% endraw %>
#         is a duplicated issue of the current issue with following information
#         test case {test_file}::{test_class}::{test_case}
#         error message <% raw %> {error_message} <% endraw %>
#         trace <% raw %> {trace} <% endraw %>
#         If the issue is a duplicate, please return true otherwise return false. No any explaination is needed. Just answer true or false.
#         """
        
#         graph = document_analysis_graph()

#         from langchain_core.messages import AIMessage
#         res = graph.invoke(AIMessage(content=input))

#         import pdb
#         pdb.set_trace()
#         if "false" in res.content.strip().lower():
#             print(f"No duplicated issue found after a double check with LLM.")
#             return None
#         else:
#             print(f"The issue {results['ids'][0]} is a verified to be a duplicated issue.")
#             return results['ids'] 
    
#     return None



# def get_duplicated_issues(id: str, skipped:list, error_message:str, trace:str, issue_folder:str, ratio: float):
#     print(f"\n\n### Checking duplicated issues for group {id} with {error_message} ...\n")
#     duplicated_issues = []
#     import os, re

#     def extract_errors_from_log(log_content):
#         """
#         Extract assertion errors and runtime errors from log content
#         """
#         # Patterns for different types of errors
#         patterns = {
#             'assertion_error': r'AssertionError:?(.*)',
#             'runtime_error': r'RuntimeError:?(.*)',
#             #'traceback': r'Traceback \(most recent call last\):\n(?:.*\n)*?(?:\w+Error:.*)',
#             'any_python_error': r'^\w+Error:.*$',
#             'exception': r'Exception:?(.*)',
#             'value_error': r'ValueError:?(.*)',
#             'type_error': r'TypeError:?(.*)',
#             'index_error': r'IndexError:?(.*)',
#             'key_error': r'KeyError:?(.*)',
#             'import_error': r'ImportError:?(.*)',
#             'crash': r'(.*)crash(.*)',
#         }
        
#         errors = {}
        
#         for error_type, pattern in patterns.items():
#             matches = re.findall(pattern, log_content, re.MULTILINE)
#             if matches:
#                 errors[error_type] = matches
        
#         return errors
    
#     for issue_file in os.listdir(issue_folder):
#         issue_file = os.path.join(issue_folder, issue_file)
#         issue_file_id = issue_file.split('/')[-1].split('.')[0]
#         print(f"## Checking issue file {issue_file_id}...")
#         if issue_file.endswith(".txt") and issue_file_id != f"issue_group{id}":
#             with open(issue_file, "r", encoding="utf-8") as f:
#                 content = f.read()
#                 # extract match with test_file, test_case and error message
#                 for skip in skipped:
#                     if skip in content and "Skipped: Yes" in content:
#                         print(f"# Skipping {skip} of issue_group{id} as it is marked skipped in {issue_file}")
#                         duplicated_issues.append(issue_file_id)
#                     else:
#                         _test_file = '/'.join(skip.split(',')[1].split('.')[:-1])
#                         test_file = _test_file + '.py'
#                         test_case = skip.split(',')[2].strip()

#                         if (f"{test_file}".replace('_xpu.py','.py').replace('test/', '') in content) and \
#                             f"{test_case}" in content and \
#                             f"{error_message}" in content:
#                             duplicated_issues.append(issue_file_id)

#                 # Check whether error message is similar
#                 errors = extract_errors_from_log(content)
#                 print(f"Extracted errors {errors} from issue {issue_file_id}")

#                 from difflib import SequenceMatcher
#                 def similar(a, b):
#                     return SequenceMatcher(None, a, b).ratio()

#                 for error_type in errors.keys():
#                     if similar(error_message, f"{error_type}: {errors[error_type][0]}") > ratio:
#                         print(f"\n# Found similar error message in issue {issue_file_id} with issue_group{id}: {error_type}: {errors[error_type][0]}    .vs    {error_message} \nsimilarity ratio is {similar(error_message, f'{error_type}: {errors[error_type][0]}')}")
#                         duplicated_issues.append(issue_file_id)
#                     else:
#                         print(f"\n# No similar error message in issue {issue_file_id} with issue_group{id}: {error_type}: {errors[error_type][0]}    .vs    {error_message} similarity ratio is {similar(error_message, f'{error_type}: {errors[error_type][0]}')}")

#     print(f"## Duplicated issues for group {id}: {duplicated_issues}")
#     print("########################################\n\n")
#     return duplicated_issues


def download_issue_content(issue):
    content = ""
    reporter = issue.user.login
    owner = issue.assignee.login if issue.assignee is not None else "Unassigned"
    title = issue.title
    skipped = "No"
    if "skipped" in [label.name for label in issue.labels]:
        skipped = "Yes"
    if issue.body is not None:
        content += issue.body + "\n"
    
    content = content.split('### Versions')[0]
    comments = issue.get_comments()
    comment = ""
    for _comment in comments:
        if _comment.body is not None:
            comment += _comment.body + "\n"
    return f"#{issue.id}\nReporter: {reporter}\nOwner: {owner}\nTitle: {title}\nSkipped: {skipped}\nBody: {content}\nComments: {comment}"


def download_all_open_issues_and_get_skiplist(repo:str, token:str, issues_folder:str):
    from github_issue import Github_Issue 
    gh = Github_Issue(repo, token)
    issues = gh.get_issues(state="open")

    skip_list = []
    for issue in issues:
        if "skipped" in [label.name for label in issue.labels]:
            skip_list.append(f"{issue.number}.txt")
        if os.path.exists(f"{issues_folder}/{issue.number}.txt") and issue.updated_at.timestamp() < os.path.getmtime(f"{issues_folder}/{issue.number}.txt"):
            #print(f"Issue #{issue.number} already downloaded.")
            continue
        content = download_issue_content(issue)
        with open(f"{issues_folder}/{issue.number}.txt", "w", encoding="utf-8") as f:
            f.write(f"Issue #{issue.number}: {issue.title}\n")
            f.write(content)
            f.write("\n" + "="*80 + "\n\n")
            print(f"Downloaded issue #{issue.number} to file.")
    return skip_list


def get_similar_issues(issue_info: dict, issues_folder:str, repo:str, token:str):    
    test_file = issue_info.get("test_file", "unknown")
    test_class  = issue_info.get("test_class", "unknown")
    test_case = issue_info.get("test_case", "unknown")
    error_message = issue_info.get("error_message", "unknown")
    trace = issue_info.get("trace", "unknown")

    _ = download_all_open_issues_and_get_skiplist(repo, token, issues_folder)
    # Use metadata filtering to find similar issues
    # Filter by test file, test class, test case, or error message similarity
    
    # Initialize embeddings and vector store
    from langchain_huggingface import HuggingFaceEmbeddings
    #from langchain_chroma import Chroma
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    
    # Use LangChain compatible embeddings
    langchain_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    client = chromadb.PersistentClient(
        path="./chroma_db",  # persist_directory parameter is now 'path'
        settings=Settings(anonymized_telemetry=False)
    )
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # Lightweight, good for general use
    )
    
    collection_name = "issues_collection"
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef,
    )

    issue_id = issue_info.get("link", None).split('/')[-1] if issue_info.get("link", None) is not None else None
    collection = add_issue_to_collection(issues_folder, collection, issue_id)

    #from langchain.vectorstores import Chroma
    from langchain_chroma import Chroma
    db = Chroma(client=client, collection_name=collection_name, embedding_function=langchain_embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_template(
        """Given the following question and context, find the most similar issue from the retrieved issues if any. 
        Question: {question}
        Context: {context}

        A similar issue:
        1) It has a similar error message as well as the error type
        2) It has a similar test case, for example, the test file and test case name are similar. 
        
        No similar issue:
        1) neither the error message is similar nor the test case is similar.

        Only inference the result based on the context, do not use any other information. 
        
        Return the result in a beutified json without explanation, just the json with the following format:
        ```json
        {{
        "most_similar_issue": "the issue id of the most similar issue, if no similar issue is found, return 'No similar issues found'",
        "most_similar_issue_title": "the title of the most similar issue, if no similar issue is found, return 'No similar issues found'",    
        "similar_issue_ids": "the list of similar issue ids, if no similar issue is found, return 'No similar issues found'"
        }}
        ```
        """
    )

    from langchain_core.output_parsers import JsonOutputParser
    # Define the chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | JsonOutputParser()
    )
    
    question = f"Find the most similar issue for the test failure with test file: {test_file}, test class: {test_class}, test case: {test_case}, error message: {error_message}, and trace: {trace}."
    answer = rag_chain.invoke(question)
    # results = collection.query(
    #     query_texts=[f"Test: {test_file}::{test_class}::{test_case}\nError: {error_message}\nTrace: {trace}"],
    #     n_results=2,
    # )
    
    issue_info["similar_issues"] = answer
    print(f"Most similar issue found: {issue_info['similar_issues']}")
    return issue_info
    


if __name__ == "__main__":
    import json, os    
    with open("results/2006/issue#2006_test_unary_ufuncs_xpu.py_TestUnaryUfuncsXPU_test_nonzero_large_xpu_int8.json", "r") as f:
        issue_info = json.load(f)
    issue_folder = "xpu_issues"
    repo = "intel/torch-xpu-ops"
    token = os.environ.get("GITHUB_TOKEN", "")
    
    get_similar_issues(issue_info, issue_folder, repo, token)
