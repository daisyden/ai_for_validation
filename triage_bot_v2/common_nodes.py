from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from vllm_service import llm
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage
import json

from datetime import datetime

################################
# State
################################
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    execution_path: list[str]
    issue_info: dict


def get_history(state: State, node: str):
    executed_nodes = state.get('execution_path')
    last_node = executed_nodes[-1] if executed_nodes else "START"
    
    print(f"=== Node {node} executed after {last_node} ====\n")
    return executed_nodes, last_node

  
def stream_graph_updates(user_input: str, graph: StateGraph):
    final_result = ""
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}], "execution_path": [], "issue_info": {}}):
        
        for value in event.values():
            if isinstance(value["messages"], list):
                json_string = value["messages"][-1].content
            else:
                json_string =value["messages"].content
            
            if len(json_string) > 0:
                try:
                    if "```json" in json_string:
                        index0 = json_string.find("```json\n")
                        json_string = json_string[index0 + len("```json\n"):]
                        index1 = json_string.rfind("}\n```")
                        json_string = json_string[:index1+1]                    

                    python_object = json.loads(json_string)

                    json_string = json.dumps(python_object, indent=4)
                    print(f"### Assistant output json: {json_string}")
                except:
                    print(f"### Assistant output text: {json_string}")
                final_result = json_string
            elif hasattr(value["messages"], 'tool_calls') and value["messages"].tool_calls:
                print(f"### Assistant tool_calls: {value['messages'].tool_calls}")
                final_result = json.dumps(value["messages"].tool_calls, indent=4)
            # elif hasattr(value["messages"], 'content') and "<tool_call>" in value["messages"].content:
            #     print(f"### Assistant tool_calls: {value['messages'].content}")
            #     tool_call_content = value['messages'].content
            #     start_tag = "<tool_call>"
            #     end_tag = "</tool_call>"
            #     start_idx = tool_call_content.find(start_tag)
            #     end_idx = tool_call_content.find(end_tag)
            #     if start_idx != -1 and end_idx != -1:
            #         final_result = tool_call_content[start_idx + len(start_tag):end_idx]
            #     else:
            #         final_result = tool_call_content
            else:               
                print("### No output from the model.")
    return final_result




################################
# Nodes 
################################

def group(state: State):
    executed_nodes, last_node = get_history(state, "group")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that groups similar pytest failures."),
        ("human", "{text}")
    ])
    chain = {"text": RunnablePassthrough()} | prompt | llm
    return {"messages": chain.invoke(state["messages"]),
            "execution_path": state.get("execution_path", []) + ["group"],
            "issue_info": {}
    }    
    

def doc_analysis(state: State):
    executed_nodes, last_node = get_history(state, "doc_analysis")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can analysis technical documents (such as test log, backtrace, git log, git show, etc.) to extract useful information for issue triage and debugging."),
        ("human", "{text}")
    ])
    chain = {"text": RunnablePassthrough()} | prompt | llm
    
    return {"messages": chain.invoke(state["messages"]),
            "execution_path": state.get("execution_path", []) + ["doc_analysis"],
            "issue_info": {}
    }
 

################################
# RAG node
###############################

def depsRAG(
    state: State,
) -> State:

    executed_nodes, last_node = get_history(state, "depsRAG")

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    json_string = ai_message.content
    print(json_string)
    
    if "```json" in json_string:
        index0 = json_string.find("```json\n")
        json_string = json_string[index0 + len("```json\n"):]
        index1 = json_string.rfind("}\n```")
        json_string = json_string[:index1+1]
    
    python_object = json.loads(json_string)


    from langchain_community.document_loaders import CSVLoader
    from langchain_community.vectorstores import FAISS
    from langchain_core.output_parsers import StrOutputParser
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", encode_kwargs={'normalize_embeddings': True})
    loader = CSVLoader("rag_doc/ops_dependency.csv")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n", " ", ""], chunk_size=200, chunk_overlap=50, keep_separator=True,)
    splits = text_splitter.split_documents(docs)

    # Create vector store
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings, distance_strategy="COSINE")
    retriever = vectorstore.as_retriever()

    # Define prompt template
    template = """Answer the question based only on the following context:
    
    Context: {context}

    Question: {question}

    DECISION CRITERIA:
    - Use the following error information to help decision making:
        * dtype: {{python_object["dtype"]}}
        * Torch operation: {{python_object["torch_op"]}}
        * Traceback: {{python_object["traceback"]}}
        * Test case: {{python_object["test_case"]}}
        * Error message: {{python_object["error_message"]}}
    - If the context has oneDNN and dtype is float32, float64, float16, bfloat16, or float8 dtypes -> answer is 'oneDNN'.
    - If the context has oneDNN and dtype is complex32, complex64, or complex128 dtypes -> answer is 'MKL'.
    - If the context has MKL -> answer is 'MKL'.
    - If the context has torch CPU -> anwser is 'torch CPU'.
    - If the context is empty and the torch operation is GEMM-like and dtype is float32, float64, float16, bfloat16, or float8 dtypes -> answer is 'oneDNN'.
    - If the context is empty and the torch operation is GEMM-like and  dtype is complex32, complex64, or complex128 dtypes -> answer is 'MKL'.
    - Otherwise -> return 'sycl'.    

    Return only the final answer with no additional explanation.
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    # Define the chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    if "NotImplementedError" in python_object['error_message']:
        answer = "sycl"
    elif python_object['torch_op'] != None:
        question = f"""
        What is the depedent library of the torch-xpu-ops operation {python_object["torch_op"]}?
        """ 
        answer = rag_chain.invoke(question)
    else:
        answer = "sycl"

    python_object["dependency"] = answer
    json_string = json.dumps(python_object)
    return {"messages": AIMessage(content=json_string),
            "execution_path": state.get("execution_path", []) + ["depRAG"],
            "issue_info": python_object
    }
         

def depsrag_graph():
    graph_builder = StateGraph(State)  
    graph_builder.add_node("doc_analysis", doc_analysis)  
    graph_builder.add_node("depsRAG", depsRAG)
    graph_builder.add_edge(START, "doc_analysis")
    graph_builder.add_edge("doc_analysis", "depsRAG")
    graph_builder.add_edge("depsRAG", END)
    graph = graph_builder.compile()
    return graph


def document_analysis_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("doc_analysis", doc_analysis)
    graph_builder.add_edge(START, "doc_analysis")
    graph_builder.add_edge("doc_analysis", END)
    graph = graph_builder.compile()
    return graph

                
