################################
# RAG node
###############################
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from .vllm_service import llm
from langchain_core.runnables import RunnablePassthrough
import os

# try:
#     import pdb
#     pdb.set_trace()
#     from sentence_transformers import SentenceTransformer
#     print("sentence-transformers imported successfully")
    
#     # Test the model directly
#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#     embeddings = model.encode(["Hello world"])
#     print(f"✓ Model loaded and embeddings created: {embeddings.shape}")
# except ImportError as e:
#     print("✗ Failed to import sentence-transformers. Please ensure it is installed.")
#     raise e

def get_rag_chain(doc_folder: str):
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load multiple text documents
    text_files = []

    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for issue_file in os.listdir(doc_folder):
        issue_file = os.path.join(doc_folder, issue_file)
        if issue_file.endswith(".txt"):
            text_files.append(issue_file)

    all_docs = []
    for text_file in text_files:
        if os.path.exists(text_file):
            loader = TextLoader(text_file)
            docs = loader.load()
            # Add document name as metadata to establish relationship
            doc_name = os.path.basename(text_file)
            for doc in docs:
                doc.metadata["source_document"] = doc_name
            all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    # Create vector store with documents from multiple sources
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Define prompt template with source document information
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}

    Note: The context includes information from multiple documents. Consider the source document when providing your answer.
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Define the chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def gen_prompt(skipped, error_message, traces):
        question = f"""
Analyze the following test failure and identify any similar GitHub issues from the existing issue database:

**TEST FAILURE DETAILS:**
- Cases: {', '.join(skipped)}
- Error Message: {error_message}
- Trace Example: {traces}

**ANALYSIS CRITERIA FOR Similar Issues**: 
- Similar issue is found if issues with exact or partially overlapping error patterns
- Similar issue is found if issues with related torch operators (such as add, matmul, reduction, etc.), modules, or components

**REQUIRED RESPONSE FORMAT:**
- For each matching issue, provide details with confidence level
- Group by confidence level (high_confidence, medium_confidence)
- If no matches found, return empty arrays

**SPECIFIC MATCHING CONSIDERATIONS:**
- Compare test file paths, class names, and test case names
- Look for similar error patterns, stack traces, or failure types
- Consider issues with the same underlying component or module
- Note issues with identical exception types or error keywords

Please analyze based on the available issue data. Return ONLY a valid JSON object in this exact format without any explanation or markdown code blocks:

{{
    "high_confidence": [
        {{
            "title": "issue title",
            "url": "issue url",
            "justification": "brief justification"
        }}
    ],
    "medium_confidence": [
        {{
            "title": "issue title",
            "url": "issue url",
            "justification": "brief justification"
        }}
    ]
}}

"""
        return question


def get_duplicated_issues_with_rag(skipped, error_message, traces, issue_folder):
    rag_chain = get_rag_chain(issue_folder)
    question = gen_prompt(skipped, error_message, traces)
    answer = rag_chain.invoke(f"{question}")
    answer = answer.replace("```json", "").replace("```", "").strip()
    import json
    results = json.loads(answer)
    return results