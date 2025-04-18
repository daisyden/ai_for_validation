# README

## Overview

This Jupyter Notebook `rag_langchain.ipynb` demonstrates how to set up RAG with the LangChain framework. It includes steps to initialize a text-generation pipeline, wrap it in a LangChain-compatible object, and create an in-memory vector store for embeddings.

---

## Prerequisites

Before running this notebook, ensure you have the following installed:

- Python 3.10 or later
- Required Python libraries:
  - `langchain-community`
  - `langchain-huggingface`
  - `langgraph`
  - `langchain-text-splitters`

You can install the required libraries using the following command:

```bash
pip install langchain-huggingface langchain-text-splitters langchain-community langgraph
```
## Notes
  Ensure that the knowledge.txt file is available in the same directory as the notebook.
  Replace "your_huggingface_api_token" with your actual Hugging Face API token.
