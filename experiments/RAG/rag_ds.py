import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a pre-trained sentence transformer model
encoder = SentenceTransformer('thenlper/gte-large')


try:
    with open("./case.txt", 'r') as file:
        lines = file.readlines()
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

document_embeddings = encoder.encode(lines)

# Create a FAISS index
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

from transformers import pipeline

# Load DeepSeek model (assuming it's available as a text generation model)
generator = pipeline('text-generation', model='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B')

def rag_response(query, top_k=3):
    # Encode the query
    query_embedding = encoder.encode([query])

    # Search the FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve relevant documents
    relevant_docs = ["".join(lines[i]) for i in indices[0]]

    # Combine the query and relevant documents
    context = " ".join(relevant_docs)
    input_text = f"Query: {query}\nContext: {context}\nAnswer:"

    print(input_text)

    # Generate response using DeepSeek
    response = generator(input_text, max_length=200, num_return_sequences=1)

    return response[0]['generated_text']

# Example usage
query = "unit test test_LayerNorm got 'Fatal: Python illegal instruction', is it a regression?"

response = rag_response(query, top_k=1)
print(response)


