from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline
generator = pipeline('text-generation', model='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', max_new_tokens=500)

llm_model = HuggingFacePipeline(pipeline=generator)


