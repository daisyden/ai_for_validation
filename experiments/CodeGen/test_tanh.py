from transformers import AutoModelForCausalLM, AutoTokenizer

#model_name = "deepseek-ai/deepseek-coder-6.7B-instruct" 
model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
#model = model.to(device='xpu')

prompt = "Write a sycl program wtih parallel_for to calcualte std::tanh on GPU device with input Nan and Inf and dtype double, also caclulate std::tanh before the parallel for on host, compare the host result with device result."
inputs = tokenizer(prompt, return_tensors="pt")
#inputs = inputs.to(device='xpu')

outputs = model.generate(**inputs, max_length=1024)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
