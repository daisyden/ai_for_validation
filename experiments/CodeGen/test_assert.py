from transformers import AutoModelForCausalLM, AutoTokenizer

#model_name = "deepseek-ai/deepseek-coder-6.7B-instruct" 
model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
#model = model.to(device='xpu')

#prompt = "Write a python statement to print the arguments of 'assert len(filtered_ops) == 0, err_msg' in one line."
prompt = "Write a python statement to print the 'self.assertEqual(results, results, atol=1e-3, rtol=1e-3)' with arugments included in {}. For example output print(f\"self.assertEqual({cuda_results}, {cpu_results}, atol=1e-3, rtol=1e-3)\") for self.assertEqual(cuda_results, cpu_results, atol=1e-3, rtol=1e-3)"
inputs = tokenizer(prompt, return_tensors="pt")
#inputs = inputs.to(device='xpu')

outputs = model.generate(**inputs, max_length=400)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
