from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

finetune_name = "SmolLM2-FT-SQL"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=finetune_name
).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=finetune_name)

prompt = "Write a SQL query with the total amount of population per city"

messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(formatted_prompt)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100)

print("Inference:")
print(tokenizer.decode(outputs[0]))
