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

prompt = "What's the average yard of a running back for the season?"

messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    padding=True,
)

outputs = model.generate(
    formatted_prompt,
    max_new_tokens=100,
    repetition_penalty=6.0,
    tokenizer=tokenizer,
    stop_strings=";",
)
print(formatted_prompt)

print("Inference:")
print(
    tokenizer.decode(
        outputs[0],
    )
)
