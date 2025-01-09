from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch

# login()


def process_dataset(sample):
    entry = [
        {"role": "user", "content": sample["sql_prompt"]},
        {"role": "assistant", "content": sample["sql"]},
    ]

    return {"template_msg": tokenizer.apply_chat_template(entry, tokenize=False)}


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load the model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name
).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

# Set up the chat format
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

# Set our name for the finetune to be saved &/ uploaded to
finetune_name = "SmolLM2-FT-SQL"
finetune_tags = ["smol-course", "module_1"]

ds = load_dataset(path="gretelai/synthetic_text_to_sql")
ds = ds.map(process_dataset)

# Configure the SFTTrainer
sft_config = SFTConfig(
    output_dir="./sft_output",
    max_steps=2000,  # Adjust based on dataset size and desired training duration
    per_device_train_batch_size=4,  # Set according to your GPU memory capacity
    learning_rate=5e-5,  # Common starting point for fine-tuning
    logging_steps=10,  # Frequency of logging training metrics
    save_steps=100,  # Frequency of saving model checkpoints
    evaluation_strategy="steps",  # Evaluate the model at regular intervals
    eval_steps=50,  # Frequency of evaluation
    use_mps_device=(
        True if device == "mps" else False
    ),  # Use MPS for mixed precision training
    hub_model_id=finetune_name,  # Set a unique name for your model
    dataset_text_field="template_msg",
)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds["train"],
    tokenizer=tokenizer,
    eval_dataset=ds["test"],
)

# Train the model
print("start training the model")
trainer.train()

# Save the model
print("saving the model")
trainer.save_model(f"./{finetune_name}")

trainer.push_to_hub(tags=finetune_tags)
