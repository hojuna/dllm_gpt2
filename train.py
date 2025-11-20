import os
import torch
from transformers import GPT2Config, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from modeling_gpt2_dllm import GPT2dLLMLMHeadModel

def train():
    # 1. Configuration
    model_name = "gpt2"
    config = GPT2Config.from_pretrained(model_name)
    config.bd_size = 32 # Set block size for dLLM
    
    # 2. Model
    print("Loading model...")
    model = GPT2dLLMLMHeadModel.from_pretrained(model_name, config=config)
    
    # 3. Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4. Dataset
    print("Loading Alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca")
    
    def tokenize_function(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        
        texts = []
        for inst, inp, out in zip(instructions, inputs, outputs):
            if inp:
                text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            else:
                text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{inst}\n\n### Response:\n{out}"
            texts.append(text + tokenizer.eos_token)
            
        return tokenizer(texts, truncation=True, max_length=256, padding="max_length")
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir="./gpt2_dllm_results",
        overwrite_output_dir=True,
        max_steps=20, # Train for only 20 steps for quick verification
        per_device_train_batch_size=4,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=5e-5,
        remove_unused_columns=False, 
        report_to="none",
    )
    
    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )
    
    # 8. Train
    print("Starting training...")
    trainer.train()
    
    # 9. Save
    model.save_pretrained("./gpt2_dllm_final")
    tokenizer.save_pretrained("./gpt2_dllm_final")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()
