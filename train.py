import os
import torch
from transformers import GPT2Config, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_from_disk
from modeling_gpt2_dllm import GPT2dLLMLMHeadModel
import wandb

def train():
    # 1. Configuration
    model_name = "gpt2-Medium"
    config = GPT2Config.from_pretrained(model_name)
    config.bd_size = 256 # Set block size for dLLM
    
    # 2. Model
    print("Loading model...")
    model = GPT2dLLMLMHeadModel.from_pretrained(model_name, config=config, dtype=torch.bfloat16)
    
    # 3. Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4. Dataset
    print("Loading merged SFT dataset from disk...")
    dataset = load_from_disk("./merged_sft_dataset")
    
    def tokenize_function(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        
        texts = []
        for inst, inp, out in zip(instructions, inputs, outputs):
            if inp:
                text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            else:
                text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
            texts.append(text + tokenizer.eos_token)
            
        return tokenizer(texts, truncation=True, max_length=256, padding="max_length")
    
    # save_to_disk / load_from_disk로 다루는 것은 단일 Dataset이므로 바로 map
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    wandb.init(project="dllm-gpt2")
    wandb.config.update({
        "model": "gpt2-Medium",
        "dataset": "merged_sft_dataset",
        "batch_size": 8,
        "learning_rate": 5e-5,
    })
    
    
    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir="./gpt2_dllm_results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        save_strategy="epoch",                
        logging_steps=10,
        learning_rate=5e-5,
        remove_unused_columns=False,
        gradient_accumulation_steps=8,
        report_to="wandb",
        lr_scheduler_type="cosine",
        run_name="dllm-gpt2-Medium",
    )
    
    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
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
