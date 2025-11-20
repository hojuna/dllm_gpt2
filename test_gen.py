import torch
from transformers import GPT2Tokenizer, GPT2Config
from modeling_gpt2_dllm import GPT2dLLMLMHeadModel
import time

def test_generation():
    model_path = "./gpt2_dllm_final"
    
    print(f"Loading model from {model_path}...")

    config = GPT2Config.from_pretrained(model_path)
    model = GPT2dLLMLMHeadModel.from_pretrained(model_path, config=config)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    model.eval()
    
    prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nTell me a joke.\n\n### Response:\n"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    print(f"\nPrompt: {prompt}")
    print("-" * 50)
    
    max_new_tokens = 20
    mask_id = tokenizer.eos_token_id 
    
    original_len = input_ids.shape[1]
    num_new_tokens = max_new_tokens
    
    x_t = torch.cat([input_ids, torch.full((input_ids.shape[0], num_new_tokens), mask_id, dtype=torch.long)], dim=1)
    
    print("Starting generation (visualizing denoising steps)...")
    
    top_p = 0.9
    temperature = 1.0
    threshold = 0.5 

    x_t = model.generate(
        input_ids, 
        max_new_tokens=max_new_tokens, 
        mask_id=mask_id, 
        block_size=32, 
        top_p=top_p, 
        temperature=temperature, 
        threshold=threshold,
        verbose=True,
        tokenizer=tokenizer
    )
        
    print("-" * 50)
    print("Final Output:")
    print(tokenizer.decode(x_t[0], skip_special_tokens=True))

if __name__ == "__main__":
    test_generation()
