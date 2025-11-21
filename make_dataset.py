from datasets import load_dataset, Dataset, concatenate_datasets

# imone/OpenOrca_FLAN

#############################################
# 1. Dolly 15k
#############################################
dolly = load_dataset("databricks/databricks-dolly-15k")

def map_dolly(e):
    return {
        "instruction": e["instruction"] if "instruction" in e else "",
        "input": e["context"] if "context" in e else "",
        "output": e["response"]
    }

dolly = dolly["train"].map(map_dolly)


#############################################
# 2. LIMA (1000 high-quality SFT)
#############################################
lima = load_dataset("imone/OpenOrca_FLAN")

def map_lima(e):
    # LIMA는 기본적으로: {"instruction":..., "output":...} 구조
    return {
        "instruction": e["system"],
        "input": e["instruction"],  # LIMA는 context/input 없음
        "output": e["response"]
    }

lima = lima["train"].map(map_lima)


#############################################
# 3. UltraChat (1M → 20k subset)
#############################################
# UltraChat: {"prompt":..., "response":...} 기반
ultra = load_dataset("tatsu-lab/alpaca")

def map_ultra(e):
    return {
        "instruction": e["instruction"],
        "input": e["input"] if "input" in e else "",
        "output": e["output"]
    }

ultra = ultra["train"].map(map_ultra)


#############################################
# 4. 합치기
#############################################
full_dataset = concatenate_datasets([dolly, lima, ultra])

print(full_dataset)
print(full_dataset[0])


#############################################
# 5. 저장
#############################################
save_path = "./merged_sft_dataset"
full_dataset.save_to_disk(save_path)
print(f"Saved merged dataset to: {save_path}")
