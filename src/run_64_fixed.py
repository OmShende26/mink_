import os
import pandas as pd
import random
import logging
import numpy as np
from pathlib import Path
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Import the plotting function from the authors' local eval.py
from eval import fig_fpr_tpr

logging.basicConfig(level='ERROR')

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_NAME = "common-pile/comma-v0.1-2t"
SEEN_CSV_PATH = "seen.csv"       # Update with your actual path
UNSEEN_CSV_PATH = "unseen.csv"   # Update with your actual path
OUTPUT_DIR = "out/custom_eval_fixed"

# Strict fixed length constraint
FIXED_TOKENS = 64

# ==========================================
# 2. DATA PREPARATION (Fixed Length & Balanced)
# ==========================================
def process_csv_fixed(csv_path, label, tokenizer, fixed_len):
    """Reads a CSV, tokenizes the text, and strictly extracts exact-length chunks."""
    df = pd.read_csv(csv_path)
    chunks = []
    
    print(f"Extracting strict {fixed_len}-token chunks from {csv_path}...")
    for text in tqdm(df['text'].dropna()):
        tokens = tokenizer.encode(str(text), add_special_tokens=False)
        
        # Slice into exact length chunks. Leftover tokens at the end are strictly ignored.
        for i in range(0, len(tokens) - fixed_len + 1, fixed_len):
            chunk_tokens = tokens[i : i + fixed_len]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append({"text": chunk_text, "label": label})
            
    return chunks

def prepare_balanced_data(tokenizer):
    """Processes both CSVs and balances them so they have the exact same number of chunks."""
    seen_data = process_csv_fixed(SEEN_CSV_PATH, label=1, tokenizer=tokenizer, fixed_len=FIXED_TOKENS)
    unseen_data = process_csv_fixed(UNSEEN_CSV_PATH, label=0, tokenizer=tokenizer, fixed_len=FIXED_TOKENS)
    
    # Balance the datasets
    min_size = min(len(seen_data), len(unseen_data))
    print(f"\nRaw chunks extracted: {len(seen_data)} seen, {len(unseen_data)} unseen.")
    print(f"Balancing dataset to exactly {min_size} chunks per class to prevent skew...")
    
    # Randomly sample to perfectly match the smaller dataset
    random.seed(42)
    seen_balanced = random.sample(seen_data, min_size)
    unseen_balanced = random.sample(unseen_data, min_size)
    
    # Combine and shuffle
    all_data = seen_balanced + unseen_balanced
    random.shuffle(all_data)
    
    print(f"Final evaluation pool: {len(all_data)} total snippets.")
    return all_data


# ==========================================
# 3. EVALUATION LOGIC
# ==========================================
def calculatePerplexity(sentence, model, tokenizer, gpu):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(gpu)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
        
    return torch.exp(loss).item(), all_prob, loss.item()

def inference(model1, tokenizer1, text, ex):
    pred = {}
    p1, all_prob, p1_likelihood = calculatePerplexity(text, model1, tokenizer1, gpu=model1.device)
    p_lower, _, p_lower_likelihood = calculatePerplexity(text.lower(), model1, tokenizer1, gpu=model1.device)

    pred["ppl"] = p1
    pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
    
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    pred["ppl/zlib"] = np.log(p1) / zlib_entropy
    
    # Min-K% Prob logic
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = max(int(len(all_prob) * ratio), 1)
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{int(ratio*100)}% Prob"] = -np.mean(topk_prob).item()

    ex["pred"] = pred
    return ex

def evaluate_data(test_data, model, tokenizer):
    print(f"\nStarting model inference on {len(test_data)} strictly balanced chunks...")
    all_output = []
    for ex in tqdm(test_data): 
        text = ex["text"]
        new_ex = inference(model, tokenizer, text, ex)
        all_output.append(new_ex)
    return all_output


# ==========================================
# 4. MAIN PIPELINE
# ==========================================
if __name__ == '__main__':
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Model and Tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, return_dict=True, device_map='auto')
    model.eval()

    # Step 1: Prepare balanced data from CSVs
    data_list = prepare_balanced_data(tokenizer)

    # Step 2: Run inference 
    all_output = evaluate_data(data_list, model, tokenizer)
    
    # Step 3: Calculate AUCs and output results
    print("\nCalculating AUCs and generating plots...")
    fig_fpr_tpr(all_output, OUTPUT_DIR)
    
    print(f"Done! Clean results saved in {OUTPUT_DIR}")