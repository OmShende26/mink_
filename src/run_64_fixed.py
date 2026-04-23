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
SEEN_CSV_PATH = "/content/multi_book_dataset.csv"       # Update with your actual path
UNSEEN_CSV_PATH = "/content/newbooks_dataset.csv"   # Update with your actual path
OUTPUT_DIR = "out/custom_eval_fixed_64"

# Strict fixed length constraint
CHUNK_SIZE = 64


# ==========================================
# 2. DATA PREPARATION (Strict Fixed Length)
# ==========================================
def process_csv_fixed(csv_path, label, tokenizer, chunk_size):
    """Reads a CSV, tokenizes the text, and slices it into strict fixed lengths."""
    df = pd.read_csv(csv_path)
    chunks = []
    
    print(f"Slicing {csv_path} into strict {chunk_size}-token chunks...")
    for text in tqdm(df['text'].dropna()):
        tokens = tokenizer.encode(str(text), add_special_tokens=False)
        
        # Iterate through the tokens and grab exact chunks
        # The + 1 ensures we don't accidentally grab a final chunk smaller than CHUNK_SIZE
        for i in range(0, len(tokens) - chunk_size + 1, chunk_size):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens)
            
            # Save the chunk along with its label (1 for seen, 0 for unseen)
            chunks.append({"text": chunk_text, "label": label})
            
    return chunks

def prepare_data(tokenizer):
    """Processes both CSVs and shuffles them together."""
    seen_data = process_csv_fixed(SEEN_CSV_PATH, label=1, tokenizer=tokenizer, chunk_size=CHUNK_SIZE)
    unseen_data = process_csv_fixed(UNSEEN_CSV_PATH, label=0, tokenizer=tokenizer, chunk_size=CHUNK_SIZE)
    
    all_data = seen_data + unseen_data
    random.seed(42)
    random.shuffle(all_data)
    print(f"\nTotal prepared snippets: {len(all_data)} ({len(seen_data)} seen, {len(unseen_data)} unseen)")
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
    print(f"\nStarting model inference on {len(test_data)} chunks...")
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

    # Step 1: Prepare the data from CSVs
    data_list = prepare_data(tokenizer)

    # Step 2: Run inference to calculate probabilities
    all_output = evaluate_data(data_list, model, tokenizer)
    
    # Step 3: Calculate AUCs and output results using the authors' existing tool
    print("\nCalculating AUCs and generating plots...")
    fig_fpr_tpr(all_output, OUTPUT_DIR)
    
    print(f"Done! Results saved in {OUTPUT_DIR}")