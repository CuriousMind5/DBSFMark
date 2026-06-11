# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import random
import warnings
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- PATHS ----------------
INPUT_DIR = "/home/sy/markove_markllm/markllm/MarkLLM-main/MarkLLM-main/DBSF_experiment/outputs_0.25_0.5_c4_realnewslike_opt-6.7B"

# 👉 change folder name automatically based on ratio
SUB_RATIO = 0.5 

OUT_DIR = f"{INPUT_DIR}/WordSBERT_{int(SUB_RATIO*100)}"

BERT_PATH = "/home/sy/watermark_unlearning_experiment/models/google/bert-base-uncased"

# ---------------- LOAD MODEL ----------------
print(">>> Loading BERT...")
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
model = BertForMaskedLM.from_pretrained(BERT_PATH).to(DEVICE)
model.eval()

# ---------------- HELPERS ----------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------- WORD-S-BERT ----------------
def bert_substitution(text, ratio=0.3):

    words = text.split()
    if len(words) == 0:
        return text

    num_replace = int(len(words) * ratio)
    indices = random.sample(range(len(words)), num_replace)

    for i in indices:

        masked_words = words.copy()
        masked_words[i] = "[MASK]"
        masked_text = " ".join(masked_words)

        inputs = tokenizer(masked_text, return_tensors="pt").to(DEVICE)

        mask_idx = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_idx) == 0:
            continue

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0, mask_idx[0]]
        top_tokens = torch.topk(logits, 5).indices
        predicted_words = tokenizer.convert_ids_to_tokens(top_tokens)

        # pick first valid token
        for tok in predicted_words:
            if tok.isalpha():
                words[i] = tok
                break

    return " ".join(words)

# ---------------- MAIN ----------------
def main():

    ensure_dir(OUT_DIR)

    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".json")])
    print(">>> Processing files:", files)

    for fname in files:

        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUT_DIR, fname)

        print(f"\n>>> Processing: {fname}")

        data = load_json(in_path)
        new_records = []

        for rec in tqdm(data, desc=fname):

            watermarked_text = rec.get("sampled", "").strip()
            rid = rec.get("id", -1)

            if not watermarked_text:
                continue

            # ✅ Apply Word-S-BERT
            attacked_text = bert_substitution(watermarked_text, ratio=SUB_RATIO)

            new_records.append({
                "id": rid,
                "original": watermarked_text,
                "sampled": attacked_text
            })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new_records, f, ensure_ascii=False, indent=2)

        print("Saved:", out_path)

    print("DONE")

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    random.seed(15485863)
    main()
