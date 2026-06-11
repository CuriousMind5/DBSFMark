# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import random
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- PATHS ----------------
INPUT_DIR = "/home/sy/markove_markllm/markllm/MarkLLM-main/MarkLLM-main/DBSF_experiment/outputs_0.25_0.5_c4_realnewslike_opt-6.7B"
OUT_DIR   = "/home/sy/markove_markllm/markllm/MarkLLM-main/MarkLLM-main/DBSF_experiment/outputs_0.25_0.5_c4_realnewslike_opt-6.7B/WordDeletion_30"

# ---------------- PARAMETERS ----------------
DELETION_RATIO = 0.3 

# ---------------- HELPERS ----------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------- WORD DELETION ATTACK ----------------
def word_deletion(text, ratio=0.3):
    words = text.split()
    if len(words) == 0:
        return text
    kept_words = [w for w in words if random.random() >= ratio]
    return " ".join(kept_words)

# ---------------- MAIN ----------------
def main():

    ensure_dir(OUT_DIR)

    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".json")])
    print(">>> Processing files:", files)

    for fname in files:

        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUT_DIR, fname)

        print(">>> Processing:", fname)

        data = load_json(in_path)
        new_records = []

        for rec in tqdm(data, desc=fname):

            watermarked_text = rec.get("sampled", "").strip()
            rid = rec.get("id", -1)

            if not watermarked_text:
                continue

            attacked_text = word_deletion(watermarked_text, ratio=DELETION_RATIO)

            new_records.append({
                "id": rid,
                "original": watermarked_text,   # watermark text
                "sampled": attacked_text        # attacked text
            })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new_records, f, ensure_ascii=False, indent=2)

        print("Saved:", out_path)

    print("DONE")

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    random.seed(15485863)   # reproducibility
    main()
