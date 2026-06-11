# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import random
import warnings
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- NLTK SETUP ----------------
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

# ---------------- PATHS ----------------
INPUT_DIR = "/home/sy/markove_markllm/markllm/MarkLLM-main/MarkLLM-main/DBSF_experiment/outputs_0.25_0.5_c4_realnewslike_Qwen3-8B"
OUT_DIR   = "/home/sy/markove_markllm/markllm/MarkLLM-main/MarkLLM-main/DBSF_experiment/outputs_0.25_0.5_c4_realnewslike_Qwen3-8B/WordSynonym_50"

# ---------------- PARAMETERS ----------------
SUB_RATIO = 0.5   

# ---------------- HELPERS ----------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------- SYNONYM ATTACK ----------------
def get_synonyms(word):
    synsets = wordnet.synsets(word)
    synonyms = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                synonyms.add(name)
    return list(synonyms)

def synonym_substitution(text, ratio=0.5):
    words = text.split()
    if len(words) == 0:
        return text

    replaceable_idx = []
    for i, w in enumerate(words):
        if get_synonyms(w):
            replaceable_idx.append(i)

    num_replace = min(int(len(words) * ratio), len(replaceable_idx))

    if num_replace > 0:
        chosen_idx = random.sample(replaceable_idx, num_replace)
        for i in chosen_idx:
            syns = get_synonyms(words[i])
            if syns:
                words[i] = random.choice(syns)

    return " ".join(words)

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

            # ✅ Apply Word-S-DICT (50%)
            attacked_text = synonym_substitution(watermarked_text, ratio=SUB_RATIO)

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
