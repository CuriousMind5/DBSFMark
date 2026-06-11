# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import torch
import warnings
from tqdm import tqdm
import nltk

from transformers import MarianMTModel, MarianTokenizer

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- GPU SETUP ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ---------------- LOCAL MODEL PATHS ----------------
MODEL_EN_ZH = "/home/sy/watermark_unlearning_experiment/models/Helsinki-NLP/opus-mt-en-fr/"
MODEL_ZH_EN = "/home/sy/watermark_unlearning_experiment/models/Helsinki-NLP/opus-mt-fr-en/"

# ---------------- DATA PATHS ----------------
INPUT_DIR = (
    "/home/sy/markove_markllm/markllm/MarkLLM-main/MarkLLM-main/"
    "DBSF_experiment/outputs_0.25_0.5_c4_realnewslike_opt-6.7B"
)

OUT_DIR = (
    "/home/sy/markove_markllm/markllm/MarkLLM-main/MarkLLM-main/"
    "DBSF_experiment/outputs_0.25_0.5_c4_realnewslike_opt-6.7B/"
    "English-French-English"
)

INPUT_FILES = [
    "vanilla.json",
    "kgw.json",
    "sweet.json",
    "ewd.json",
    "dip.json",
    "unbiased.json",
    "synthid.json",
    "morphmark.json",
    "dbsf.json",
]

GEN_KWARGS = dict(
    num_beams=2,
    max_new_tokens=280,
    early_stopping=True,
)

MAX_INPUT_LEN = 512

# ---------------- HELPERS ----------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def check_path(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def list_existing_input_files():
    existing = []

    for fname in INPUT_FILES:
        fpath = os.path.join(INPUT_DIR, fname)

        if os.path.exists(fpath):
            existing.append(fname)
        else:
            print("Missing input file, skipping:", fpath)

    return existing


# ---------------- TRANSLATION ----------------
@torch.inference_mode()
def translate_once(texts, tokenizer, model):
    batch = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
        padding=True,
    )

    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    out = model.generate(
        **batch,
        **GEN_KWARGS,
    )

    decoded = tokenizer.batch_decode(
        out,
        skip_special_tokens=True,
    )

    return decoded


# ---------------- MAIN ----------------
def main():

    check_path(INPUT_DIR, "INPUT_DIR")
    check_path(MODEL_EN_ZH, "MODEL_EN_ZH")
    check_path(MODEL_ZH_EN, "MODEL_ZH_EN")

    ensure_dir(OUT_DIR)

    print(">>> Loading LOCAL OPUS translation models on", DEVICE)
    print(">>> INPUT_DIR:", INPUT_DIR)
    print(">>> OUT_DIR:", OUT_DIR)

    tok1 = MarianTokenizer.from_pretrained(
        MODEL_EN_ZH,
        local_files_only=True,
    )

    mod1 = MarianMTModel.from_pretrained(
        MODEL_EN_ZH,
        local_files_only=True,
        dtype=DTYPE,
    ).to(DEVICE).eval()

    tok2 = MarianTokenizer.from_pretrained(
        MODEL_ZH_EN,
        local_files_only=True,
    )

    mod2 = MarianMTModel.from_pretrained(
        MODEL_ZH_EN,
        local_files_only=True,
        dtype=DTYPE,
    ).to(DEVICE).eval()

    files = list_existing_input_files()

    print(">>> Translating files:", files)

    if not files:
        raise RuntimeError(
            "No input JSON files found. Check INPUT_DIR and INPUT_FILES."
        )

    for fname in files:

        in_path = os.path.join(INPUT_DIR, fname)

        out_path = os.path.join(
            OUT_DIR,
            fname.replace(".json", ".json")
        )

        print("\n>>> Processing:", fname)
        print(">>> Input :", in_path)
        print(">>> Output:", out_path)

        data = load_json(in_path)
        new_records = []

        for rec in tqdm(data, desc=fname):

            text = rec.get("sampled", "").strip()
            rid = rec.get("id", -1)

            if not text:
                continue

            # -------- ENGLISH → CHINESE --------
            mid_text = translate_once(
                texts=[text],
                tokenizer=tok1,
                model=mod1,
            )

            # -------- CHINESE → ENGLISH --------
            final_text = translate_once(
                texts=mid_text,
                tokenizer=tok2,
                model=mod2,
            )

            attacked = final_text[0].strip()

            new_records.append({
                "id": rid,
                "original": text,
                "sampled": attacked,
            })

        save_json(out_path, new_records)

        print("Saved:", out_path)

    print("\nDONE")


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    torch.manual_seed(15485863)
    main()