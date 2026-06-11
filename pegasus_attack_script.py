# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import torch
import warnings
from tqdm import tqdm
import nltk
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ---------------- LOCAL PATHS ----------------
PEGASUS_PATH = "/home/sy/watermark_unlearning_experiment/models/pegasus/"
INPUT_DIR = "/home/sy/markove_markllm/markllm/MarkLLM-main/MarkLLM-main/Utility/outputs_c4_realnewslike_opt-6.7B_UG500_/"
OUT_DIR = "/home/sy/markove_markllm/markllm/MarkLLM-main/MarkLLM-main/Utility/outputs_c4_realnewslike_opt-6.7B_UG500_/Pegasus1"

# ---------------- PARAMETERS ----------------
MAX_LEN = 60
NUM_BEAMS = 25
NUM_RETURN = 25
TEMPERATURE = 1.5

# Generate 25 candidates
KEEP_FINAL = 1
KEEP_INDEX = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- HELPERS ----------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def paraphrase_sentence_multi(model, tokenizer, sent):
    """
    Generate 25 Pegasus paraphrases for one sentence.
    We later keep only one candidate from these 25.
    """
    batch = tokenizer(
        [sent],
        truncation=True,
        padding="longest",
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **batch,
            max_length=MAX_LEN,
            num_beams=NUM_BEAMS,
            num_return_sequences=NUM_RETURN,
            temperature=TEMPERATURE,
        )

    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    return texts  # length = 25


# ---------------- MAIN ----------------
def main():

    ensure_dir(OUT_DIR)

    print(">>> Loading Pegasus (local)")
    tokenizer = PegasusTokenizer.from_pretrained(
        PEGASUS_PATH,
        local_files_only=True,
    )

    model = PegasusForConditionalGeneration.from_pretrained(
        PEGASUS_PATH,
        local_files_only=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)

    model.eval()

    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".json")])

    print(">>> Paraphrasing files:", files)
    print(f">>> Generate {NUM_RETURN} candidates per sentence, keep candidate index {KEEP_INDEX}")

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

            sents = nltk.sent_tokenize(watermarked_text)

            # One final paraphrased continuation per original record
            final_para = ""

            for s in sents:
                all_paraphrases = paraphrase_sentence_multi(
                    model=model,
                    tokenizer=tokenizer,
                    sent=s,
                )  # generates 25 candidates

                # Keep only one candidate from the 25 outputs
                chosen = all_paraphrases[KEEP_INDEX]

                final_para += " " + chosen

            new_records.append({
                "id": rid,
                "original": watermarked_text,       # before attack
                "sampled": final_para.strip(),      # after Pegasus attack
            })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new_records, f, ensure_ascii=False, indent=2)

        print("Saved:", out_path)

    print("DONE")


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    main()
