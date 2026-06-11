# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import time
import torch
import warnings
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- FORCE BOTH GPUs ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# ---------------- NLTK ----------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ---------------- LOCAL PATHS ----------------
DIPPER_PATH = "/home/sy/watermark_unlearning_experiment/models/Dipper/"
T5_PATH     = "/home/sy/watermark_unlearning_experiment/models/google-t5/t5-large/"

INPUT_DIR = "/home/sy/markove_markllm/markllm/MarkLLM-main/MarkLLM-main/outputs_reddit_eli5_opt-6.7B_semanticRAW_0.50_0.9_dip_un/"
OUT_DIR   = "/home/sy/markove_markllm/markllm/MarkLLM-main/MarkLLM-main/outputs_reddit_eli5_opt-6.7B_semanticRAW_0.50_0.9_dip_un/Dipper2"

# ---------------- PARAMS ----------------
LEX_DIVERSITY   = 60
ORDER_DIVERSITY = 60
SENT_INTERVAL   = 3


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------- DIPPER CLASS ----------------
class DipperParaphraser(object):
    def __init__(self, model_path, t5_path, verbose=True):
        t0 = time.time()

        self.tokenizer = T5Tokenizer.from_pretrained(t5_path)

        self.model = T5ForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",                
            torch_dtype=torch.float16,
        )

        if verbose:
            print(f"{model_path} model loaded in {time.time() - t0:.2f}s")
            print("Device map:", self.model.hf_device_map)

        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):

        assert lex_diversity in [0, 20, 40, 60, 80, 100]
        assert order_diversity in [0, 20, 40, 60, 80, 100]

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])

            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            batch = self.tokenizer([final_input_text], return_tensors="pt")

            with torch.inference_mode():
                outputs = self.model.generate(**batch, **kwargs)

            text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            prefix += " " + text
            output_text += " " + text

        return output_text.strip()


# ---------------- MAIN ----------------
def main():

    ensure_dir(OUT_DIR)

    print(">>> Loading DIPPER (multi-GPU)")
    dipper = DipperParaphraser(DIPPER_PATH, T5_PATH)

    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".json")])
    print(">>> Paraphrasing files:", files)

    for fname in files:

        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUT_DIR, fname.replace(".json", "_dipper_attack.json"))

        print(">>> Processing:", fname)

        data = load_json(in_path)
        new_records = []

        for rec in tqdm(data, desc=fname):

            watermarked_text = rec.get("sampled", "").strip()
            rid = rec.get("id", -1)

            if not watermarked_text:
                continue

            attacked = dipper.paraphrase(
                watermarked_text,
                lex_diversity=LEX_DIVERSITY,
                order_diversity=ORDER_DIVERSITY,
                prefix="",
                sent_interval=SENT_INTERVAL,
                do_sample=True,
                top_p=0.75,
                top_k=None,
                max_length=512,
            )

            new_records.append({
                "id": rid,
                "original": watermarked_text,
                "sampled": attacked,
            })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new_records, f, ensure_ascii=False, indent=2)

        print("Saved:", out_path)

    print("DONE")


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    torch.manual_seed(15485863)
    main()

