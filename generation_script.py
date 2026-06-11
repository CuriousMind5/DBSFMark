# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.transformers_config import TransformersConfig

# ================= METHODS =================
from watermark.kgw.kgw import KGW, KGWConfig
from watermark.dbsf.dbsf import DBSF, DBSFConfig
from watermark.sweet.sweet import SWEET, SWEETConfig
from watermark.ewd.ewd import EWD, EWDConfig
from watermark.dip.dip import DIP, DIPConfig
from watermark.unbiased.unbiased import UnbiasedWatermark
from watermark.synthid.synthid import SynthID
from watermark.morphmark.morphmark import MorphMark

# =========================================================
# ENV
# =========================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================================================
# CLI DESIGN
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=500)
parser.add_argument("--prompt_tokens", type=int, default=50)
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument(
    "--model_path",
    type=str,
    default="/home/sy/watermark_unlearning_experiment/models/Qwen3-8B",
)
parser.add_argument(
    "--input_json",
    type=str,
    default="/home/sy/watermark_unlearning_experiment/data/c4_realnewslike.json",
)
args = parser.parse_args()

# =========================================================
# DEVICE USED
# =========================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
print("Running on device:", DEVICE)

# =========================================================
# PATHS
# =========================================================
BASE = Path("/home/sy/markove_markllm/markllm/MarkLLM-main/MarkLLM-main")
MODEL_PATH = Path(args.model_path)
INPUT_JSON = Path(args.input_json)

MODEL_TAG = MODEL_PATH.name.replace("/", "_")

OUT_DIR = BASE / (
    f"DBSF_experiment/outputs_0.25_0.5_{INPUT_JSON.stem}_{MODEL_TAG}"
)
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ================= CONFIG FILES =================
KGW_CFG = BASE / "config/KGW.json"
DBSF_CFG = BASE / "config/DBSF.json"
SWEET_CFG = BASE / "config/SWEET.json"
EWD_CFG = BASE / "config/EWD.json"
DIP_CFG = BASE / "config/DIP.json"
UNBIASED_CFG = BASE / "config/Unbiased.json"
SYNTHID_CFG = BASE / "config/SynthID.json"
MORPHMARK_CFG = BASE / "config/MorphMark.json"

# =========================================================
# SANITY CHECK FOR THE PATHS
# =========================================================
for p in [
    MODEL_PATH,
    INPUT_JSON,
    KGW_CFG,
    DBSF_CFG,
    SWEET_CFG,
    EWD_CFG,
    DIP_CFG,
    UNBIASED_CFG,
    SYNTHID_CFG,
    MORPHMARK_CFG,
]:
    if not Path(p).exists():
        raise FileNotFoundError(f"Missing: {p}")

# =========================================================
# JSON LOADER
# =========================================================
def load_any_json(path: Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        if first == "[":
            data = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

    return data

# =========================================================
# TEXT EXTRACTION TYPES
# =========================================================
def extract_text(item: dict) -> str:
    for key in ["text", "document", "article", "content", "prefix"]:
        if key in item:
            return item[key]

    if "question" in item:
        q = item["question"]

        if "human_answers" in item:
            return q + " " + item["human_answers"][0]

        if "chatgpt_answers" in item:
            return q + " " + item["chatgpt_answers"][0]

        return q

    raise ValueError("Unknown format")

# =========================================================
# MODEL LOADER
# =========================================================
def load_model_and_tokenizer(model_path: Path, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=DTYPE,
    ).to(device)

    model.eval()
    return model, tokenizer

# =========================================================
# LOAD DATA
# =========================================================
data = load_any_json(INPUT_JSON)[:args.num_samples]

# =========================================================
# LOAD MODEL
# =========================================================
model, tokenizer = load_model_and_tokenizer(MODEL_PATH, DEVICE)

tf_cfg = TransformersConfig(
    model=model,
    tokenizer=tokenizer,
    device=DEVICE,
    vocab_size=model.get_output_embeddings().weight.shape[0],
    max_new_tokens=args.max_new_tokens,
    temperature=args.temperature,
    do_sample=True,
)

tf_cfg.gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

# =========================================================
# WATERMARK OBJECTS
# =========================================================
kgw = KGW(KGWConfig(KGW_CFG, tf_cfg), tf_cfg)
sweet = SWEET(SWEETConfig(SWEET_CFG, tf_cfg), tf_cfg)
ewd = EWD(EWDConfig(EWD_CFG, tf_cfg), tf_cfg)
dip = DIP(DIPConfig(DIP_CFG, tf_cfg), tf_cfg)
unbiased = UnbiasedWatermark(str(UNBIASED_CFG), tf_cfg)

synthid = SynthID(str(SYNTHID_CFG), tf_cfg)
morphmark = MorphMark(str(MORPHMARK_CFG), tf_cfg)
dbsf = DBSF(DBSFConfig(DBSF_CFG, tf_cfg), tf_cfg)

# =========================================================
# METHODS
# =========================================================
methods = {
    "kgw": kgw,
    "sweet": sweet,
    "ewd": ewd,
    "dip": dip,
    "unbiased": unbiased,
    "synthid": synthid,
    "morphmark": morphmark,
    "dbsf": dbsf,
}

# =========================================================
# HELPERS
# =========================================================
def get_prompt_tokens(text, n):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n]
    return tokenizer.decode(ids, skip_special_tokens=True)

def strip_prompt(full, prompt):
    return full[len(prompt):].lstrip() if full.startswith(prompt) else full

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def generate_with_method(wm, prompt):
    if hasattr(wm, "generate_watermarked_text"):
        return wm.generate_watermarked_text(prompt)

    if hasattr(wm, "generate_unwatermarked_text"):
        return wm.generate_unwatermarked_text(prompt)

    raise AttributeError(f"{wm.__class__.__name__} has no generation method")

# =========================================================
# NON WATERMARKED TEXT
# =========================================================
print("\n===== Running vanilla =====")
vanilla_records = []

for i, item in enumerate(tqdm(data, desc="vanilla")):
    text = extract_text(item)
    prompt = get_prompt_tokens(text, args.prompt_tokens)

    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model.generate(**tf_cfg.gen_kwargs, **enc)

    full = tokenizer.decode(out[0], skip_special_tokens=True)
    continuation = strip_prompt(full, prompt)

    vanilla_records.append(
        {
            "id": i,
            "original": prompt,
            "sampled": continuation,
        }
    )

save_json(OUT_DIR / "vanilla.json", vanilla_records)

# =========================================================
# WATERMARK LOOP
# =========================================================
for name, wm in methods.items():
    print(f"\nRunning {name}")
    records = []

    for i, item in enumerate(tqdm(data, desc=name)):
        text = extract_text(item)
        prompt = get_prompt_tokens(text, args.prompt_tokens)

        with torch.no_grad():
            full = generate_with_method(wm, prompt)

        continuation = strip_prompt(full, prompt)

        records.append(
            {
                "id": i,
                "original": prompt,
                "sampled": continuation,
            }
        )

    save_json(OUT_DIR / f"{name}.json", records)

print("\nDone.")
