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
from watermark.morphmark.morphmark import MorphMark

# =========================================================
# ENV
# =========================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================================================
# CLI
# =========================================================
parser = argparse.ArgumentParser()

parser.add_argument("--num_samples", type=int, default=500)
parser.add_argument("--prompt_tokens", type=int, default=50)
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.7)

parser.add_argument(
    "--gamma_values",
    type=str,
    default="0.1,0.2,0.3,0.4,0.5",
    help="Comma-separated gamma values, e.g. 0.1,0.2,0.3,0.4,0.5",
)

parser.add_argument(
    "--delta",
    type=float,
    default=2.0,
    help="Delta value only for folder naming and internal override if method/config has delta.",
)

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

GAMMA_LIST = [float(x.strip()) for x in args.gamma_values.split(",") if x.strip()]

# =========================================================
# DEVICE
# =========================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print("Running on device:", DEVICE)
print("Gamma sweep:", GAMMA_LIST)
print("Delta:", args.delta)
print("SynthID removed because it is sampling-based.")

# =========================================================
# PATHS
# =========================================================
BASE = Path("/home/sy/markove_markllm/markllm/MarkLLM-main/MarkLLM-main")
MODEL_PATH = Path(args.model_path)
INPUT_JSON = Path(args.input_json)

DATASET_NAME = INPUT_JSON.stem
MODEL_NAME = MODEL_PATH.name

# Parent output folder
OUT_ROOT = BASE / "DBSF_experiment/outputs_0.25_0.5_c4_realnewslike_Qwen3-8B/Gamma Sweep"
OUT_ROOT.mkdir(exist_ok=True, parents=True)

# ================= CONFIG FILES =================
KGW_CFG = BASE / "config/KGW.json"
DBSF_CFG = BASE / "config/DBSF.json"
SWEET_CFG = BASE / "config/SWEET.json"
EWD_CFG = BASE / "config/EWD.json"
DIP_CFG = BASE / "config/DIP.json"
UNBIASED_CFG = BASE / "config/Unbiased.json"
MORPHMARK_CFG = BASE / "config/MorphMark.json"

for p in [
    MODEL_PATH,
    INPUT_JSON,
    KGW_CFG,
    DBSF_CFG,
    SWEET_CFG,
    EWD_CFG,
    DIP_CFG,
    UNBIASED_CFG,
    MORPHMARK_CFG,
]:
    if not Path(p).exists():
        raise FileNotFoundError(f"Missing: {p}")

# =========================================================
# LOAD DATA
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


def extract_text(item: dict) -> str:
    for key in ["text", "document", "article", "content", "prefix"]:
        if key in item:
            return item[key]

    if "question" in item:
        q = item["question"]

        if "human_answers" in item and item["human_answers"]:
            return q + " " + item["human_answers"][0]

        if "chatgpt_answers" in item and item["chatgpt_answers"]:
            return q + " " + item["chatgpt_answers"][0]

        return q

    raise ValueError(f"Unknown format: {list(item.keys())}")


data = load_any_json(INPUT_JSON)[:args.num_samples]
print("Total loaded:", len(data))

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
    use_fast=False,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
    torch_dtype=DTYPE,
).to(DEVICE)

model.eval()
model.generation_config.pad_token_id = tokenizer.eos_token_id

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
# WATERMARK OBJECTS FROM ORIGINAL CONFIG JSON
# =========================================================
kgw = KGW(KGWConfig(KGW_CFG, tf_cfg), tf_cfg)
sweet = SWEET(SWEETConfig(SWEET_CFG, tf_cfg), tf_cfg)
ewd = EWD(EWDConfig(EWD_CFG, tf_cfg), tf_cfg)
dip = DIP(DIPConfig(DIP_CFG, tf_cfg), tf_cfg)
unbiased = UnbiasedWatermark(str(UNBIASED_CFG), tf_cfg)
morphmark = MorphMark(str(MORPHMARK_CFG), tf_cfg)
dbsf = DBSF(DBSFConfig(DBSF_CFG, tf_cfg), tf_cfg)

methods = {
    "kgw": kgw,
    "sweet": sweet,
    "ewd": ewd,
    "dip": dip,
    "unbiased": unbiased,
    "morphmark": morphmark,
    "dbsf": dbsf,
}

def get_prompt_tokens(text: str, n_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def strip_prompt(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt):].lstrip()
    return full_text


def save_json(path: Path, records: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def generate_with_method(wm, prompt: str) -> str:
    if hasattr(wm, "generate_watermarked_text"):
        return wm.generate_watermarked_text(prompt)

    if hasattr(wm, "generate_unwatermarked_text"):
        return wm.generate_unwatermarked_text(prompt)

    raise AttributeError(f"{wm.__class__.__name__} has no generation method")


def set_gamma_and_delta(wm, gamma_value: float, delta_value: float):
    """
    Internal override only.
    No config file is edited.
    No temporary config is created.
    """

    # gamma override
    if hasattr(wm, "gamma"):
        wm.gamma = gamma_value

    if hasattr(wm, "greenlist_ratio"):
        wm.greenlist_ratio = gamma_value

    if hasattr(wm, "green_ratio"):
        wm.green_ratio = gamma_value

    if hasattr(wm, "gamma_value"):
        wm.gamma_value = gamma_value

    if hasattr(wm, "config"):
        if hasattr(wm.config, "gamma"):
            wm.config.gamma = gamma_value

        if hasattr(wm.config, "greenlist_ratio"):
            wm.config.greenlist_ratio = gamma_value

        if hasattr(wm.config, "green_ratio"):
            wm.config.green_ratio = gamma_value

        if hasattr(wm.config, "gamma_value"):
            wm.config.gamma_value = gamma_value

    # delta override
    if hasattr(wm, "delta"):
        wm.delta = delta_value

    if hasattr(wm, "config") and hasattr(wm.config, "delta"):
        wm.config.delta = delta_value


def value_to_str(x: float) -> str:
    """
    Keep folder style like:
    0.1
    0.2
    2.0
    """
    return f"{x:.1f}"


# =========================================================
# GAMMA SWEEP
# =========================================================
for gamma_value in GAMMA_LIST:

    print("\n========================================")
    print(f"Running GAMMA = {gamma_value}")
    print("========================================")

    gamma_str = value_to_str(gamma_value)
    delta_str = value_to_str(args.delta)

    # Your requested folder style:
    # outputs_0.1_2.0_c4_realnewslike_opt-6.7B
    OUT_DIR = OUT_ROOT / f"outputs_{gamma_str}_{delta_str}_{DATASET_NAME}_{MODEL_NAME}"
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    # ---------------- VANILLA ----------------
    print("Running vanilla...")
    vanilla_records = []

    for i, item in enumerate(tqdm(data, desc=f"vanilla_g{gamma_str}")):
        text = extract_text(item)
        prompt = get_prompt_tokens(text, args.prompt_tokens)

        enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out_ids = model.generate(**enc, **tf_cfg.gen_kwargs)

        full_gen = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        continuation = strip_prompt(full_gen, prompt)

        vanilla_records.append(
            {
                "id": i,
                "gamma": gamma_value,
                "delta": args.delta,
                "method": "vanilla",
                "original": prompt,
                "sampled": continuation,
            }
        )

    save_json(OUT_DIR / "vanilla.json", vanilla_records)

    # ---------------- WATERMARK METHODS ----------------
    for name, wm in methods.items():

        print(f"\n----- Running {name} | gamma = {gamma_value}, delta = {args.delta} -----")

        set_gamma_and_delta(wm, gamma_value, args.delta)

        records = []

        for i, item in enumerate(tqdm(data, desc=f"{name}_g{gamma_str}")):
            text = extract_text(item)
            prompt = get_prompt_tokens(text, args.prompt_tokens)

            with torch.no_grad():
                full_gen = generate_with_method(wm, prompt)

            continuation = strip_prompt(full_gen, prompt)

            records.append(
                {
                    "id": i,
                    "gamma": gamma_value,
                    "delta": args.delta,
                    "method": name,
                    "original": prompt,
                    "sampled": continuation,
                }
            )

        save_json(OUT_DIR / f"{name}.json", records)

    print(f"\nSaved gamma {gamma_str} results to:")
    print(OUT_DIR)

print("\nALL DONE.")