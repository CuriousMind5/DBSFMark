# -*- coding: utf-8 -*-
from __future__ import annotations

# ================= GPU =================
USE_MULTI_GPU = True
DEVICE = "cuda:0"
DTYPE = "float16"
BATCH_SIZE = 4

MAX_MEMORY = {
    0: "22GiB",
    1: "22GiB",
    "cpu": "60GiB",
}

# ================= PATHS =================
INPUT_DIR = "/home/sy/markove_markllm/markllm/MarkLLM-main/MarkLLM-main/DBSF_experiment/outputs_0.25_0.5_c4_realnewslike_opt-6.7B/"
OUT_DIR   = INPUT_DIR + "PPL/opt-6.7B"
MODEL_PATH = "/home/sy/watermark_unlearning_experiment/models/facebook/opt-13B"

# ================= FILES =================
FILES = [
    "vanilla.json",
    "kgw.json",
    "dip.json",
    "ewd.json",
    "morphmark.json",
    "sweet.json",
    "unbiased.json",
    "synthid.json",
    "dbsf.json",   # moved last
]

PROMPT_FIELD = "original"
TEXT_FIELD   = "sampled"

# ================= CSV OUTPUTS =================
OUT_ALL_CSV = OUT_DIR + "ppl_all_results.csv"
OUT_SUMMARY_CSV = OUT_DIR + "ppl_summary.csv"

# ==================================================
import os
import csv
import json
import math
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from transformers import AutoTokenizer, AutoModelForCausalLM

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- DISPLAY NAMES ----------------
METHOD_DISPLAY = OrderedDict([
    ("vanilla", "Vanilla"),
    ("kgw", "KGW"),
    ("dip", "DIP"),
    ("ewd", "EWD"),
    ("morphmark", "MorphMark"),
    ("sweet", "SWEET"),
    ("unbiased", "Unbiased"),
    ("synthid", "SynthID"),
    ("dbsf", "DBSF"),   # moved last
])

# ---------------- TOKENIZER ----------------
def load_tok(path: str):
    tok = AutoTokenizer.from_pretrained(
        path,
        use_fast=True,
        local_files_only=True,
    )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    return tok

# ---------------- MODEL ----------------
def load_model(path: str):
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    if USE_MULTI_GPU:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=dtype_map[DTYPE],
            device_map="auto",
            max_memory=MAX_MEMORY,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=dtype_map[DTYPE],
        ).to(DEVICE)

    model.eval()
    return model

# ---------------- HELPERS ----------------
def strip_bos(ids, bos):
    if bos is not None and len(ids) > 0 and ids[0] == bos:
        return ids[1:]
    return ids

def safe_mean(values):
    return float(np.mean(values)) if values else 0.0

def safe_std(values):
    return float(np.std(values)) if values else 0.0

def safe_median(values):
    return float(np.median(values)) if values else 0.0

def safe_min(values):
    return float(np.min(values)) if values else 0.0

def safe_max(values):
    return float(np.max(values)) if values else 0.0

def safe_percentile(values, q):
    return float(np.percentile(values, q)) if values else 0.0

# ---------------- CE/PPL ----------------
@torch.no_grad()
def compute_ce_ppl(model, tok, prompts, conts, max_ctx):
    enc = tok(
        [p + c for p, c in zip(prompts, conts)],
        add_special_tokens=False,
    )

    bos = tok.bos_token_id
    seqs = []

    for ids in enc["input_ids"]:
        ids = strip_bos(ids, bos)

        if len(ids) > max_ctx:
            ids = ids[-max_ctx:]

        seqs.append(torch.tensor(ids))

    if not seqs:
        return [], []

    T = max(len(s) for s in seqs) - 1

    if T <= 0:
        return [], []

    pad = tok.pad_token_id
    xs, ys, ms = [], [], []

    for s in seqs:
        x = s[:-1]
        y = s[1:]
        m = torch.ones_like(x)

        pad_len = T - x.size(0)

        if pad_len > 0:
            x = torch.cat([x, torch.full((pad_len,), pad)])
            y = torch.cat([y, torch.full((pad_len,), pad)])
            m = torch.cat([m, torch.zeros(pad_len)])

        xs.append(x)
        ys.append(y)
        ms.append(m)

    device = next(model.parameters()).device

    X = torch.stack(xs).to(device)
    Y = torch.stack(ys).to(device)
    M = torch.stack(ms).to(device)

    logits = model(input_ids=X, attention_mask=M).logits
    logprob = torch.log_softmax(logits, dim=-1)

    tok_lp = logprob.gather(-1, Y.unsqueeze(-1)).squeeze(-1)
    mask = M.float()

    nll = -(tok_lp * mask).sum(dim=1)
    N = mask.sum(dim=1)

    ce = nll / torch.clamp_min(N, 1.0)

    return ce.cpu().tolist(), N.cpu().tolist()

# ---------------- FILE PROCESS ----------------
def process_file(model, tok, path, batch, max_ctx):
    name = os.path.basename(path).replace(".json", "")
    out_path = os.path.join(OUT_DIR, f"{name}__ppl_all.jsonl")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    with open(out_path, "w", encoding="utf-8") as w:
        buf_p = []
        buf_c = []
        buf_ids = []

        for obj_idx, obj in enumerate(tqdm(data, desc=name)):
            p = obj.get(PROMPT_FIELD, "")
            c = obj.get(TEXT_FIELD, "")

            if not isinstance(c, str) or not c.strip():
                continue

            sample_id = obj.get("id", obj_idx)

            buf_p.append(p)
            buf_c.append(c)
            buf_ids.append(sample_id)

            if len(buf_p) == batch:
                ces, toks = compute_ce_ppl(model, tok, buf_p, buf_c, max_ctx)

                for sample_id, ce, tokc in zip(buf_ids, ces, toks):
                    ppl = math.exp(ce)

                    rec = {
                        "method": name,
                        "sample_id": sample_id,
                        "ce": ce,
                        "ppl": ppl,
                        "tokens": tokc,
                    }

                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    rows.append((name, sample_id, tokc, ce, ppl))

                buf_p = []
                buf_c = []
                buf_ids = []

        # -------- IMPORTANT FIX: process leftover final batch --------
        if buf_p:
            ces, toks = compute_ce_ppl(model, tok, buf_p, buf_c, max_ctx)

            for sample_id, ce, tokc in zip(buf_ids, ces, toks):
                ppl = math.exp(ce)

                rec = {
                    "method": name,
                    "sample_id": sample_id,
                    "ce": ce,
                    "ppl": ppl,
                    "tokens": tokc,
                }

                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows.append((name, sample_id, tokc, ce, ppl))

    print("Saved:", out_path)
    return rows

# ---------------- CSV WRITERS ----------------
def write_all_csv(rows):
    with open(OUT_ALL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "display_name",
            "sample_id",
            "tokens",
            "ce",
            "ppl",
        ])

        for method, sample_id, tokc, ce, ppl in rows:
            display = METHOD_DISPLAY.get(method, method)
            writer.writerow([
                method,
                display,
                sample_id,
                tokc,
                ce,
                ppl,
            ])

    print("Saved:", OUT_ALL_CSV)

def write_summary_csv(stats_ppl, stats_ce, stats_tokens):
    with open(OUT_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "method",
            "display_name",
            "samples",
            "mean_ppl",
            "std_ppl",
            "median_ppl",
            "min_ppl",
            "max_ppl",
            "p95_ppl",
            "mean_ce",
            "std_ce",
            "mean_tokens",
        ])

        for file_name in FILES:
            key = file_name.replace(".json", "")
            display = METHOD_DISPLAY.get(key, key)

            ppl_vals = stats_ppl[key]
            ce_vals = stats_ce[key]
            token_vals = stats_tokens[key]

            writer.writerow([
                key,
                display,
                len(ppl_vals),
                safe_mean(ppl_vals),
                safe_std(ppl_vals),
                safe_median(ppl_vals),
                safe_min(ppl_vals),
                safe_max(ppl_vals),
                safe_percentile(ppl_vals, 95),
                safe_mean(ce_vals),
                safe_std(ce_vals),
                safe_mean(token_vals),
            ])

    print("Saved:", OUT_SUMMARY_CSV)

# ---------------- MAIN ----------------
def main():
    tok = load_tok(MODEL_PATH)
    model = load_model(MODEL_PATH)

    all_rows = []

    for f in FILES:
        path = os.path.join(INPUT_DIR, f)

        if not os.path.exists(path):
            print("Missing:", f)
            continue

        all_rows += process_file(
            model=model,
            tok=tok,
            path=path,
            batch=BATCH_SIZE,
            max_ctx=model.config.max_position_embeddings,
        )

    stats_ppl = defaultdict(list)
    stats_ce = defaultdict(list)
    stats_tokens = defaultdict(list)

    for method, sample_id, tokc, ce, ppl in all_rows:
        stats_ppl[method].append(ppl)
        stats_ce[method].append(ce)
        stats_tokens[method].append(tokc)

    print("\n===== PPL SUMMARY =====\n")
    print(f"{'Method':<15}{'Samples':>10}{'MeanPPL':>12}{'StdPPL':>12}{'MedianPPL':>12}")
    print("-" * 65)

    for f in FILES:
        key = f.replace(".json", "")
        display = METHOD_DISPLAY.get(key, key)
        vals = stats_ppl[key]

        print(
            f"{display:<15}"
            f"{len(vals):>10}"
            f"{safe_mean(vals):>12.4f}"
            f"{safe_std(vals):>12.4f}"
            f"{safe_median(vals):>12.4f}"
        )

    write_all_csv(all_rows)
    write_summary_csv(stats_ppl, stats_ce, stats_tokens)

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()