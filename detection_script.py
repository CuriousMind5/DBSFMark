# -*- coding: utf-8 -*-
from __future__ import annotations

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import math
import statistics as st
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.transformers_config import TransformersConfig

from watermark.kgw.kgw import KGW, KGWConfig
from watermark.dbsf.dbsf import DBSF, DBSFConfig
from watermark.sweet.sweet import SWEET, SWEETConfig
from watermark.ewd.ewd import EWD, EWDConfig
from watermark.dip.dip import DIP, DIPConfig
from watermark.unbiased.unbiased import UnbiasedWatermark
from watermark.synthid.synthid import SynthID
from watermark.morphmark.morphmark import MorphMark


# =========================================================
# SETTINGS OF LOCAL PATH
# =========================================================

BASE_DIR = Path("/home/sy/markove_markllm/markllm/MarkLLM-main/MarkLLM-main")
MODEL_PATH = Path("/home/sy/watermark_unlearning_experiment/models/facebook/opt-6.7B")
INPUT_DIR = BASE_DIR / "DBSF_experiment/outputs_0.25_0.5_c4_realnewslike_opt-6.7B/WordSBERT_50/"

INPUT_FILES = {
    "vanilla": "vanilla.json",
    "kgw": "kgw.json",
    "sweet": "sweet.json",
    "ewd": "ewd.json",
    "dip": "dip.json",
    "unbiased": "unbiased.json",
    "synthid": "synthid.json",
    "morphmark": "morphmark.json",
    "dbsf": "dbsf.json",
}

ALL_METHODS = [
    "kgw",
    "sweet",
    "ewd",
    "dip",
    "unbiased",
    "synthid",
    "morphmark",
    "dbsf",
]

OUT_JSONL = INPUT_DIR / "detect_all.jsonl"
OUT_CSV = INPUT_DIR / "detect_summary.csv"

Z_THRESHOLD = 4.0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# =========================================================
# IO HELPERS
# =========================================================

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def extract_text(rec: dict):
    """
    Supports common text fields used in generation files.
    """

    if "sampled" in rec:
        return rec["sampled"]

    if "text" in rec:
        return rec["text"]

    if "completion" in rec:
        return rec["completion"]

    if "output" in rec:
        return rec["output"]

    if "generated_text" in rec:
        return rec["generated_text"]

    raise KeyError(
        "No valid text field found. Expected one of: "
        "sampled, text, completion, output, generated_text"
    )


# =========================================================
# ACTIVE METHOD SELECTION
# =========================================================

def get_active_methods():
    """
    Select only methods whose JSON files exist in INPUT_DIR.
    """

    active_methods = []

    for method in ALL_METHODS:
        if method not in INPUT_FILES:
            print(f"[SKIP] {method}: not found in INPUT_FILES.")
            continue

        method_path = INPUT_DIR / INPUT_FILES[method]

        if not method_path.exists():
            print(f"[SKIP] {method}: missing file {method_path}")
            continue

        active_methods.append(method)

    return active_methods


# =========================================================
# MODEL
# =========================================================

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        torch_dtype=DTYPE,
    ).to(DEVICE)

    model.eval()
    model.config.use_cache = False

    return model, tokenizer


# =========================================================
# DETECTORS
# =========================================================

def build_detectors(tf_cfg, active_methods):
    """
    Build only active detectors.
    """

    detectors = {}

    if "kgw" in active_methods:
        detectors["kgw"] = KGW(
            KGWConfig(BASE_DIR / "config/KGW.json", tf_cfg),
            tf_cfg,
        )

    if "sweet" in active_methods:
        detectors["sweet"] = SWEET(
            SWEETConfig(BASE_DIR / "config/SWEET.json", tf_cfg),
            tf_cfg,
        )

    if "ewd" in active_methods:
        detectors["ewd"] = EWD(
            EWDConfig(BASE_DIR / "config/EWD.json", tf_cfg),
            tf_cfg,
        )

    if "dip" in active_methods:
        detectors["dip"] = DIP(
            DIPConfig(BASE_DIR / "config/DIP.json", tf_cfg),
            tf_cfg,
        )

    if "unbiased" in active_methods:
        detectors["unbiased"] = UnbiasedWatermark(
            str(BASE_DIR / "config/Unbiased.json"),
            tf_cfg,
        )

    if "synthid" in active_methods:
        detectors["synthid"] = SynthID(
            str(BASE_DIR / "config/SynthID.json"),
            tf_cfg,
        )

    if "morphmark" in active_methods:
        detectors["morphmark"] = MorphMark(
            str(BASE_DIR / "config/MorphMark.json"),
            tf_cfg,
        )

    if "dbsf" in active_methods:
        detectors["dbsf"] = DBSF(
            DBSFConfig(BASE_DIR / "config/DBSF.json", tf_cfg),
            tf_cfg,
        )

    return detectors


def build_tasks(active_methods):
    tasks = []

    vanilla_path = INPUT_DIR / INPUT_FILES["vanilla"]

    if not vanilla_path.exists():
        raise FileNotFoundError(f"Missing vanilla file: {vanilla_path}")

    vanilla_data = load_json(vanilla_path)

    for method in active_methods:
        for i, rec in enumerate(vanilla_data):
            tasks.append({
                "scheme": f"vanilla_{method}",
                "detector_scheme": method,
                "source_scheme": "vanilla",
                "idx": i,
                "text": extract_text(rec),
                "label": 0,
            })

    for method in active_methods:
        method_file = INPUT_DIR / INPUT_FILES[method]

        if not method_file.exists():
            print(f"[SKIP] positive tasks for {method}: missing {method_file}")
            continue

        data = load_json(method_file)

        for i, rec in enumerate(data):
            tasks.append({
                "scheme": method,
                "detector_scheme": method,
                "source_scheme": method,
                "idx": i,
                "text": extract_text(rec),
                "label": 1,
            })

    return tasks


# =========================================================
# DETECTION SCORE EXTRACTION
# =========================================================

def extract_detection_score(out: dict, detector_scheme: str):
    """
    Extract detector score from different method outputs.
    """

    candidate_keys = [
        "score",
        "z_score",
        "confidence",
        "z",
        "detection_score",
        "watermark_score",
    ]

    for k in candidate_keys:
        if k in out:
            v = safe_float(out.get(k), float("nan"))
            if math.isfinite(v):
                return v

    return float("nan")


def extract_flag(out: dict, detector_scheme: str, score: float):
    """
    Keep detector decision if available.
    Otherwise use Z_THRESHOLD.
    """

    if "is_watermarked" in out:
        return bool(out.get("is_watermarked", False))

    if "flagged" in out:
        return bool(out.get("flagged", False))

    return bool(math.isfinite(score) and score >= Z_THRESHOLD)


def detect_one(task, detectors):
    scheme = task["scheme"]
    detector_scheme = task["detector_scheme"]
    text = task["text"]

    result = {
        "scheme": scheme,
        "detector_scheme": detector_scheme,
        "source_scheme": task["source_scheme"],
        "idx": task["idx"],
        "label": task["label"],
        "z": float("nan"),
        "score": float("nan"),
        "flagged": False,
    }

    try:
        detector = detectors[detector_scheme]

        with torch.inference_mode():
            out = detector.detect_watermark(text, return_dict=True)

        if not isinstance(out, dict):
            raise TypeError(f"Detector output is not dict: {type(out)}")

        score = extract_detection_score(out, detector_scheme)
        flagged = extract_flag(out, detector_scheme, score)

        result["z"] = score
        result["score"] = score
        result["flagged"] = flagged

        for k, v in out.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                result[f"raw_{k}"] = v

    except Exception as e:
        result["error"] = str(e)

    return result


# =========================================================
# SUMMARY
# =========================================================

def summarize(results):
    stats = {}

    for r in results:
        scheme = r["scheme"]

        if scheme not in stats:
            stats[scheme] = {
                "n": 0,
                "flag": 0,
                "zs": [],
                "errors": 0,
                "label": r.get("label", -1),
                "detector_scheme": r.get("detector_scheme", ""),
                "source_scheme": r.get("source_scheme", ""),
            }

        z = safe_float(r.get("z", float("nan")), float("nan"))

        if math.isfinite(z):
            stats[scheme]["zs"].append(z)

        stats[scheme]["n"] += 1

        if r.get("flagged", False):
            stats[scheme]["flag"] += 1

        if "error" in r:
            stats[scheme]["errors"] += 1

    return stats


def write_summary_csv(stats):
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write(
            "scheme,"
            "detector_scheme,"
            "source_scheme,"
            "label,"
            "n,"
            "valid_scores,"
            "errors,"
            "mean_z,"
            "std_z,"
            "min_z,"
            "max_z,"
            "percent_flagged\n"
        )

        for scheme, v in stats.items():
            zs = v["zs"]

            mean_z = st.mean(zs) if zs else 0.0
            std_z = st.pstdev(zs) if len(zs) > 1 else 0.0
            min_z = min(zs) if zs else 0.0
            max_z = max(zs) if zs else 0.0
            pct_flagged = (100.0 * v["flag"] / v["n"]) if v["n"] else 0.0

            f.write(
                f"{scheme},"
                f"{v['detector_scheme']},"
                f"{v['source_scheme']},"
                f"{v['label']},"
                f"{v['n']},"
                f"{len(zs)},"
                f"{v['errors']},"
                f"{mean_z:.4f},"
                f"{std_z:.4f},"
                f"{min_z:.4f},"
                f"{max_z:.4f},"
                f"{pct_flagged:.2f}\n"
            )


# =========================================================
# MAIN
# =========================================================

def main():
    print("Input directory:", INPUT_DIR)

    print("\nChecking active methods...")
    active_methods = get_active_methods()

    if not active_methods:
        raise RuntimeError("No active methods found. Please check INPUT_DIR and method files.")

    print("Active methods:", active_methods)

    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    tf_cfg = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        vocab_size=model.get_output_embeddings().weight.shape[0],
    )

    print("Building detectors...")
    detectors = build_detectors(tf_cfg, active_methods)

    print("Built detectors:", list(detectors.keys()))

    print("Building detection tasks...")
    tasks = build_tasks(active_methods)

    print("Total tasks:", len(tasks))
    print("Output JSONL:", OUT_JSONL)
    print("Output CSV:", OUT_CSV)

    results = []

    for task in tqdm(tasks, desc="Detecting"):
        results.append(detect_one(task, detectors))

    save_jsonl(OUT_JSONL, results)

    stats = summarize(results)
    write_summary_csv(stats)

    print("\nSaved:", OUT_JSONL)
    print("Saved:", OUT_CSV)

    print("\nSummary:")
    for scheme, v in stats.items():
        zs = v["zs"]
        mean_z = st.mean(zs) if zs else 0.0
        pct_flagged = (100.0 * v["flag"] / v["n"]) if v["n"] else 0.0

        print(
            f"{scheme:<20} "
            f"detector={v['detector_scheme']:<10} "
            f"source={v['source_scheme']:<10} "
            f"label={v['label']} "
            f"n={v['n']:<5} "
            f"valid={len(zs):<5} "
            f"mean_z={mean_z:>9.4f} "
            f"flagged={pct_flagged:>7.2f}% "
            f"errors={v['errors']}"
        )


if __name__ == "__main__":
    main()
