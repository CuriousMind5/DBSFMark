# -*- coding: utf-8 -*-

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Path settings
# ============================================================

INPUT_DIR = (
    "/home/sy/markove_markllm/markllm/MarkLLM-main/MarkLLM-main/"
    "DBSF_experiment/outputs_0.25_0.5_c4_realnewslike_opt-6.7B"
)

DETECT_FILE = os.path.join(INPUT_DIR, "detect_all.jsonl")

SAVE_DIR = os.path.join(INPUT_DIR, "z_score_radar_combined_only")
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_NAME = "z_score_radar_all_methods_reviewer_style"


# ============================================================
# 2. Method settings
# ============================================================

METHODS = [
    "kgw",
    "sweet",
    "morphmark",
    "ewd",
    "dip",
    "unbiased",
    "synthid",
    "dbsf",
]

METHOD_LABELS = {
    "kgw": "KGW",
    "sweet": "SWEET",
    "morphmark": "MorphMark",
    "ewd": "EWD",
    "dip": "DIP",
    "unbiased": "Unbiased",
    "synthid": "SynthID",
    "dbsf": "DBSF",
}

METHOD_COLORS = {
    "kgw": "#1f77b4",
    "sweet": "#ff7f0e",
    "morphmark": "#2ca02c",
    "ewd": "#d62728",
    "dip": "#9467bd",
    "unbiased": "#8c564b",
    "synthid": "#e377c2",
    "dbsf": "#c2185b",
}

VANILLA_COLOR = "#4d4d4d"


# ============================================================
# 3. Global font settings
# ============================================================

plt.rcParams["font.family"] = "DejaVu Sans"

# cleaner reviewer-friendly sizes
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.titleweight"] = "normal"

plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 10

plt.rcParams["savefig.dpi"] = 600


# ============================================================
# 4. Helper functions
# ============================================================

def safe_float(x):
    try:
        value = float(x)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return np.nan


def get_score(row):
    for key in ["z", "score", "raw_score", "z_score", "detect_score"]:
        if key in row:
            return safe_float(row[key])
    return np.nan


def normalize_text(x):
    if x is None:
        return ""
    return str(x).strip().lower()


def read_jsonl(path):
    rows = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"detect_all.jsonl not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARNING] Could not parse line {line_no}: {e}")

    return rows


def get_method_scores(df, method):
    sub = df[df["detector_scheme"] == method].copy()

    vanilla_scores = sub[sub["label"] == 0]["z_score_clean"].values
    watermarked_scores = sub[sub["label"] == 1]["z_score_clean"].values

    vanilla_scores = vanilla_scores[np.isfinite(vanilla_scores)]
    watermarked_scores = watermarked_scores[np.isfinite(watermarked_scores)]

    return vanilla_scores, watermarked_scores


def compute_stats(scores):
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]

    if len(scores) == 0:
        return None

    return {
        "Mean": np.mean(scores),
        "Median": np.median(scores),
        "Std": np.std(scores),
        "P75": np.percentile(scores, 75),
        "P90": np.percentile(scores, 90),
        "Max": np.max(scores),
    }


def normalize_pair_stats(vanilla_stats, watermarked_stats, stat_names):
    raw_values = []

    for stat in stat_names:
        raw_values.append(vanilla_stats[stat])
        raw_values.append(watermarked_stats[stat])

    raw_values = np.array(raw_values, dtype=float)
    raw_values = raw_values[np.isfinite(raw_values)]

    v_min = np.min(raw_values)
    v_max = np.max(raw_values)

    vanilla_norm = []
    watermarked_norm = []

    for stat in stat_names:
        v_val = vanilla_stats[stat]
        w_val = watermarked_stats[stat]

        if abs(v_max - v_min) < 1e-12:
            vanilla_norm.append(1.0)
            watermarked_norm.append(1.0)
        else:
            vanilla_norm.append((v_val - v_min) / (v_max - v_min))
            watermarked_norm.append((w_val - v_min) / (v_max - v_min))

    return vanilla_norm, watermarked_norm


def refine_radar_text_alignment(ax, angles, labels):

    ax.set_xticks(angles[:-1])

    # reviewer-friendly: slightly bigger but NOT bold
    ax.set_xticklabels(
        labels,
        fontsize=11,
        fontweight="normal"
    )

    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):

        angle_deg = np.degrees(angle)

        if 0 <= angle_deg < 45 or angle_deg >= 315:
            label.set_horizontalalignment("left")
        elif 135 < angle_deg < 225:
            label.set_horizontalalignment("right")
        else:
            label.set_horizontalalignment("center")

        if 45 <= angle_deg <= 135:
            label.set_verticalalignment("bottom")
        elif 225 <= angle_deg <= 315:
            label.set_verticalalignment("top")
        else:
            label.set_verticalalignment("center")


# ============================================================
# 5. Load detect_all.jsonl
# ============================================================

records = read_jsonl(DETECT_FILE)

if len(records) == 0:
    raise RuntimeError("detect_all.jsonl is empty.")

df = pd.DataFrame(records)

print("[INFO] Loaded rows:", len(df))
print("[INFO] Columns:", df.columns.tolist())


# ============================================================
# 6. Clean fields
# ============================================================

if "detector_scheme" not in df.columns:
    raise RuntimeError("Column 'detector_scheme' not found.")

if "label" not in df.columns:
    raise RuntimeError("Column 'label' not found.")

df["detector_scheme"] = df["detector_scheme"].apply(normalize_text)
df["label"] = df["label"].astype(int)
df["z_score_clean"] = df.apply(get_score, axis=1)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["z_score_clean", "label", "detector_scheme"])

print("[INFO] Valid score rows:", len(df))


# ============================================================
# 7. Prepare valid methods
# ============================================================

valid_methods = []

STAT_NAMES = [
    "Mean",
    "Median",
    "Std",
    "P75",
    "P90",
    "Max",
]

for method in METHODS:

    vanilla_scores, watermarked_scores = get_method_scores(df, method)

    if len(vanilla_scores) < 2 or len(watermarked_scores) < 2:
        continue

    valid_methods.append(method)


if len(valid_methods) == 0:
    raise RuntimeError("No valid methods found.")


# ============================================================
# 8. Radar plotting
# ============================================================

n_cols = 4
n_rows = int(np.ceil(len(valid_methods) / n_cols))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    subplot_kw=dict(polar=True),

    # slightly larger panels for readability
    figsize=(5.0 * n_cols, 4.8 * n_rows),

    squeeze=False
)

axes_flat = axes.flatten()

num_vars = len(STAT_NAMES)

angles = np.linspace(
    0,
    2 * np.pi,
    num_vars,
    endpoint=False
).tolist()

angles += angles[:1]


for ax_idx, method in enumerate(valid_methods):

    ax = axes_flat[ax_idx]

    method_label = METHOD_LABELS.get(method, method.upper())
    method_color = METHOD_COLORS.get(method, "#c2185b")

    vanilla_scores, watermarked_scores = get_method_scores(df, method)

    vanilla_stats = compute_stats(vanilla_scores)
    watermarked_stats = compute_stats(watermarked_scores)

    vanilla_values, watermarked_values = normalize_pair_stats(
        vanilla_stats,
        watermarked_stats,
        STAT_NAMES
    )

    vanilla_values += vanilla_values[:1]
    watermarked_values += watermarked_values[:1]

    # Vanilla
    ax.plot(
        angles,
        vanilla_values,
        color=VANILLA_COLOR,
        linewidth=2.0,
        marker="o",
        markersize=4.8,
        alpha=0.95
    )

    ax.fill(
        angles,
        vanilla_values,
        color=VANILLA_COLOR,
        alpha=0.10
    )

    # Watermarked
    ax.plot(
        angles,
        watermarked_values,
        color=method_color,
        linewidth=2.4,
        marker="o",
        markersize=5.0,
        alpha=0.98
    )

    ax.fill(
        angles,
        watermarked_values,
        color=method_color,
        alpha=0.20
    )

    refine_radar_text_alignment(
        ax,
        angles,
        STAT_NAMES
    )

    ax.tick_params(axis="x", pad=10)

    ax.set_ylim(0, 1.0)

    ax.set_yticks([0.25, 0.50, 0.75, 1.00])

    # NOT bold now
    ax.set_yticklabels(
        ["0.25", "0.50", "0.75", "1.00"],
        fontsize=10,
        fontweight="normal"
    )

    ax.tick_params(axis="y", pad=3)

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.9,
        alpha=0.35
    )

    ax.spines["polar"].set_linewidth(1.2)
    ax.spines["polar"].set_alpha(0.85)

    # reviewer-friendly title
    ax.set_title(
        method_label,
        fontsize=15,
        fontweight="normal",
        pad=16
    )


# Remove unused axes
for j in range(len(valid_methods), len(axes_flat)):
    axes_flat[j].axis("off")


plt.subplots_adjust(
    left=0.04,
    right=0.985,
    top=0.94,
    bottom=0.06,
    wspace=0.38,
    hspace=0.38
)

out_pdf = os.path.join(
    SAVE_DIR,
    f"{SAVE_NAME}.pdf"
)

out_png = os.path.join(
    SAVE_DIR,
    f"{SAVE_NAME}.png"
)

plt.savefig(
    out_pdf,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    out_png,
    dpi=600,
    bbox_inches="tight"
)

print(f"[INFO] Saved PDF: {out_pdf}")
print(f"[INFO] Saved PNG: {out_png}")

plt.show()