# -*- coding: utf-8 -*-
from __future__ import annotations
from collections import OrderedDict

# ======================== TOGGLES ========================
ENABLE_ORIGINAL      = False

ENABLE_PEGASUS       = False
ENABLE_DIPPER_1      = False
ENABLE_DIPPER_2      = False

# Word deletion attacks
ENABLE_WORDDEL_10    = False
ENABLE_WORDDEL_30    = False
ENABLE_WORDDEL_50    = False

# Word synonym attacks
ENABLE_WORDSYN_10    = False
ENABLE_WORDSYN_30    = False
ENABLE_WORDSYN_50    = False

# WordSBERT / semantic word substitution attacks
ENABLE_WORDSBERT_10  = False
ENABLE_WORDSBERT_30  = False
ENABLE_WORDSBERT_50  = True

# Translation attack
ENABLE_TRANS_EN_ZH_EN = False


# ======================== FILE PATHS ========================
BASE_DIR = (
    "/home/sy/markove_markllm/markllm/MarkLLM-main/MarkLLM-main/"
    "DBSF_experiment/outputs_0.25_0.5_c4_realnewslike_opt-6.7B/"
)

FILE_ORIGINAL = BASE_DIR + "detect_all.jsonl"

FILE_PEGASUS = BASE_DIR + "Pegasus/detect_all.jsonl"
FILE_DIPPER_1 = BASE_DIR + "Dipper1/detect_all.jsonl"
FILE_DIPPER_2 = BASE_DIR + "Dipper2/detect_all.jsonl"

FILE_WORDDEL_10 = BASE_DIR + "WordDeletion_10/detect_all.jsonl"
FILE_WORDDEL_30 = BASE_DIR + "WordDeletion_30/detect_all.jsonl"
FILE_WORDDEL_50 = BASE_DIR + "WordDeletion_50/detect_all.jsonl"

FILE_WORDSYN_10 = BASE_DIR + "WordSynonym_10/detect_all.jsonl"
FILE_WORDSYN_30 = BASE_DIR + "WordSynonym_30/detect_all.jsonl"
FILE_WORDSYN_50 = BASE_DIR + "WordSynonym_50/detect_all.jsonl"

FILE_WORDSBERT_10 = BASE_DIR + "WordSBERT_10/detect_all.jsonl"
FILE_WORDSBERT_30 = BASE_DIR + "WordSBERT_30/detect_all.jsonl"
FILE_WORDSBERT_50 = BASE_DIR + "WordSBERT_50/detect_all.jsonl"

FILE_TRANS_EN_ZH_EN = BASE_DIR + "English-German-English/detect_all.jsonl"


OUT_DIR = (
    "/home/sy/markove_markllm/markllm/MarkLLM-main/MarkLLM-main/"
    "DBSF_experiment/outputs_0.25_0.5_c4_realnewslike_Qwen3-8B/auroc_curves/"
)

OUT_CSV = OUT_DIR + "clean_metrics_noattack_attacks.csv"
OUT_FIG_DIR = OUT_DIR + "auroc_curves/"


# ======================== SCHEMES ========================
SCHEMES = OrderedDict([
    ("KGW", "kgw"),
    ("SWEET", "sweet"),
    ("MorphMark", "morphmark"),
    ("EWD", "ewd"),
    ("DIP", "dip"),
    ("Unbiased", "unbiased"),
    ("SynthID", "synthid"),
    ("DBSF", "dbsf"),
])


NEG_SCHEMES = {
    "kgw": "vanilla_kgw",
    "sweet": "vanilla_sweet",
    "morphmark": "vanilla_morphmark",
    "ewd": "vanilla_ewd",
    "dip": "vanilla_dip",
    "unbiased": "vanilla_unbiased",
    "synthid": "vanilla_synthid",
    "dbsf": "vanilla_dbsf",
}


TARGET_FPR = 0.05


# ======================== IMPORTS ========================
import os
import re
import json
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# ======================== PLOT STYLE ========================
TITLE_FS = 22
LABEL_FS = 19
TICK_FS = 15
LEGEND_FS = 11

LINE_W = 2.5
RANDOM_LINE_W = 2.1

FIG_W = 6.8
FIG_H = 6.8

SAVE_DPI = 450

plt.rcParams.update({
    "axes.titlesize": TITLE_FS,
    "axes.labelsize": LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "legend.fontsize": LEGEND_FS,
    "font.size": 15,
    "axes.linewidth": 1.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ======================== HELPERS ========================
def _fail(msg: str):
    print("[error]", msg)
    raise SystemExit(1)


def _load_scores_and_flags(path: str, scheme: str):
    zs = []
    flags = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            try:
                obj = json.loads(ln)
            except Exception:
                continue

            if obj.get("scheme") != scheme:
                continue

            val = obj.get(
                "z",
                obj.get(
                    "score",
                    obj.get(
                        "z_score",
                        obj.get("confidence", "nan")
                    )
                )
            )

            try:
                z = float(val)
            except Exception:
                z = float("nan")

            if math.isfinite(z):
                zs.append(z)
                flags.append(bool(obj.get("flagged", False)))

    return zs, flags


def _normalize(scores):
    if not scores:
        return scores

    mn = min(scores)
    mx = max(scores)

    if mx - mn < 1e-8:
        return [0.5] * len(scores)

    return [(x - mn) / (mx - mn) for x in scores]


def _quantile(sorted_vals, q: float):
    n = len(sorted_vals)

    if n == 0:
        return float("nan")

    pos = q * (n - 1)
    lo = int(pos)
    hi = int(math.ceil(pos))

    if lo == hi:
        return sorted_vals[lo]

    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _tau_for_fpr(neg_scores, fpr: float):
    return _quantile(sorted(neg_scores), 1.0 - fpr)


def _safe(a: float, b: float):
    return (a / b) if b else 0.0


def _auc(pos, neg):
    if not pos or not neg:
        return float("nan")

    all_scores = pos + neg
    norm = _normalize(all_scores)

    pos_norm = norm[:len(pos)]
    neg_norm = norm[len(pos):]

    try:
        y_true = [1] * len(pos_norm) + [0] * len(neg_norm)
        y_score = pos_norm + neg_norm
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float("nan")


def _roc_points(pos, neg):
    if not pos or not neg:
        return None, None, float("nan")

    all_scores = pos + neg
    norm = _normalize(all_scores)

    pos_norm = norm[:len(pos)]
    neg_norm = norm[len(pos):]

    y_true = [1] * len(pos_norm) + [0] * len(neg_norm)
    y_score = pos_norm + neg_norm

    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        aucv = roc_auc_score(y_true, y_score)
        return fpr, tpr, aucv
    except Exception:
        return None, None, float("nan")


def _fmt(x):
    return "nan" if not math.isfinite(x) else f"{x:.3f}"


def _fmt_pct(x):
    return "nan" if not math.isfinite(x) else f"{100.0 * x:.2f}"


def _safe_filename(name: str):
    name = name.strip()
    name = name.replace(" ", "_")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = re.sub(r"[^A-Za-z0-9_.\-]+", "_", name)
    return name


def _linestyle_for_method(method_name: str):
    return "-"


def _linewidth_for_method(method_name: str):
    if method_name == "DBSF":
        return LINE_W + 0.4
    return LINE_W


def _style_roc_axis(ax, scenario_name: str):
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.04, 1.04)

    ax.set_title(scenario_name, fontsize=TITLE_FS, pad=16)

    ax.set_xlabel("False Positive Rate (FPR)", fontsize=LABEL_FS, labelpad=10)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    ax.set_ylabel("True Positive Rate (TPR)", fontsize=LABEL_FS, labelpad=10)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FS,
        width=1.2,
        length=5
    )

    ax.grid(True, linestyle="--", linewidth=0.75, alpha=0.55)

    for spine in ax.spines.values():
        spine.set_linewidth(1.3)


# ======================== METRICS ========================
def compute_for_file(path: str):
    results = OrderedDict()

    for name, scheme in SCHEMES.items():
        pos, flags = _load_scores_and_flags(path, scheme)

        neg_scheme = NEG_SCHEMES.get(scheme)
        if neg_scheme is None:
            _fail(f"No negative mapping found for scheme={scheme}")

        neg, _ = _load_scores_and_flags(path, neg_scheme)

        if not neg:
            _fail(
                f"No negative scores found for scheme={scheme}. "
                f"Expected negative scheme='{neg_scheme}' in {path}"
            )

        if not pos:
            print(f"[warning] No positive scores for scheme={scheme} in {path}")

        tau = _tau_for_fpr(neg, TARGET_FPR)

        TP = sum(z >= tau for z in pos)
        FN = len(pos) - TP
        FP = sum(z >= tau for z in neg)
        TN = len(neg) - FP

        tpr = _safe(TP, TP + FN)
        prec = _safe(TP, TP + FP)
        f1 = _safe(2 * prec * tpr, prec + tpr)

        aucv = _auc(pos, neg)

        results[name] = {
            "tpr": tpr,
            "f1": f1,
            "auroc": aucv,
            "pos_scores": pos,
            "neg_scores": neg,
        }

    return results


# ======================== ROC PLOTTING ========================
def plot_roc_curve_for_scenario(scenario_name: str, scenario_results):
    os.makedirs(OUT_FIG_DIR, exist_ok=True)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = plt.gca()

    has_curve = False

    for method_name in SCHEMES:
        vals = scenario_results.get(method_name)

        if vals is None:
            continue

        pos_scores = vals.get("pos_scores", [])
        neg_scores = vals.get("neg_scores", [])

        fpr, tpr, aucv = _roc_points(pos_scores, neg_scores)

        if fpr is None or tpr is None:
            print(f"[warning] ROC skipped for {scenario_name} / {method_name}")
            continue

        has_curve = True

        ax.step(
            fpr,
            tpr,
            where="post",
            linewidth=_linewidth_for_method(method_name),
            linestyle=_linestyle_for_method(method_name),
            label=f"{method_name} (AUC: {_fmt(aucv)})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=RANDOM_LINE_W,
        color="black",
        label="Random"
    )

    _style_roc_axis(ax, scenario_name)

    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.985, 0.035),
        fontsize=LEGEND_FS,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=0.92,
        borderpad=0.65,
        labelspacing=0.45,
        handlelength=2.20,
        handletextpad=0.65,
        borderaxespad=0.35
    )

    plt.tight_layout()

    safe_name = _safe_filename(scenario_name)

    png_path = os.path.join(OUT_FIG_DIR, f"roc_curve_{safe_name}.png")
    pdf_path = os.path.join(OUT_FIG_DIR, f"roc_curve_{safe_name}.pdf")

    if has_curve:
        plt.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.savefig(pdf_path, bbox_inches="tight")
        print("ROC PNG saved:", png_path)
        print("ROC PDF saved:", pdf_path)
    else:
        print(f"[warning] No valid ROC curves for scenario: {scenario_name}")

    plt.close(fig)


def plot_all_roc_curves(no_attack_results, attack_results):
    if no_attack_results is not None:
        plot_roc_curve_for_scenario("No Attack", no_attack_results)

    for attack_name, results in attack_results.items():
        plot_roc_curve_for_scenario(attack_name, results)


# ======================== TABLE PRINTING ========================
def _table_line(widths):
    return "+" + "+".join("-" * (w + 2) for w in widths) + "+"


def _table_row(values, widths, aligns=None):
    if aligns is None:
        aligns = ["left"] * len(values)

    cells = []

    for value, width, align in zip(values, widths, aligns):
        value = str(value)

        if align == "right":
            cells.append(" " + value.rjust(width) + " ")
        elif align == "center":
            cells.append(" " + value.center(width) + " ")
        else:
            cells.append(" " + value.ljust(width) + " ")

    return "|" + "|".join(cells) + "|"


def _print_table(title, headers, rows, aligns):
    print(f"\n==== {title} ====\n")

    widths = []
    for col_idx, header in enumerate(headers):
        width = len(str(header))
        for row in rows:
            width = max(width, len(str(row[col_idx])))
        widths.append(width)

    print(_table_line(widths))
    print(_table_row(headers, widths, ["center"] * len(headers)))
    print(_table_line(widths))

    for row in rows:
        print(_table_row(row, widths, aligns))

    print(_table_line(widths))


def print_no_attack_results(no_attack_results):
    headers = ["Method", "TPR@5%FPR", "F1", "AUROC"]
    rows = []

    for method_name in SCHEMES:
        vals = no_attack_results[method_name]
        rows.append([
            method_name,
            _fmt_pct(vals["tpr"]),
            _fmt(vals["f1"]),
            _fmt(vals["auroc"]),
        ])

    aligns = ["left", "right", "right", "right"]
    _print_table("No Attack Results", headers, rows, aligns)


def print_attack_auroc(attack_results):
    if not attack_results:
        print("\n[warning] No attack results available.")
        return

    headers = ["Method"] + list(attack_results.keys())
    rows = []

    for method_name in SCHEMES:
        row = [method_name]

        for attack_name in attack_results:
            vals = attack_results[attack_name][method_name]
            row.append(_fmt(vals["auroc"]))

        rows.append(row)

    aligns = ["left"] + ["right"] * (len(headers) - 1)
    _print_table("Attack Results: AUROC", headers, rows, aligns)


# ======================== CSV ========================
def write_csv(no_attack_results, attack_results):
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("scenario,method,tpr_at_5fpr,f1_at_5fpr,auroc\n")

        if no_attack_results is not None:
            for method_name in SCHEMES:
                v = no_attack_results[method_name]

                f.write(
                    f"No Attack,"
                    f"{method_name},"
                    f"{v['tpr']},"
                    f"{v['f1']},"
                    f"{v['auroc']}\n"
                )

        for attack_name in attack_results:
            for method_name in SCHEMES:
                v = attack_results[attack_name][method_name]

                f.write(
                    f"{attack_name},"
                    f"{method_name},"
                    f","
                    f","
                    f"{v['auroc']}\n"
                )

    print("\nCSV saved:", OUT_CSV)


# ======================== MAIN ========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_FIG_DIR, exist_ok=True)

    no_attack_results = None
    attack_results = OrderedDict()

    if ENABLE_ORIGINAL:
        if not os.path.exists(FILE_ORIGINAL):
            _fail(f"No Attack file not found: {FILE_ORIGINAL}")

        print("Loading No Attack:", FILE_ORIGINAL)
        no_attack_results = compute_for_file(FILE_ORIGINAL)

    attack_datasets = OrderedDict()

    if ENABLE_TRANS_EN_ZH_EN:
        attack_datasets["Translation"] = FILE_TRANS_EN_ZH_EN

    if ENABLE_DIPPER_1:
        attack_datasets["Dipper-1"] = FILE_DIPPER_1

    if ENABLE_DIPPER_2:
        attack_datasets["Dipper-2"] = FILE_DIPPER_2

    if ENABLE_PEGASUS:
        attack_datasets["Pegasus"] = FILE_PEGASUS

    if ENABLE_WORDDEL_10:
        attack_datasets["WordDel-10"] = FILE_WORDDEL_10

    if ENABLE_WORDDEL_30:
        attack_datasets["WordDel-30"] = FILE_WORDDEL_30

    if ENABLE_WORDDEL_50:
        attack_datasets["WordDel-50"] = FILE_WORDDEL_50

    if ENABLE_WORDSYN_10:
        attack_datasets["WordSyn-10"] = FILE_WORDSYN_10

    if ENABLE_WORDSYN_30:
        attack_datasets["WordSyn-30"] = FILE_WORDSYN_30

    if ENABLE_WORDSYN_50:
        attack_datasets["WordSyn-50"] = FILE_WORDSYN_50

    if ENABLE_WORDSBERT_10:
        attack_datasets["WordSBERT-10"] = FILE_WORDSBERT_10

    if ENABLE_WORDSBERT_30:
        attack_datasets["WordSBERT-30"] = FILE_WORDSBERT_30

    if ENABLE_WORDSBERT_50:
        attack_datasets["WordSBERT-50"] = FILE_WORDSBERT_50

    for attack_name, attack_path in attack_datasets.items():
        if not os.path.exists(attack_path):
            print(
                f"[warning] Missing attack file, skipping {attack_name}: "
                f"{attack_path}"
            )
            continue

        print("Loading attack:", attack_name, attack_path)
        attack_results[attack_name] = compute_for_file(attack_path)

    if no_attack_results is not None:
        print_no_attack_results(no_attack_results)

    if attack_results:
        print_attack_auroc(attack_results)

    write_csv(no_attack_results, attack_results)
    plot_all_roc_curves(no_attack_results, attack_results)

    print("\nAll outputs saved.")
    print("CSV folder:", OUT_DIR)
    print("ROC figure folder:", OUT_FIG_DIR)


if __name__ == "__main__":
    main()

