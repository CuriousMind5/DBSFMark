# DBSF: Distribution-Based Score Fusion for Robust and Low-Distortion Text Watermarking

This repository contains the implementation of **DBSF**, a Distribution-Based Score Fusion watermarking method for large language model generated text. DBSF is the author’s method in this repository. The code also includes several baseline watermarking methods for comparison, including KGW, SWEET, EWD, DIP, Unbiased, SynthID, and MorphMark.

## Overview

DBSF is Distribution-Based Score Fusion for Robust and Low-Distortion Text Watermarking. During generation, DBSF selects eligible greenlist tokens and assigns fused bias weights using token rank, probability, and entropy information. During detection, it applies a matched fused weighted z-score, with entropy gating, to improve watermark detectability and robustness.

Main DBSF files:

```text
watermark/dbsf/dbsf.py        # DBSF injection and detection implementation
config/DBSF.json              # DBSF hyperparameters
generation_script.py          # Text generation script
detection_script.py           # Detection script
auroc_script.py               # AUROC and metric calculation script
ppl_script.py                 # Perplexity evaluation script
gammasweep_script.py          # Gamma sweep experiment script
```

generation_script.py and detection_script.py can be used to run the main DBSF experiments as well as all baseline experiments.
I have used MarkLLM toolkit for the baselines.
## Installation

```bash
git clone  https://github.com/CuriousMind5/DBSFMark.git
cd DBSF
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## DBSF Configuration

The default DBSF configuration is stored in `config/DBSF.json`:

```json
{
  "algorithm_name": "DBSF",
  "gamma": 0.25,
  "delta": 2.0,
  "hash_key": 15485863,
  "z_threshold": 4.0,
  "prefix_length": 1,
  "f_scheme": "time",
  "window_scheme": "left",
  "entropy_threshold": 0.9,
  "topk": 50,
  "rank_r0": 50.0,
  "rank_decay": "inv",
  "alpha_rank": 0.5,
  "alpha_prob": 0.3,
  "alpha_entropy": 0.2,
  "normalize_fused_weights": true
}
```

## Basic Usage

Generate watermarked text:

```bash
python generation_script.py \
  --model_path /path/to/local/model \
  --input_json /path/to/input_dataset.json \
  --num_samples 500 \
  --prompt_tokens 50 \
  --max_new_tokens 200 \
  --temperature 0.7
```

Run detection:

```bash
python detection_script.py
```

Compute AUROC and detection metrics:

```bash
python auroc_script.py
```

Evaluate perplexity:

```bash
python ppl_script.py
```

## Important Notes

Some experiment scripts currently contain local absolute paths such as `/home/sy/...`. Before running the full experiments on another machine, update those paths to match your local model, dataset, and output directories.

## Models Used
1- OPT-1.3B
2- OPT-2.7B
3- OPT-6.7B
4- OPT-13B
5- Qwen3-8B

All models were used locally in this experiment. To fully reproduce the experiments, please update the model paths according to your local environment.

# DBSFMark

