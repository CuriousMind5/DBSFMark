
# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================
# transformers_config.py
# Description: Configuration for transformers
# ============================================

class TransformersConfig:
    """Configuration class for transformers."""

    def __init__(self, model, tokenizer, vocab_size=None, device='cuda', *args, **kwargs):
        """
        Initialize the transformers configuration.

        Parameters:
            model (object): The model object.
            tokenizer (object): The tokenizer object.
            vocab_size (int): The vocabulary size.
            device (str): The device to use.
            kwargs: Additional keyword arguments (passed to model.generate via gen_kwargs).
        """
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

        # IMPORTANT:
        # For some models (e.g., OPT), tokenizer length can differ from LM head vocab size.
        # Watermark algorithms (Unigram/SWEET/EWD) build masks against logits size, so we must
        # trust the model output head size first to avoid shape mismatch errors.
        if vocab_size is not None:
            self.vocab_size = int(vocab_size)
        else:
            self.vocab_size = self._infer_vocab_size(model, tokenizer)

        self.gen_kwargs = {}
        self.gen_kwargs.update(kwargs)

    @staticmethod
    def _infer_vocab_size(model, tokenizer):
        # 1) Prefer lm_head.out_features (OPT/GPT-like)
        try:
            lm_head = getattr(model, "lm_head", None)
            if lm_head is not None and getattr(lm_head, "out_features", None) is not None:
                return int(lm_head.out_features)
        except Exception:
            pass

        # 2) Prefer output embeddings weight shape (general HF)
        try:
            get_out = getattr(model, "get_output_embeddings", None)
            if callable(get_out):
                out_emb = get_out()
                if out_emb is not None and hasattr(out_emb, "weight") and out_emb.weight is not None:
                    return int(out_emb.weight.shape[0])
        except Exception:
            pass

        # 3) Fall back to config.vocab_size if present
        try:
            cfg = getattr(model, "config", None)
            if cfg is not None and getattr(cfg, "vocab_size", None) is not None:
                return int(cfg.vocab_size)
        except Exception:
            pass

        # 4) Last resort: tokenizer length
        return int(len(tokenizer))

