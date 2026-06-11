# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import torch
from math import sqrt
from functools import partial

from ..base import BaseWatermark, BaseConfig
from utils.transformers_config import TransformersConfig
from transformers import LogitsProcessor, LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization


class DBSFConfig(BaseConfig):
    """Config class for DBSF algorithm."""

    def initialize_parameters(self) -> None:
        # KGW-style basics
        self.gamma = float(self.config_dict["gamma"])
        self.delta = float(self.config_dict["delta"])
        self.hash_key = int(self.config_dict["hash_key"])
        self.z_threshold = float(self.config_dict["z_threshold"])
        self.prefix_length = int(self.config_dict["prefix_length"])
        self.f_scheme = str(self.config_dict.get("f_scheme", "time"))
        self.window_scheme = str(self.config_dict.get("window_scheme", "left"))

        # DBSF-specific
        self.entropy_threshold = float(self.config_dict.get("entropy_threshold", 0.0))
        self.topk = int(self.config_dict.get("topk", 50))
        self.rank_r0 = float(self.config_dict.get("rank_r0", 50.0))
        self.rank_decay = str(self.config_dict.get("rank_decay", "inv"))  # inv / exp

        # fusion weights
        self.alpha_rank = float(self.config_dict.get("alpha_rank", 0.5))
        self.alpha_prob = float(self.config_dict.get("alpha_prob", 0.3))
        self.alpha_entropy = float(self.config_dict.get("alpha_entropy", 0.2))

        # keep total added bias stable
        self.normalize_fused_weights = bool(
            self.config_dict.get("normalize_fused_weights", True)
        )

    @property
    def algorithm_name(self) -> str:
        return "DBSF"


class DBSFUtils:
    """Utility class for DBSF algorithm."""

    def __init__(self, config: DBSFConfig, *args, **kwargs) -> None:
        self.config = config
        self.rng = torch.Generator(device=self.config.device)
        self.rng.manual_seed(self.config.hash_key)

        self.prf = torch.randperm(
            self.config.vocab_size,
            device=self.config.device,
            generator=self.rng,
        )

        self.f_scheme_map = {
            "time": self._f_time,
            "additive": self._f_additive,
            "skip": self._f_skip,
            "min": self._f_min,
        }
        self.window_scheme_map = {
            "left": self._get_greenlist_ids_left,
            "self": self._get_greenlist_ids_self,
        }

    # =========================================================
    # Prefix hashing
    # =========================================================

    def _f(self, input_ids: torch.LongTensor) -> int:
        return int(self.f_scheme_map[self.config.f_scheme](input_ids))

    def _f_time(self, input_ids: torch.LongTensor):
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        return self.prf[time_result % self.config.vocab_size]

    def _f_additive(self, input_ids: torch.LongTensor):
        additive_result = 0
        for i in range(0, self.config.prefix_length):
            additive_result += input_ids[-1 - i].item()
        return self.prf[additive_result % self.config.vocab_size]

    def _f_skip(self, input_ids: torch.LongTensor):
        return self.prf[input_ids[-self.config.prefix_length].item()]

    def _f_min(self, input_ids: torch.LongTensor):
        return min(
            self.prf[input_ids[-1 - i].item()]
            for i in range(0, self.config.prefix_length)
        )

    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        return self.window_scheme_map[self.config.window_scheme](input_ids)

    def _get_greenlist_ids_left(
        self, input_ids: torch.LongTensor
    ) -> torch.LongTensor:
        self.rng.manual_seed(
            (self.config.hash_key * self._f(input_ids)) % self.config.vocab_size
        )
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        vocab_permutation = torch.randperm(
            self.config.vocab_size,
            device=input_ids.device,
            generator=self.rng,
        )
        return vocab_permutation[:greenlist_size]

    def _get_greenlist_ids_self(
        self, input_ids: torch.LongTensor
    ) -> torch.LongTensor:
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        greenlist_ids = []

        f_x = self._f(input_ids)
        for k in range(0, self.config.vocab_size):
            h_k = f_x * int(self.prf[k])
            self.rng.manual_seed(h_k % self.config.vocab_size)
            vocab_permutation = torch.randperm(
                self.config.vocab_size,
                device=input_ids.device,
                generator=self.rng,
            )
            temp_greenlist_ids = vocab_permutation[:greenlist_size]
            if k in temp_greenlist_ids:
                greenlist_ids.append(k)

        return torch.tensor(
            greenlist_ids,
            device=input_ids.device,
            dtype=torch.long,
        )

    # =========================================================
    # DBSF score fusion helpers
    # =========================================================

    @torch.no_grad()
    def calculate_entropy_from_scores(self, scores: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(scores.float(), dim=-1)
        ent = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
        return ent

    @torch.no_grad()
    def calculate_entropy_sequence(
        self, model, tokenized_text: torch.Tensor
    ) -> list[float]:
        with torch.no_grad():
            output = model(torch.unsqueeze(tokenized_text, 0), return_dict=True)
            probs = torch.softmax(output.logits.float(), dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
            entropy = entropy[0].cpu().tolist()
            entropy.insert(0, -10000.0)
            return entropy[:-1]

    @torch.no_grad()
    def calculate_logits_sequence(
        self, model, tokenized_text: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns logits aligned with the next-token positions.
        output[idx - 1] predicts tokenized_text[idx].
        """
        with torch.no_grad():
            output = model(torch.unsqueeze(tokenized_text, 0), return_dict=True)
            logits = output.logits[0].float()  # [seq_len, vocab]
            return logits[:-1]  # aligned for positions idx = 1..len-1

    @torch.no_grad()
    def _rank_weights(self, ranks: torch.Tensor) -> torch.Tensor:
        r = ranks.float()
        if self.config.rank_decay == "exp":
            w = torch.exp(-r / self.config.rank_r0)
        else:
            w = 1.0 / (1.0 + (r / self.config.rank_r0))
        return torch.clamp(w, min=1e-8)

    @torch.no_grad()
    def _entropy_scalar(self, entropy_value: torch.Tensor | float) -> float:
        if isinstance(entropy_value, torch.Tensor):
            entropy_value = float(entropy_value.item())

        if entropy_value <= self.config.entropy_threshold:
            return 0.0

        # normalize by log(V)
        max_entropy = math.log(max(2, self.config.vocab_size))
        return min(1.0, entropy_value / max_entropy)

    @torch.no_grad()
    def fused_green_weights(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        batch_index: int,
    ) -> tuple[torch.LongTensor, torch.Tensor, bool]:
        """
        Returns:
            eligible_green_ids: token ids to bias
            fused_weights: per-token weights aligned with eligible_green_ids
            active: whether watermarking is active at this step
        """
        greenlist_ids = self.get_greenlist_ids(input_ids[batch_index])
        logits = scores[batch_index]

        # step-level entropy gate
        ent = self.calculate_entropy_from_scores(logits.unsqueeze(0))[0]
        ent_scalar = self._entropy_scalar(ent)
        if ent_scalar <= 0.0:
            return (
                greenlist_ids,
                torch.zeros_like(greenlist_ids, dtype=logits.dtype),
                False,
            )

        probs = torch.softmax(logits.float(), dim=-1)

        # ranks from logits
        order = torch.argsort(logits, descending=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(
            logits.shape[0],
            device=logits.device,
            dtype=order.dtype,
        )

        # keep only green tokens that are inside top-k
        topk = min(self.config.topk, logits.shape[0])
        topk_ids = order[:topk]

        topk_mask = torch.zeros(
            logits.shape[0],
            device=logits.device,
            dtype=torch.bool,
        )
        topk_mask[topk_ids] = True

        green_mask = torch.zeros(
            logits.shape[0],
            device=logits.device,
            dtype=torch.bool,
        )
        green_mask[greenlist_ids] = True

        eligible_mask = green_mask & topk_mask
        eligible_green_ids = torch.nonzero(eligible_mask, as_tuple=False).squeeze(-1)

       
        if eligible_green_ids.numel() == 0:
            eligible_green_ids = greenlist_ids

        rank_w = self._rank_weights(ranks[eligible_green_ids]).float()

        prob_w = probs[eligible_green_ids]
        prob_w = prob_w / (prob_w.sum() + 1e-12)

        ent_w = torch.full_like(prob_w, fill_value=ent_scalar)

        fused = (
            self.config.alpha_rank * rank_w
            + self.config.alpha_prob * prob_w
            + self.config.alpha_entropy * ent_w
        )

        fused = torch.clamp(fused, min=1e-8)

        if self.config.normalize_fused_weights:
            target_mass = float(eligible_green_ids.numel())
            fused = fused / (fused.sum() + 1e-12) * target_mass

        return eligible_green_ids, fused.to(logits.dtype), True

    # =========================================================
    # Detection
    # =========================================================

    def _compute_z_score_from_moments(
        self,
        observed_sum: float,
        expected_sum: float,
        variance_sum: float,
    ) -> float:
        variance_sum = max(float(variance_sum), 1e-12)
        return (float(observed_sum) - float(expected_sum)) / math.sqrt(variance_sum)

    @torch.no_grad()
    def score_sequence(
        self,
        input_ids: torch.Tensor,
        logits_sequence: torch.Tensor,
    ) -> tuple[
        float, list[float], list[int], float, float, float, int
    ]:
        """
        Detection matched to injection:

        For each valid step:
          - rebuild the same eligible green tokens and fused weights
          - if observed token is one of those eligible tokens,
            observed statistic is its fused weight, else 0

        Null expectation and variance are computed under the
        *unwatermarked* next-token distribution from logits.

        Returns:
            z_score
            token_level_observed_weights
            token_level_active_flags
            observed_sum
            expected_sum
            variance_sum
            num_tokens_scored
        """
        prefix_pad = self.config.prefix_length

        token_level_observed_weights = [-1.0 for _ in range(prefix_pad)]
        token_level_active_flags = [-1 for _ in range(prefix_pad)]

        observed_sum = 0.0
        expected_sum = 0.0
        variance_sum = 0.0
        num_tokens_scored = 0

        for idx in range(prefix_pad, len(input_ids)):
            # logits_sequence[idx - 1] predicts input_ids[idx]
            step_logits = logits_sequence[idx - 1]
            step_prefix = input_ids[:idx].unsqueeze(0)

            eligible_ids, fused_weights, active = self.fused_green_weights(
                input_ids=step_prefix,
                scores=step_logits.unsqueeze(0),
                batch_index=0,
            )

            curr_token = int(input_ids[idx].item())

            if not active or eligible_ids.numel() == 0:
                token_level_observed_weights.append(0.0)
                token_level_active_flags.append(0)
                continue

            probs = torch.softmax(step_logits.float(), dim=-1)

            stat_vec = torch.zeros_like(probs)
            stat_vec[eligible_ids] = fused_weights.float()

            # observed weighted hit
            observed_value = float(stat_vec[curr_token].item())

            # null moments under original LM distribution
            mu_t = float(torch.sum(probs * stat_vec).item())
            second_t = float(torch.sum(probs * (stat_vec ** 2)).item())
            var_t = max(0.0, second_t - (mu_t ** 2))

            observed_sum += observed_value
            expected_sum += mu_t
            variance_sum += var_t
            num_tokens_scored += 1

            token_level_observed_weights.append(observed_value)
            token_level_active_flags.append(1)

        if num_tokens_scored < 1:
            raise ValueError(
                "Must have at least 1 scored token after fused-logic gating."
            )

        z_score = self._compute_z_score_from_moments(
            observed_sum=observed_sum,
            expected_sum=expected_sum,
            variance_sum=variance_sum,
        )

        return (
            z_score,
            token_level_observed_weights,
            token_level_active_flags,
            observed_sum,
            expected_sum,
            variance_sum,
            num_tokens_scored,
        )


class DBSFLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for DBSF watermark."""

    def __init__(self, config: DBSFConfig, utils: DBSFUtils, *args, **kwargs) -> None:
        self.config = config
        self.utils = utils

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        scores = torch.nan_to_num(scores, neginf=-1e9, posinf=1e9)

        for b_idx in range(input_ids.shape[0]):
            eligible_ids, fused_weights, active = self.utils.fused_green_weights(
                input_ids=input_ids,
                scores=scores,
                batch_index=b_idx,
            )
            if not active or eligible_ids.numel() == 0:
                continue

            scores[b_idx, eligible_ids] = (
                scores[b_idx, eligible_ids]
                + self.config.delta * fused_weights
            )

        return scores


class DBSF(BaseWatermark):
    """Top-level class for DBSF algorithm."""

    def __init__(
        self,
        algorithm_config: str | DBSFConfig,
        transformers_config: TransformersConfig | None = None,
        *args,
        **kwargs,
    ) -> None:
        if isinstance(algorithm_config, str):
            self.config = DBSFConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, DBSFConfig):
            self.config = algorithm_config
        else:
            raise TypeError(
                "algorithm_config must be either a path string or a DBSFConfig instance"
            )

        self.utils = DBSFUtils(self.config)
        self.logits_processor = DBSFLogitsProcessor(self.config, self.utils)

    # =========================================================
    # GENERATION
    # =========================================================

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs,
        )

        encoded_prompt = self.config.generation_tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.config.device)

        encoded_output = generate_with_watermark(**encoded_prompt)
        watermarked_text = self.config.generation_tokenizer.batch_decode(
            encoded_output,
            skip_special_tokens=True,
        )[0]
        return watermarked_text

    # =========================================================
    # DETECTION
    # =========================================================

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        encoded_text = self.config.generation_tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0].to(self.config.device)

        logits_sequence = self.utils.calculate_logits_sequence(
            self.config.generation_model,
            encoded_text,
        )

        (
            z_score,
            token_observed_weights,
            active_flags,
            observed_sum,
            expected_sum,
            variance_sum,
            num_scored,
        ) = self.utils.score_sequence(
            encoded_text,
            logits_sequence,
        )

        is_watermarked = z_score > self.config.z_threshold

        if return_dict:
            return {
                "is_watermarked": bool(is_watermarked),
                "score": float(z_score),
                "observed_sum": float(observed_sum),
                "expected_sum": float(expected_sum),
                "variance_sum": float(variance_sum),
                "tokens_scored": int(num_scored),
                "weights": token_observed_weights,
                "active_flags": active_flags,
            }
        else:
            return bool(is_watermarked), float(z_score)

    # =========================================================
    # VISUALIZATION
    # =========================================================

    def get_data_for_visualization(self, text: str, *args, **kwargs):
        encoded_text = self.config.generation_tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0].to(self.config.device)

        logits_sequence = self.utils.calculate_logits_sequence(
            self.config.generation_model,
            encoded_text,
        )

        (
            _,
            highlight_values,
            active_flags,
            _,
            _,
            _,
            _,
        ) = self.utils.score_sequence(
            encoded_text,
            logits_sequence,
        )

        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)

        return DataForVisualization(decoded_tokens, highlight_values, active_flags)
