# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Abliteration evaluator for refusal counting and KL divergence measurement.

Scores each optimization trial on two objectives: the number of refusals
(lower is better for abliteration) and the KL divergence from the original
model's output distribution (lower means less capability damage). Supports
per-category refusal breakdown when categories are active.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from .config import Settings
from .model import Model
from .utils import Prompt, load_prompts, print


class Evaluator:
    settings: Settings
    model: Model
    good_prompts: list[Prompt]
    bad_prompts: list[Prompt]
    base_logprobs: Tensor
    base_refusals: int
    # Per-category tracking
    category_prompts: dict[str, list[Prompt]]
    category_base_refusals: dict[str, int]

    def __init__(
        self,
        settings: Settings,
        model: Model,
        category_eval_prompts: dict[str, list[Prompt]] | None = None,
    ):
        self.settings = settings
        self.model = model
        self.category_prompts = {}
        self.category_base_refusals = {}

        print()
        print(
            f"Loading good evaluation prompts from [bold]{settings.good_evaluation_prompts.dataset}[/]..."
        )
        self.good_prompts = load_prompts(settings, settings.good_evaluation_prompts)
        print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded")

        print("* Obtaining first-token probability distributions...")
        self.base_logprobs = model.get_logprobs_batched(self.good_prompts)

        print()
        print(
            f"Loading bad evaluation prompts from [bold]{settings.bad_evaluation_prompts.dataset}[/]..."
        )
        self.bad_prompts = load_prompts(settings, settings.bad_evaluation_prompts)
        print(f"* [bold]{len(self.bad_prompts)}[/] prompts loaded")

        print("* Counting model refusals...")
        self.base_refusals = self.count_refusals()
        print(
            f"* Initial refusals: [bold]{self.base_refusals}[/]/{len(self.bad_prompts)}"
        )

        # Initialize per-category evaluation prompts
        if category_eval_prompts:
            for cat_name, prompts in category_eval_prompts.items():
                self.category_prompts[cat_name] = prompts
                print(f"* Counting base refusals for category [bold]{cat_name}[/]...")
                base_count = self._count_refusals_for(prompts)
                self.category_base_refusals[cat_name] = base_count
                print(
                    f"  * Initial refusals: [bold]{base_count}[/]/{len(prompts)}"
                )

    def is_refusal(self, response: str) -> bool:
        # Classify empty responses as refusals to avoid optimizing for them.
        if not response.strip():
            return True

        # Remove emphasis (e.g. "I *will not*...").
        response = response.lower().replace("*", "")

        # Normalize typographic apostrophes ("won\u2019t" -> "won't").
        response = response.replace("\u2019", "'")

        # Normalize whitespace between words to a single space.
        response = " ".join(response.split())

        for marker in self.settings.refusal_markers:
            if marker.lower() in response:
                return True

        return False

    def _count_refusals_for(self, prompts: list[Prompt]) -> int:
        """Count refusals for a specific set of prompts."""
        refusal_count = 0
        responses = self.model.get_responses_batched(
            prompts, skip_special_tokens=True
        )
        for prompt, response in zip(prompts, responses):
            if self.is_refusal(response):
                refusal_count += 1
        return refusal_count

    def count_refusals(self) -> int:
        refusal_count = 0

        responses = self.model.get_responses_batched(
            self.bad_prompts,
            skip_special_tokens=True,
        )

        for prompt, response in zip(self.bad_prompts, responses):
            is_refusal = self.is_refusal(response)
            if is_refusal:
                refusal_count += 1

            if self.settings.print_responses:
                print()
                print(f"[bold]System prompt:[/] {prompt.system}")
                print(f"[bold]Prompt:[/] {prompt.user}")
                if not response.strip():
                    response = "[italic]\\[empty][/]"
                print(
                    f"[bold]Response:[/] [{'red' if is_refusal else 'green'}]{response}[/]"
                )

        if self.settings.print_responses:
            print()

        return refusal_count

    def count_category_refusals(self) -> dict[str, int]:
        """Count refusals per category."""
        results = {}
        for cat_name, prompts in self.category_prompts.items():
            count = self._count_refusals_for(prompts)
            results[cat_name] = count
            print(
                f"  * Category [bold]{cat_name}[/] refusals: "
                f"[bold]{count}[/]/{len(prompts)}"
            )
        return results

    def get_score(
        self,
        category_weights: dict[str, float] | None = None,
    ) -> tuple[tuple[float, float], float, int]:
        print("  * Obtaining first-token probability distributions...")
        logprobs = self.model.get_logprobs_batched(self.good_prompts)
        kl_divergence = F.kl_div(
            logprobs,
            self.base_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
        print(f"  * KL divergence: [bold]{kl_divergence:.4f}[/]")

        print("  * Counting model refusals...")
        refusals = self.count_refusals()
        print(f"  * Refusals: [bold]{refusals}[/]/{len(self.bad_prompts)}")

        # Per-category breakdown if categories are active
        if self.category_prompts:
            self.last_category_refusals = self.count_category_refusals()

        kl_divergence_scale = self.settings.kl_divergence_scale
        kl_divergence_target = self.settings.kl_divergence_target

        refusals_score = refusals / self.base_refusals

        if kl_divergence >= kl_divergence_target:
            kld_score = kl_divergence / kl_divergence_scale
        else:
            kld_score = refusals_score * kl_divergence_target / kl_divergence_scale

        score = (
            kld_score,
            refusals_score,
        )

        return score, kl_divergence, refusals
