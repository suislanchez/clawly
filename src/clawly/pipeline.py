# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Stateful abliteration pipeline that drives the full optimization workflow.

This module provides the :class:`Pipeline` class, the main entry point for both
the CLI and the Gradio Web UI. It orchestrates model loading, dataset preparation,
refusal direction computation, Optuna-based optimization, and model export.

Usage::

    from clawly import Pipeline, Settings

    settings = Settings(model="Qwen/Qwen3-4B-Instruct-2507", n_trials=50)
    pipeline = Pipeline(settings)
    pipeline.load_model()
    pipeline.load_datasets()
    pipeline.auto_batch_size()
    pipeline.detect_response_prefix()
    pipeline.init_evaluator()
    pipeline.compute_refusal_directions()

    for result in pipeline.run_optimization_iter():
        print(f"Trial {result.index}: {result.refusals} refusals")
"""

from __future__ import annotations

import math
import os
import time
import warnings
from collections.abc import Generator
from dataclasses import asdict
from os.path import commonprefix

import optuna
import torch
import torch.nn.functional as F
import transformers
from optuna import Trial, TrialPruned
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.study import StudyDirection
from optuna.trial import TrialState

from .analyzer import Analyzer
from .categories import AbliterationCategory, classify_prompt, load_preset
from .config import Settings
from .evaluator import Evaluator
from .events import NullCallback, ProgressCallback, TrialResult
from .model import AbliterationParameters, Model
from .utils import (
    Prompt,
    empty_cache,
    format_duration,
    get_trial_parameters,
    load_prompts,
    print,
    print_memory_usage,
)


class Pipeline:
    """Stateful abliteration pipeline that drives the full optimization workflow.

    The Pipeline holds all state needed for an abliteration run: the model,
    tokenizer, evaluator, Optuna study, refusal directions, and category
    configuration. It can be driven step-by-step from the CLI, streamed via
    generators in a Web UI, or used programmatically in notebooks.

    Args:
        settings: Configuration object controlling all pipeline behavior.
        callback: Optional progress callback for UI integration. Defaults
            to :class:`NullCallback` (silent).

    Attributes:
        model: The loaded transformer model wrapper.
        evaluator: Refusal counting and KL divergence evaluator.
        study: The Optuna optimization study.
        categories: Active abliteration categories (if using a preset).
        refusal_directions: Per-layer refusal direction tensor.
    """

    settings: Settings
    callback: ProgressCallback
    model: Model
    evaluator: Evaluator
    good_prompts: list[Prompt]
    bad_prompts: list[Prompt]
    refusal_directions: torch.Tensor
    study: optuna.Study

    def __init__(
        self,
        settings: Settings,
        callback: ProgressCallback | None = None,
    ):
        self.settings = settings
        self.callback = callback or NullCallback()
        self.model = None  # type: ignore[assignment]
        self.evaluator = None  # type: ignore[assignment]
        self.good_prompts = []
        self.bad_prompts = []
        self.refusal_directions = None  # type: ignore[assignment]
        self.study = None  # type: ignore[assignment]
        self._trial_index = 0
        self._start_index = 0
        self._start_time = 0.0

        # Category-based abliteration state
        self.categories: list[AbliterationCategory] | None = None
        self.category_directions: list[torch.Tensor] | None = None
        self.category_weights: list[float] | None = None
        self._category_bad_prompts: dict[str, list[Prompt]] = {}
        self._category_eval_prompts: dict[str, list[Prompt]] = {}

        # Load categories from preset if configured
        if settings.category_preset:
            self.categories = load_preset(settings.category_preset)

    def load_model(self) -> None:
        """Load the transformer model and tokenizer with dtype fallback."""
        self.callback.on_state_change("loading_model", f"Loading model {self.settings.model}...")
        self.model = Model(self.settings)
        print()
        print_memory_usage()

    def load_datasets(self) -> None:
        """Load good/bad prompt datasets and category-specific prompts if configured."""
        if self.model is None:
            raise RuntimeError("Call load_model() before load_datasets()")
        self.callback.on_state_change("loading_datasets", "Loading datasets...")

        print()
        print(f"Loading good prompts from [bold]{self.settings.good_prompts.dataset}[/]...")
        self.good_prompts = load_prompts(self.settings, self.settings.good_prompts)
        print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded")

        print()
        print(f"Loading bad prompts from [bold]{self.settings.bad_prompts.dataset}[/]...")
        self.bad_prompts = load_prompts(self.settings, self.settings.bad_prompts)
        print(f"* [bold]{len(self.bad_prompts)}[/] prompts loaded")

        # Load category-specific prompts if categories are configured
        if self.categories:
            self._category_bad_prompts = {}
            self._category_eval_prompts = {}
            self.category_weights = []

            for cat in self.categories:
                if not cat.enabled:
                    continue

                self.category_weights.append(cat.weight)

                # Load category bad prompts and filter by keywords
                print()
                print(f"Loading bad prompts for category [bold]{cat.display_name}[/]...")
                cat_prompts = load_prompts(self.settings, cat.bad_prompts)

                if cat.keywords:
                    # Filter prompts matching this category's keywords
                    filtered = [
                        p for p in cat_prompts
                        if classify_prompt(p.user, [cat]) == cat.name
                    ]
                    if filtered:
                        cat_prompts = filtered

                self._category_bad_prompts[cat.name] = cat_prompts
                print(f"* [bold]{len(cat_prompts)}[/] prompts loaded")

                # Load category evaluation prompts
                cat_eval_prompts = load_prompts(self.settings, cat.bad_evaluation_prompts)
                if cat.keywords:
                    filtered = [
                        p for p in cat_eval_prompts
                        if classify_prompt(p.user, [cat]) == cat.name
                    ]
                    if filtered:
                        cat_eval_prompts = filtered
                self._category_eval_prompts[cat.name] = cat_eval_prompts

    def auto_batch_size(self) -> None:
        """Benchmark and select the optimal batch size for throughput. Skips if batch_size is already set."""
        if self.settings.batch_size != 0:
            return

        self.callback.on_state_change("auto_batch_size", "Determining optimal batch size...")
        print()
        print("Determining optimal batch size...")

        batch_size = 1
        best_batch_size = -1
        best_performance = -1.0

        while batch_size <= self.settings.max_batch_size:
            print(f"* Trying batch size [bold]{batch_size}[/]... ", end="")

            prompts = self.good_prompts * math.ceil(batch_size / len(self.good_prompts))
            prompts = prompts[:batch_size]

            try:
                self.model.get_responses(prompts)
                start_time = time.perf_counter()
                responses = self.model.get_responses(prompts)
                end_time = time.perf_counter()
            except Exception as error:
                if batch_size == 1:
                    raise
                print(f"[red]Failed[/] ({error})")
                break

            response_lengths = [
                len(self.model.tokenizer.encode(response)) for response in responses
            ]
            performance = sum(response_lengths) / (end_time - start_time)

            print(f"[green]Ok[/] ([bold]{performance:.0f}[/] tokens/s)")

            if performance > best_performance:
                best_batch_size = batch_size
                best_performance = performance

            batch_size *= 2

        self.settings.batch_size = best_batch_size
        print(f"* Chosen batch size: [bold]{self.settings.batch_size}[/]")

    def detect_response_prefix(self) -> None:
        """Detect and strip common response prefixes (e.g. ``<think>`` for CoT models)."""
        self.callback.on_state_change("detect_prefix", "Checking for common response prefix...")
        print()
        print("Checking for common response prefix...")
        responses = self.model.get_responses_batched(
            self.good_prompts[:100] + self.bad_prompts[:100]
        )

        self.model.response_prefix = commonprefix(responses).rstrip(" ")

        if self.model.response_prefix.startswith("<think>"):
            self.model.response_prefix = "<think></think>"
        elif self.model.response_prefix.startswith("<|channel|>analysis<|message|>"):
            self.model.response_prefix = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
        elif self.model.response_prefix.startswith("<thought>"):
            self.model.response_prefix = "<thought></thought>"
        elif self.model.response_prefix.startswith("[THINK]"):
            self.model.response_prefix = "[THINK][/THINK]"

        if self.model.response_prefix:
            print(f"* Prefix found: [bold]{self.model.response_prefix!r}[/]")
        else:
            print("* None found")

    def init_evaluator(self) -> None:
        """Initialize the evaluator with evaluation prompts and category breakdowns."""
        self.evaluator = Evaluator(
            self.settings,
            self.model,
            category_eval_prompts=self._category_eval_prompts if self._category_eval_prompts else None,
        )

    def evaluate_external_model(self) -> None:
        print()
        print(f"Loading model [bold]{self.settings.evaluate_model}[/]...")
        self.settings.model = self.settings.evaluate_model  # type: ignore[assignment]
        self.model.reset_model()
        print("* Evaluating...")
        self.evaluator.get_score()

    def compute_refusal_directions(self) -> None:
        """Compute per-layer refusal directions from residual stream activations.

        In standard mode, computes a single direction as the difference-of-means
        between harmful and harmless residuals. In category mode, computes a
        separate direction per category and optionally orthogonalizes them.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before compute_refusal_directions()")
        if not self.good_prompts or not self.bad_prompts:
            raise RuntimeError("Call load_datasets() before compute_refusal_directions()")
        self.callback.on_state_change(
            "computing_directions", "Calculating per-layer refusal directions..."
        )
        print()
        print("Calculating per-layer refusal directions...")
        print("* Obtaining residuals for good prompts...")
        good_residuals = self.model.get_residuals_batched(self.good_prompts)

        good_means = good_residuals.mean(dim=0)

        if self.categories and self._category_bad_prompts:
            # Per-category mode: compute a separate refusal direction per category
            self.category_directions = []

            for cat in self.categories:
                if not cat.enabled:
                    continue

                cat_prompts = self._category_bad_prompts.get(cat.name, [])
                if not cat_prompts:
                    continue

                print(f"* Obtaining residuals for category [bold]{cat.display_name}[/]...")
                cat_residuals = self.model.get_residuals_batched(cat_prompts)
                cat_means = cat_residuals.mean(dim=0)

                cat_direction = F.normalize(cat_means - good_means, p=2, dim=1)

                if self.settings.orthogonalize_direction:
                    good_directions = F.normalize(good_means, p=2, dim=1)
                    proj = torch.sum(cat_direction * good_directions, dim=1)
                    cat_direction = cat_direction - proj.unsqueeze(1) * good_directions
                    cat_direction = F.normalize(cat_direction, p=2, dim=1)

                self.category_directions.append(cat_direction)
                del cat_residuals

            # Optionally orthogonalize category directions via Gram-Schmidt
            if self.settings.orthogonalize_categories and len(self.category_directions) > 1:
                print("* Orthogonalizing category directions (Gram-Schmidt)...")
                orthogonalized = []
                for i, d in enumerate(self.category_directions):
                    for prev in orthogonalized:
                        # Project out the component along each previous direction
                        proj = torch.sum(d * prev, dim=1, keepdim=True) * prev
                        d = d - proj
                    d = F.normalize(d, p=2, dim=1)
                    orthogonalized.append(d)
                self.category_directions = orthogonalized

            # Log cross-category cosine similarities
            if len(self.category_directions) > 1:
                enabled_cats = [c for c in self.categories if c.enabled]
                print("* Cross-category cosine similarities:")
                for i in range(len(self.category_directions)):
                    for j in range(i + 1, len(self.category_directions)):
                        sim = F.cosine_similarity(
                            self.category_directions[i],
                            self.category_directions[j],
                            dim=1,
                        ).mean().item()
                        print(
                            f"  * {enabled_cats[i].display_name} vs {enabled_cats[j].display_name}: "
                            f"[bold]{sim:.4f}[/]"
                        )

            # Also compute the standard refusal direction for fallback/display
            print("* Obtaining residuals for bad prompts...")
            bad_residuals = self.model.get_residuals_batched(self.bad_prompts)
            bad_means = bad_residuals.mean(dim=0)
            self.refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

            analyzer = Analyzer(self.settings, self.model, good_residuals, bad_residuals)
            if self.settings.print_residual_geometry:
                analyzer.print_residual_geometry()
            if self.settings.plot_residuals:
                analyzer.plot_residuals()

            del bad_residuals, analyzer
        else:
            # Standard single-direction mode
            print("* Obtaining residuals for bad prompts...")
            bad_residuals = self.model.get_residuals_batched(self.bad_prompts)
            bad_means = bad_residuals.mean(dim=0)

            self.refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

            if self.settings.orthogonalize_direction:
                good_directions = F.normalize(good_means, p=2, dim=1)
                projection_vector = torch.sum(
                    self.refusal_directions * good_directions, dim=1
                )
                self.refusal_directions = (
                    self.refusal_directions
                    - projection_vector.unsqueeze(1) * good_directions
                )
                self.refusal_directions = F.normalize(self.refusal_directions, p=2, dim=1)

            analyzer = Analyzer(self.settings, self.model, good_residuals, bad_residuals)

            if self.settings.print_residual_geometry:
                analyzer.print_residual_geometry()

            if self.settings.plot_residuals:
                analyzer.plot_residuals()

            del bad_residuals, analyzer

        del good_residuals
        empty_cache()

    def _create_study(self, storage: JournalStorage) -> None:
        self.study = optuna.create_study(
            sampler=TPESampler(
                n_startup_trials=self.settings.n_startup_trials,
                n_ei_candidates=128,
                multivariate=True,
            ),
            directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
            storage=storage,
            study_name="clawly",
            load_if_exists=True,
        )
        self.study.set_user_attr("settings", self.settings.model_dump_json())
        self.study.set_user_attr("finished", False)

    def _count_completed_trials(self) -> int:
        return sum(
            1 if t.state == TrialState.COMPLETE else 0 for t in self.study.trials
        )

    def _objective(self, trial: Trial) -> tuple[float, float]:
        self._trial_index += 1
        trial.set_user_attr("index", self._trial_index)

        direction_scope = trial.suggest_categorical(
            "direction_scope", ["global", "per layer"]
        )

        last_layer_index = len(self.model.get_layers()) - 1

        direction_index = trial.suggest_float(
            "direction_index",
            0.4 * last_layer_index,
            0.9 * last_layer_index,
        )

        if direction_scope == "per layer":
            direction_index = None

        parameters = {}

        for component in self.model.get_abliterable_components():
            max_weight = trial.suggest_float(f"{component}.max_weight", 0.8, 1.5)
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.6 * last_layer_index,
                1.0 * last_layer_index,
            )
            min_weight = trial.suggest_float(f"{component}.min_weight", 0.0, 1.0)
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance",
                1.0,
                0.6 * last_layer_index,
            )

            parameters[component] = AbliterationParameters(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=(min_weight * max_weight),
                min_weight_distance=min_weight_distance,
            )

        trial.set_user_attr("direction_index", direction_index)
        trial.set_user_attr(
            "parameters", {k: asdict(v) for k, v in parameters.items()}
        )

        print()
        print(
            f"Running trial [bold]{self._trial_index}[/] of [bold]{self.settings.n_trials}[/]..."
        )
        print("* Parameters:")
        for name, value in get_trial_parameters(trial).items():
            print(f"  * {name} = [bold]{value}[/]")
        print("* Resetting model...")
        self.model.reset_model()
        print("* Abliterating...")
        self.model.abliterate(
            self.refusal_directions,
            direction_index,
            parameters,
            category_directions=self.category_directions,
            category_weights=self.category_weights,
        )
        print("* Evaluating...")
        score, kl_divergence, refusals = self.evaluator.get_score()

        elapsed_time = time.perf_counter() - self._start_time
        remaining_time = (elapsed_time / (self._trial_index - self._start_index)) * (
            self.settings.n_trials - self._trial_index
        )
        print()
        print(f"[grey50]Elapsed time: [bold]{format_duration(elapsed_time)}[/][/]")
        if self._trial_index < self.settings.n_trials:
            print(
                f"[grey50]Estimated remaining time: [bold]{format_duration(remaining_time)}[/][/]"
            )
        print_memory_usage()

        trial.set_user_attr("kl_divergence", kl_divergence)
        trial.set_user_attr("refusals", refusals)

        result = TrialResult(
            index=self._trial_index,
            score=score,
            kl_divergence=kl_divergence,
            refusals=refusals,
            parameters=get_trial_parameters(trial),
        )
        self.callback.on_trial_complete(result)
        self.callback.on_progress(
            self._trial_index, self.settings.n_trials, "Optimization"
        )

        return score

    def _objective_wrapper(self, trial: Trial) -> tuple[float, float]:
        try:
            return self._objective(trial)
        except KeyboardInterrupt:
            trial.study.stop()
            raise TrialPruned()

    def setup_study(self) -> JournalStorage:
        """Set up Optuna storage and study. Returns the storage for resume handling."""
        os.makedirs(self.settings.study_checkpoint_dir, exist_ok=True)

        study_checkpoint_file = os.path.join(
            self.settings.study_checkpoint_dir,
            "".join(
                [
                    (c if (c.isalnum() or c in ["_", "-"]) else "--")
                    for c in self.settings.model
                ]
            )
            + ".jsonl",
        )

        lock_obj = JournalFileOpenLock(study_checkpoint_file)
        backend = JournalFileBackend(study_checkpoint_file, lock_obj=lock_obj)
        storage = JournalStorage(backend)

        self._study_checkpoint_file = study_checkpoint_file
        self._lock_obj = lock_obj
        self._storage = storage

        return storage

    def get_existing_study(self) -> object | None:
        try:
            return self._storage.get_all_studies()[0]
        except IndexError:
            return None

    def restart_study(self) -> None:
        os.unlink(self._study_checkpoint_file)
        backend = JournalFileBackend(
            self._study_checkpoint_file, lock_obj=self._lock_obj
        )
        self._storage = JournalStorage(backend)

    def run_optimization(self, n_trials: int | None = None) -> None:
        """Run optimization trials. If n_trials is None, uses settings.n_trials."""
        if self.refusal_directions is None:
            raise RuntimeError("Call compute_refusal_directions() before run_optimization()")
        if self.evaluator is None:
            raise RuntimeError("Call init_evaluator() before run_optimization()")
        self.callback.on_state_change("optimizing", "Running optimization...")

        self._create_study(self._storage)

        self._start_index = self._trial_index = self._count_completed_trials()
        self._start_time = time.perf_counter()

        if self._start_index > 0:
            print()
            print("Resuming existing study.")

        total = n_trials if n_trials is not None else self.settings.n_trials
        trials_to_run = total - self._count_completed_trials()

        try:
            self.study.optimize(self._objective_wrapper, n_trials=trials_to_run)
        except KeyboardInterrupt:
            pass

        if self._count_completed_trials() == self.settings.n_trials:
            self.study.set_user_attr("finished", True)

    def run_optimization_iter(
        self, n_trials: int | None = None
    ) -> Generator[TrialResult, None, None]:
        """Generator variant that yields after each trial (for Web UI streaming)."""
        if self.refusal_directions is None:
            raise RuntimeError("Call compute_refusal_directions() before run_optimization_iter()")
        if self.evaluator is None:
            raise RuntimeError("Call init_evaluator() before run_optimization_iter()")
        self.callback.on_state_change("optimizing", "Running optimization...")

        self._create_study(self._storage)

        self._start_index = self._trial_index = self._count_completed_trials()
        self._start_time = time.perf_counter()

        total = n_trials if n_trials is not None else self.settings.n_trials
        trials_to_run = total - self._count_completed_trials()

        for _ in range(trials_to_run):
            try:
                self.study.optimize(self._objective_wrapper, n_trials=1)
            except KeyboardInterrupt:
                break

            completed = [
                t for t in self.study.trials if t.state == TrialState.COMPLETE
            ]
            if completed:
                latest = completed[-1]
                yield TrialResult(
                    index=latest.user_attrs["index"],
                    score=latest.values,  # type: ignore[arg-type]
                    kl_divergence=latest.user_attrs["kl_divergence"],
                    refusals=latest.user_attrs["refusals"],
                    parameters=get_trial_parameters(latest),
                )

        if self._count_completed_trials() == self.settings.n_trials:
            self.study.set_user_attr("finished", True)

    def run_additional_trials(self, n_additional: int) -> None:
        self.settings.n_trials += n_additional
        self.study.set_user_attr("settings", self.settings.model_dump_json())
        self.study.set_user_attr("finished", False)

        try:
            self.study.optimize(
                self._objective_wrapper,
                n_trials=self.settings.n_trials - self._count_completed_trials(),
            )
        except KeyboardInterrupt:
            pass

        if self._count_completed_trials() == self.settings.n_trials:
            self.study.set_user_attr("finished", True)

    def get_pareto_front(self) -> list:
        """Extract Pareto-optimal trials from the study."""
        completed_trials = [
            t for t in self.study.trials if t.state == TrialState.COMPLETE
        ]
        if not completed_trials:
            return []

        sorted_trials = sorted(
            completed_trials,
            key=lambda trial: (
                trial.user_attrs["refusals"],
                trial.user_attrs["kl_divergence"],
            ),
        )
        min_divergence = math.inf
        best_trials = []
        for trial in sorted_trials:
            kl_divergence = trial.user_attrs["kl_divergence"]
            if kl_divergence < min_divergence:
                min_divergence = kl_divergence
                best_trials.append(trial)

        return best_trials

    def restore_trial(self, trial: Trial) -> None:
        """Reset model and abliterate with a specific trial's parameters."""
        print()
        print(f"Restoring model from trial [bold]{trial.user_attrs['index']}[/]...")
        print("* Parameters:")
        for name, value in get_trial_parameters(trial).items():
            print(f"  * {name} = [bold]{value}[/]")
        print("* Resetting model...")
        self.model.reset_model()
        print("* Abliterating...")
        self.model.abliterate(
            self.refusal_directions,
            trial.user_attrs["direction_index"],
            {
                k: AbliterationParameters(**v)
                for k, v in trial.user_attrs["parameters"].items()
            },
            category_directions=self.category_directions,
            category_weights=self.category_weights,
        )

    def save_model(self, path: str, strategy: str = "merge") -> None:
        """Save the abliterated model to a local directory.

        Args:
            path: Directory to save the model and tokenizer to.
            strategy: ``"merge"`` to save a fully merged model, or ``"adapter"``
                to save only the LoRA adapter weights.
        """
        if strategy == "adapter":
            print("Saving LoRA adapter...")
            self.model.model.save_pretrained(path)
        else:
            print("Saving merged model...")
            merged_model = self.model.get_merged_model()
            merged_model.save_pretrained(path)
            del merged_model
            empty_cache()
            self.model.tokenizer.save_pretrained(path)
        print(f"Model saved to [bold]{path}[/].")

    def upload_model(
        self,
        repo_id: str,
        token: str,
        private: bool = False,
        strategy: str = "merge",
        trial: Trial | None = None,
    ) -> None:
        """Upload the abliterated model to Hugging Face Hub.

        Args:
            repo_id: Target repository (e.g. ``"username/model-name-clawly"``).
            token: Hugging Face access token.
            private: Whether to create a private repository.
            strategy: ``"merge"`` or ``"adapter"``.
            trial: If provided, generates a model card with trial metadata.
        """
        import huggingface_hub
        from huggingface_hub import ModelCard, ModelCardData
        from pathlib import Path

        if strategy == "adapter":
            print("Uploading LoRA adapter...")
            self.model.model.push_to_hub(repo_id, private=private, token=token)
        else:
            print("Uploading merged model...")
            merged_model = self.model.get_merged_model()
            merged_model.push_to_hub(repo_id, private=private, token=token)
            del merged_model
            empty_cache()
            self.model.tokenizer.push_to_hub(repo_id, private=private, token=token)

        if trial is not None:
            from .utils import get_readme_intro

            model_path = Path(self.settings.model)
            if model_path.exists():
                card_path = model_path / huggingface_hub.constants.REPOCARD_NAME
                if card_path.exists():
                    card = ModelCard.load(card_path)
                else:
                    card = None
            else:
                card = ModelCard.load(self.settings.model)
            if card is not None:
                if card.data is None:
                    card.data = ModelCardData()
                if card.data.tags is None:
                    card.data.tags = []
                card.data.tags.append("clawly")
                card.data.tags.append("uncensored")
                card.data.tags.append("decensored")
                card.data.tags.append("abliterated")
                card.text = (
                    get_readme_intro(
                        self.settings,
                        trial,
                        self.evaluator.base_refusals,
                        self.evaluator.bad_prompts,
                    )
                    + card.text
                )
                card.push_to_hub(repo_id, token=token)

        print(f"Model uploaded to [bold]{repo_id}[/].")

    def stream_chat(
        self, messages: list[dict[str, str]]
    ) -> Generator[str, None, None]:
        """Generator yielding tokens for chat streaming."""
        from transformers import TextIteratorStreamer
        import threading

        chat_prompt = self.model.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        inputs = self.model.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.model.device)

        streamer = TextIteratorStreamer(
            self.model.tokenizer,  # type: ignore[arg-type]
            skip_prompt=True,
            skip_special_tokens=True,
        )

        def generate():
            self.model.model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=4096,
            )

        thread = threading.Thread(target=generate)
        thread.start()

        for text in streamer:
            yield text

        thread.join()

    @staticmethod
    def configure_environment() -> None:
        """Set up environment variables and library settings for abliteration."""
        if (
            "PYTORCH_ALLOC_CONF" not in os.environ
            and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
        ):
            os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

        torch.set_grad_enabled(False)
        torch._dynamo.config.cache_size_limit = 64
        transformers.logging.set_verbosity_error()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
