# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Progress callback protocol and trial result dataclass.

Defines the :class:`ProgressCallback` protocol that decouples the pipeline
from any specific UI (CLI, Web, notebook). Implementations can react to
state changes, trial completions, and progress updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class TrialResult:
    """Result of a single optimization trial.

    Attributes:
        index: 1-based trial index within the study.
        score: Two-objective score tuple ``(refusal_score, kl_score)``.
        kl_divergence: KL divergence from the original model.
        refusals: Number of refusals detected in evaluation prompts.
        parameters: Human-readable parameter names and values.
        category_refusals: Per-category refusal counts (if categories active).
    """

    index: int
    score: tuple[float, float]
    kl_divergence: float
    refusals: int
    parameters: dict[str, str]
    category_refusals: dict[str, int] | None = None


class ProgressCallback(Protocol):
    """Protocol for receiving pipeline progress updates.

    Implement this to integrate the pipeline with any UI framework.
    """

    def on_state_change(self, state: str, message: str) -> None: ...
    def on_trial_complete(self, result: TrialResult) -> None: ...
    def on_log(self, message: str) -> None: ...
    def on_progress(self, current: int, total: int, desc: str) -> None: ...


class NullCallback:
    """No-op callback for headless or programmatic usage."""

    def on_state_change(self, state: str, message: str) -> None:
        pass

    def on_trial_complete(self, result: TrialResult) -> None:
        pass

    def on_log(self, message: str) -> None:
        pass

    def on_progress(self, current: int, total: int, desc: str) -> None:
        pass
