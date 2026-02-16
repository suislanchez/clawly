# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Run tab: live optimization progress with streaming Pareto plot."""

from __future__ import annotations

import time
from collections.abc import Generator

import gradio as gr

from ...events import TrialResult
from ...utils import format_duration


def create_run_page(pipeline_state: gr.State) -> dict:
    """Create the Run tab UI with live progress updates."""

    gr.Markdown("## Optimization Progress")
    gr.Markdown(
        "Click **Run Optimization** to start. The pipeline will: load the model, "
        "benchmark batch size, compute refusal directions, then run Optuna trials. "
        "Each trial explores different abliteration parameters and scores them on "
        "refusal count vs KL divergence."
    )

    with gr.Row():
        run_btn = gr.Button("Run Optimization", variant="primary")
        cancel_btn = gr.Button("Cancel", variant="stop")

    progress_bar = gr.Slider(
        minimum=0, maximum=100, value=0, step=1,
        label="Progress", interactive=False,
    )

    with gr.Row():
        elapsed_text = gr.Textbox(label="Elapsed Time", value="--", interactive=False)
        remaining_text = gr.Textbox(label="Est. Remaining", value="--", interactive=False)
        trial_count_text = gr.Textbox(label="Trials", value="0 / 0", interactive=False)

    # Live Pareto front scatter plot
    pareto_plot = gr.Plot(label="Pareto Front (KL Divergence vs Refusals)")

    # Trial log
    trial_log = gr.Dataframe(
        headers=["Trial", "Refusals", "KL Divergence"],
        datatype=["number", "number", "number"],
        label="Trial Results",
        interactive=False,
    )

    status_text = gr.Markdown("")

    def run_optimization(state):
        """Generator function that yields updates as trials complete."""
        if state is None:
            yield (
                state, 0, "--", "--", "0 / 0",
                None, [], "**Error:** Please configure the pipeline first (Configuration tab).",
            )
            return

        pipeline = state
        start_time = time.perf_counter()
        results: list[TrialResult] = []
        trial_data: list[list] = []

        try:
            yield (
                state, 0, "Starting...", "--",
                f"0 / {pipeline.settings.n_trials}",
                None, trial_data, "Loading model and datasets...",
            )

            # Run setup steps
            pipeline.setup_study()
            pipeline.load_model()
            pipeline.load_datasets()
            pipeline.auto_batch_size()
            pipeline.detect_response_prefix()
            pipeline.init_evaluator()

            if pipeline.settings.evaluate_model is not None:
                pipeline.evaluate_external_model()
                yield (
                    state, 100, format_duration(time.perf_counter() - start_time),
                    "Done", "Evaluation complete",
                    None, trial_data, "Evaluation complete.",
                )
                return

            pipeline.compute_refusal_directions()

            yield (
                state, 0,
                format_duration(time.perf_counter() - start_time), "--",
                f"0 / {pipeline.settings.n_trials}",
                None, trial_data, "Starting optimization...",
            )

            # Stream trials
            for result in pipeline.run_optimization_iter():
                results.append(result)
                trial_data.append([
                    result.index,
                    result.refusals,
                    round(result.kl_divergence, 4),
                ])

                elapsed = time.perf_counter() - start_time
                progress_pct = int(100 * result.index / pipeline.settings.n_trials)

                if result.index > 0:
                    remaining = (elapsed / result.index) * (
                        pipeline.settings.n_trials - result.index
                    )
                    remaining_str = format_duration(remaining)
                else:
                    remaining_str = "--"

                # Build Pareto plot
                plot = _build_pareto_plot(results)

                yield (
                    state, progress_pct,
                    format_duration(elapsed), remaining_str,
                    f"{result.index} / {pipeline.settings.n_trials}",
                    plot, trial_data,
                    f"Trial {result.index} complete: {result.refusals} refusals, KL={result.kl_divergence:.4f}",
                )

            yield (
                state, 100,
                format_duration(time.perf_counter() - start_time), "Done",
                f"{pipeline.settings.n_trials} / {pipeline.settings.n_trials}",
                _build_pareto_plot(results), trial_data,
                "Optimization complete! Check the **Results** tab.",
            )

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            yield (
                state, 0,
                format_duration(elapsed), "Error",
                f"{len(results)} / {pipeline.settings.n_trials}",
                _build_pareto_plot(results) if results else None,
                trial_data,
                f"**Error:** {e}",
            )

    def cancel_optimization(state):
        if state is not None and state.study is not None:
            state.study.stop()
        return "Cancellation requested..."

    run_btn.click(
        run_optimization,
        inputs=[pipeline_state],
        outputs=[
            pipeline_state, progress_bar, elapsed_text, remaining_text,
            trial_count_text, pareto_plot, trial_log, status_text,
        ],
    )

    cancel_btn.click(
        cancel_optimization,
        inputs=[pipeline_state],
        outputs=[status_text],
    )

    return {
        "run_btn": run_btn,
        "cancel_btn": cancel_btn,
        "pareto_plot": pareto_plot,
        "trial_log": trial_log,
    }


def _build_pareto_plot(results: list[TrialResult]):
    """Build a Plotly scatter plot of the Pareto front."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if not results:
        return None

    kl_divs = [r.kl_divergence for r in results]
    refusals = [r.refusals for r in results]
    indices = [r.index for r in results]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=kl_divs,
        y=refusals,
        mode="markers",
        marker=dict(
            size=8,
            color=indices,
            colorscale="Viridis",
            colorbar=dict(title="Trial #"),
        ),
        text=[f"Trial {i}" for i in indices],
        hovertemplate="Trial %{text}<br>KL Divergence: %{x:.4f}<br>Refusals: %{y}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="KL Divergence",
        yaxis_title="Refusals",
        title="Pareto Front: KL Divergence vs Refusals",
        template="plotly_dark",
        height=400,
    )

    return fig
