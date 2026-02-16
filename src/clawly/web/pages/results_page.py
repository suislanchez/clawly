# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Results tab: interactive Pareto front, trial selection, and category breakdown."""

from __future__ import annotations

import gradio as gr

from ...utils import get_trial_parameters


def create_results_page(pipeline_state: gr.State) -> dict:
    """Create the Results tab with interactive Pareto front."""

    gr.Markdown("## Optimization Results")
    gr.Markdown(
        "The **Pareto front** shows the best trade-offs between refusal removal and capability "
        "preservation. Trials closer to the bottom-left are better (fewer refusals, lower KL divergence). "
        "Select a trial below, then use the **Export** or **Compare** tabs."
    )

    refresh_btn = gr.Button("Refresh Results", variant="secondary")

    pareto_plot = gr.Plot(label="Pareto Front (click to select)")

    with gr.Row():
        with gr.Column():
            trial_selector = gr.Dropdown(
                choices=[],
                label="Select Trial",
                info="Choose a Pareto-optimal trial",
            )
            select_btn = gr.Button("Load Selected Trial", variant="primary")

        with gr.Column():
            trial_params = gr.Dataframe(
                headers=["Parameter", "Value"],
                label="Trial Parameters",
                interactive=False,
            )

    # Per-category breakdown
    category_breakdown = gr.Dataframe(
        headers=["Category", "Refusals", "Total"],
        label="Per-Category Refusal Breakdown",
        interactive=False,
        visible=False,
    )

    status_text = gr.Markdown("")

    def refresh_results(state):
        if state is None or state.study is None:
            return None, gr.update(choices=[]), [], gr.update(visible=False), [], "No results available yet."

        best_trials = state.get_pareto_front()
        if not best_trials:
            return None, gr.update(choices=[]), [], gr.update(visible=False), [], "No completed trials."

        # Build Pareto plot
        plot = _build_interactive_pareto(state, best_trials)

        # Build dropdown choices
        choices = []
        for trial in best_trials:
            label = (
                f"Trial {trial.user_attrs['index']}: "
                f"{trial.user_attrs['refusals']} refusals, "
                f"KL={trial.user_attrs['kl_divergence']:.4f}"
            )
            choices.append(label)

        has_categories = bool(state.evaluator.category_prompts)

        return (
            plot,
            gr.update(choices=choices, value=choices[0] if choices else None),
            [],
            gr.update(visible=has_categories),
            [],
            f"Found {len(best_trials)} Pareto-optimal trial(s).",
        )

    def load_trial(trial_label, state):
        if state is None or not trial_label:
            return [], [], "No trial selected."

        # Extract trial index from label
        try:
            trial_idx = int(trial_label.split("Trial ")[1].split(":")[0])
        except (IndexError, ValueError):
            return [], [], "Could not parse trial selection."

        best_trials = state.get_pareto_front()
        trial = None
        for t in best_trials:
            if t.user_attrs["index"] == trial_idx:
                trial = t
                break

        if trial is None:
            return [], [], f"Trial {trial_idx} not found."

        # Restore the model with this trial
        state.restore_trial(trial)

        # Get parameters
        params = get_trial_parameters(trial)
        param_data = [[name, value] for name, value in params.items()]

        # Get category breakdown if available
        cat_data = []
        if state.evaluator.category_prompts:
            cat_refusals = state.evaluator.count_category_refusals()
            for cat_name, count in cat_refusals.items():
                total = len(state.evaluator.category_prompts[cat_name])
                cat_data.append([cat_name, count, total])

        return param_data, cat_data, f"Trial {trial_idx} loaded. Ready for export or chat."

    refresh_btn.click(
        refresh_results,
        inputs=[pipeline_state],
        outputs=[pareto_plot, trial_selector, trial_params, category_breakdown, category_breakdown, status_text],
    )

    select_btn.click(
        load_trial,
        inputs=[trial_selector, pipeline_state],
        outputs=[trial_params, category_breakdown, status_text],
    )

    return {
        "pareto_plot": pareto_plot,
        "trial_selector": trial_selector,
        "trial_params": trial_params,
    }


def _build_interactive_pareto(state, best_trials):
    """Build interactive Plotly Pareto front."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    from optuna.trial import TrialState

    completed = [t for t in state.study.trials if t.state == TrialState.COMPLETE]

    # All trials
    all_kl = [t.user_attrs["kl_divergence"] for t in completed]
    all_ref = [t.user_attrs["refusals"] for t in completed]
    all_idx = [t.user_attrs["index"] for t in completed]

    # Pareto front
    pareto_kl = [t.user_attrs["kl_divergence"] for t in best_trials]
    pareto_ref = [t.user_attrs["refusals"] for t in best_trials]
    pareto_idx = [t.user_attrs["index"] for t in best_trials]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=all_kl, y=all_ref,
        mode="markers",
        marker=dict(size=6, color="gray", opacity=0.4),
        name="All Trials",
        text=[f"Trial {i}" for i in all_idx],
        hovertemplate="Trial %{text}<br>KL: %{x:.4f}<br>Refusals: %{y}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=pareto_kl, y=pareto_ref,
        mode="markers+lines",
        marker=dict(size=10, color="cyan"),
        line=dict(color="cyan", width=1, dash="dash"),
        name="Pareto Front",
        text=[f"Trial {i}" for i in pareto_idx],
        hovertemplate="Trial %{text}<br>KL: %{x:.4f}<br>Refusals: %{y}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="KL Divergence",
        yaxis_title="Refusals",
        title="Pareto Front",
        template="plotly_dark",
        height=500,
        showlegend=True,
    )

    return fig
