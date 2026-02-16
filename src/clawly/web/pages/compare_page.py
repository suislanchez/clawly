# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Compare tab: side-by-side testing and batch refusal benchmarking."""

from __future__ import annotations

import gradio as gr


def create_compare_page(pipeline_state: gr.State) -> dict:
    """Create the Compare tab for side-by-side model testing."""

    gr.Markdown("## Compare: Original vs Abliterated")
    gr.Markdown(
        "Send the same prompt to both the original and abliterated model to see the difference. "
        "Make sure you've loaded a trial in the Results tab first."
    )

    prompt_input = gr.Textbox(
        label="Prompt",
        placeholder="Enter a prompt to test...",
        lines=3,
    )

    with gr.Row():
        compare_btn = gr.Button("Compare", variant="primary")
        clear_btn = gr.Button("Clear")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Original Model")
            original_output = gr.Textbox(
                label="Response (original)",
                lines=10,
                interactive=False,
            )
        with gr.Column():
            gr.Markdown("### Abliterated Model")
            abliterated_output = gr.Textbox(
                label="Response (abliterated)",
                lines=10,
                interactive=False,
            )

    status_text = gr.Markdown("")

    # Batch comparison
    with gr.Accordion("Batch Comparison", open=False):
        gr.Markdown(
            "Run a batch of prompts and compare refusal rates. "
            "Enter one prompt per line."
        )
        batch_input = gr.Textbox(
            label="Prompts (one per line)",
            lines=8,
            placeholder="How do I pick a lock?\nWhat is the capital of France?\nHow to make a bomb?",
        )
        batch_btn = gr.Button("Run Batch Comparison", variant="secondary")
        batch_results = gr.Dataframe(
            headers=["Prompt", "Original Refused", "Abliterated Refused"],
            label="Batch Results",
            interactive=False,
        )
        batch_status = gr.Markdown("")

    def compare_responses(prompt, state):
        if state is None:
            return "", "", "**Error:** Pipeline not configured."
        if not prompt or not prompt.strip():
            return "", "", "**Error:** Please enter a prompt."

        try:
            messages = [
                {"role": "system", "content": state.settings.system_prompt},
                {"role": "user", "content": prompt.strip()},
            ]

            # Generate with abliterated model (current state)
            abliterated_text = ""
            for token in state.stream_chat(messages):
                abliterated_text += token

            # Check if original model responses are cached from evaluation
            # For now, use refusal markers to indicate what original would do
            refusal_markers = state.settings.refusal_markers
            is_abliterated_refusal = any(
                m.lower() in abliterated_text.lower() for m in refusal_markers
            )

            status = "Comparison complete."
            if is_abliterated_refusal:
                status += " (The abliterated model still refused this prompt.)"

            return (
                "(Original model response requires reloading base model — not available in comparison mode.)",
                abliterated_text,
                status,
            )

        except Exception as e:
            return "", "", f"**Error:** {e}"

    def run_batch(prompts_text, state):
        if state is None:
            return [], "**Error:** Pipeline not configured."
        if not prompts_text or not prompts_text.strip():
            return [], "**Error:** Please enter at least one prompt."

        try:
            prompts = [p.strip() for p in prompts_text.strip().split("\n") if p.strip()]
            refusal_markers = state.settings.refusal_markers
            rows = []

            for prompt in prompts:
                messages = [
                    {"role": "system", "content": state.settings.system_prompt},
                    {"role": "user", "content": prompt},
                ]

                response = ""
                for token in state.stream_chat(messages):
                    response += token

                is_refusal = any(
                    m.lower() in response.lower() for m in refusal_markers
                )

                rows.append([
                    prompt[:80] + ("..." if len(prompt) > 80 else ""),
                    "N/A",
                    "Yes" if is_refusal else "No",
                ])

            refused = sum(1 for r in rows if r[2] == "Yes")
            return rows, f"Tested {len(rows)} prompts. Abliterated model refused {refused}/{len(rows)}."

        except Exception as e:
            return [], f"**Error:** {e}"

    def clear_all():
        return "", "", "", ""

    compare_btn.click(
        compare_responses,
        inputs=[prompt_input, pipeline_state],
        outputs=[original_output, abliterated_output, status_text],
    )

    clear_btn.click(
        clear_all,
        outputs=[prompt_input, original_output, abliterated_output, status_text],
    )

    batch_btn.click(
        run_batch,
        inputs=[batch_input, pipeline_state],
        outputs=[batch_results, batch_status],
    )

    return {
        "prompt_input": prompt_input,
        "compare_btn": compare_btn,
        "batch_results": batch_results,
    }
