# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Gradio Web UI for Clawly. Launch with ``clawly-web``."""

from __future__ import annotations

import gradio as gr

from .pages.compare_page import create_compare_page
from .pages.config_page import create_config_page
from .pages.export_page import create_export_page
from .pages.research_page import create_research_page
from .pages.results_page import create_results_page
from .pages.run_page import create_run_page

CUSTOM_CSS = """
.clawly-header {
    text-align: center;
    padding: 1rem 0 0.5rem 0;
}
.clawly-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
}
.clawly-header p {
    opacity: 0.7;
    font-size: 0.95rem;
    margin-top: 0;
}
.clawly-banner {
    font-family: monospace;
    font-size: 0.7rem;
    line-height: 1.1;
    text-align: center;
    color: #6ea8fe;
    margin-bottom: 0.5rem;
}
"""


def create_app() -> gr.Blocks:
    """Create the main Gradio application with tabbed layout."""
    with gr.Blocks(
        title="Clawly - Automatic LLM Abliteration",
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CUSTOM_CSS,
    ) as app:
        gr.HTML(
            """<div class="clawly-header">
                <pre class="clawly-banner">█▀▀░█░░░█▀█░█░█░█░░░█░█
█░░░█░░░█▀█░█▄█░█░░░░█░
▀▀▀░▀▀▀░▀░▀░▀░▀░▀▀▀░░▀░</pre>
                <h1>Clawly</h1>
                <p>Automatic LLM abliteration with selective uncensoring</p>
            </div>"""
        )

        # Shared state across tabs
        pipeline_state = gr.State(value=None)

        with gr.Tabs() as tabs:
            with gr.Tab("Configuration", id="config"):
                config_components = create_config_page(pipeline_state)

            with gr.Tab("Run", id="run"):
                run_components = create_run_page(pipeline_state)

            with gr.Tab("Results", id="results"):
                results_components = create_results_page(pipeline_state)

            with gr.Tab("Export", id="export"):
                export_components = create_export_page(pipeline_state)

            with gr.Tab("Compare", id="compare"):
                compare_components = create_compare_page(pipeline_state)

            with gr.Tab("Research", id="research"):
                research_components = create_research_page(pipeline_state)

    return app


def main():
    """Entry point for clawly-web command."""
    from ..pipeline import Pipeline

    Pipeline.configure_environment()

    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
