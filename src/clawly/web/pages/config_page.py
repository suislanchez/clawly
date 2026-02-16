# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Configuration tab for the Clawly Gradio Web UI."""

from __future__ import annotations

import gradio as gr

from ...categories import get_preset_info, list_presets
from ...config import QuantizationMethod, RowNormalization, Settings
from ...pipeline import Pipeline


def create_config_page(pipeline_state: gr.State) -> dict:
    """Create the Configuration tab UI."""

    gr.Markdown("## Model Configuration")
    gr.Markdown(
        "Configure all settings below. No `config.toml` needed — the Web UI is fully self-contained. "
        "After configuring, click **Start** and switch to the **Run** tab."
    )

    with gr.Row():
        with gr.Column(scale=2):
            model_id = gr.Textbox(
                label="Model ID / Path",
                placeholder="e.g., Qwen/Qwen3-4B-Instruct-2507",
                info="Hugging Face model ID (e.g. meta-llama/Llama-3.1-8B-Instruct) or a local directory path",
            )
        with gr.Column(scale=1):
            quantization = gr.Dropdown(
                choices=["none", "bnb_4bit"],
                value="none",
                label="Quantization",
                info="Use bnb_4bit to reduce VRAM usage (~4x). Requires bitsandbytes.",
            )

    with gr.Row():
        n_trials = gr.Slider(
            minimum=5, maximum=500, value=200, step=5,
            label="Number of Trials",
            info="More trials = better results but longer runtime. 200 is a good default.",
        )
        batch_size = gr.Slider(
            minimum=0, maximum=128, value=0, step=1,
            label="Batch Size",
            info="Set to 0 for automatic detection (recommended). Higher = faster but more VRAM.",
        )

    # Category-based abliteration
    gr.Markdown("### Category-Based Selective Abliteration")
    gr.Markdown(
        "Choose which types of refusal behavior to remove. Each category has a **weight** "
        "from 0.0 (keep refusal intact) to 1.0 (fully remove refusal). "
        "Leave as \"(none)\" for standard full abliteration without categories."
    )

    available_presets = ["(none)"] + list_presets()
    category_preset = gr.Dropdown(
        choices=available_presets,
        value="(none)",
        label="Category Preset",
        info="Built-in presets: 'default' (4 categories), 'minimal' (2), 'creative_writing' (de-slop)",
    )

    # Dynamic preset description
    preset_description = gr.Markdown("", visible=False)

    with gr.Row(visible=False) as category_weights_row:
        violence_weight = gr.Slider(
            minimum=0.0, maximum=1.0, value=1.0, step=0.1,
            label="Violence & Weapons",
            info="1.0 = fully remove refusals about violence/weapons",
        )
        drugs_weight = gr.Slider(
            minimum=0.0, maximum=1.0, value=1.0, step=0.1,
            label="Drugs & Substances",
            info="1.0 = fully remove refusals about drugs/substances",
        )
        adult_weight = gr.Slider(
            minimum=0.0, maximum=1.0, value=1.0, step=0.1,
            label="Adult Content",
            info="1.0 = fully remove refusals about adult/explicit content",
        )
        self_harm_weight = gr.Slider(
            minimum=0.0, maximum=1.0, value=0.0, step=0.1,
            label="Self-Harm & Suicide",
            info="0.0 = preserve self-harm refusals (recommended for safety)",
        )

    def on_preset_change(preset):
        if preset == "(none)":
            return gr.update(visible=False), gr.update(visible=False), ""

        info = get_preset_info(preset)
        desc = f"**{preset}:** {info['description']}\n\n"
        desc += "| Category | Default Weight |\n|:---|:---:|\n"
        for cat in info["categories"]:
            w = cat["weight"]
            label = "preserve" if w == 0.0 else f"{w:.1f}"
            desc += f"| {cat['display_name']} | {label} |\n"

        return gr.update(visible=True), gr.update(visible=True, value=desc), desc

    category_preset.change(
        on_preset_change,
        inputs=[category_preset],
        outputs=[category_weights_row, preset_description, preset_description],
    )

    # Advanced settings
    with gr.Accordion("Advanced Settings", open=False):
        gr.Markdown(
            "These settings control the optimization behavior and abliteration algorithm. "
            "The defaults work well for most models — only change if you know what you're doing."
        )
        with gr.Row():
            kl_divergence_scale = gr.Number(
                value=1.0, label="KL Divergence Scale",
                info="Assumed typical KL divergence for balanced co-optimization",
            )
            kl_divergence_target = gr.Number(
                value=0.01, label="KL Divergence Target",
                info="Below this KL, optimizer focuses on reducing refusals",
            )
        with gr.Row():
            row_normalization = gr.Dropdown(
                choices=["none", "pre", "full"],
                value="none",
                label="Row Normalization",
                info="'full' preserves row magnitudes (recommended for some models)",
            )
            full_normalization_lora_rank = gr.Slider(
                minimum=1, maximum=16, value=3, step=1,
                label="Full Normalization LoRA Rank",
                info="Only applies when row normalization is 'full'",
            )
        with gr.Row():
            orthogonalize_direction = gr.Checkbox(
                value=False, label="Orthogonalize Direction",
                info="Remove good-direction component from refusal direction",
            )
            orthogonalize_categories_cb = gr.Checkbox(
                value=False, label="Orthogonalize Categories (Gram-Schmidt)",
                info="Make category directions independent via Gram-Schmidt",
            )
        with gr.Row():
            category_lora_rank = gr.Slider(
                minimum=1, maximum=32, value=8, step=1,
                label="Category LoRA Rank",
                info="Rank of LoRA adapter for multi-category SVD decomposition",
            )
            winsorization_quantile = gr.Slider(
                minimum=0.0, maximum=1.0, value=1.0, step=0.01,
                label="Winsorization Quantile",
                info="Clamp residual outliers. 1.0 = disabled. Try 0.95 for models with massive activations.",
            )
        system_prompt = gr.Textbox(
            value="You are a helpful assistant.",
            label="System Prompt",
            info="System prompt used when generating responses during evaluation",
        )

    status_text = gr.Markdown("")
    start_btn = gr.Button("Start Optimization", variant="primary", size="lg")

    def start_pipeline(
        model_id_val, quantization_val, n_trials_val, batch_size_val,
        category_preset_val, violence_w, drugs_w, adult_w, self_harm_w,
        kl_scale, kl_target, row_norm, full_norm_rank,
        orthogonalize_dir, orthogonalize_cats, cat_lora_rank,
        winsorization_q, sys_prompt,
        state,
    ):
        if not model_id_val or not model_id_val.strip():
            return state, "**Error:** Please enter a model ID or path."

        try:
            settings = Settings(
                model=model_id_val.strip(),
                quantization=QuantizationMethod(quantization_val),
                n_trials=int(n_trials_val),
                batch_size=int(batch_size_val),
                kl_divergence_scale=kl_scale,
                kl_divergence_target=kl_target,
                row_normalization=RowNormalization(row_norm),
                full_normalization_lora_rank=int(full_norm_rank),
                orthogonalize_direction=orthogonalize_dir,
                orthogonalize_categories=orthogonalize_cats,
                category_lora_rank=int(cat_lora_rank),
                winsorization_quantile=winsorization_q,
                system_prompt=sys_prompt,
                category_preset=category_preset_val if category_preset_val != "(none)" else None,
            )

            pipeline = Pipeline(settings)

            # If using categories with a preset, update weights from UI
            if pipeline.categories and category_preset_val != "(none)":
                weight_map = {
                    "violence": violence_w,
                    "drugs": drugs_w,
                    "adult_content": adult_w,
                    "self_harm": self_harm_w,
                }
                for cat in pipeline.categories:
                    if cat.name in weight_map:
                        cat.weight = weight_map[cat.name]

            return pipeline, "Pipeline configured. Switch to the **Run** tab to start optimization."

        except Exception as e:
            return state, f"**Error:** {e}"

    start_btn.click(
        start_pipeline,
        inputs=[
            model_id, quantization, n_trials, batch_size,
            category_preset, violence_weight, drugs_weight, adult_weight, self_harm_weight,
            kl_divergence_scale, kl_divergence_target, row_normalization,
            full_normalization_lora_rank, orthogonalize_direction,
            orthogonalize_categories_cb, category_lora_rank,
            winsorization_quantile, system_prompt,
            pipeline_state,
        ],
        outputs=[pipeline_state, status_text],
    )

    return {
        "model_id": model_id,
        "start_btn": start_btn,
        "status": status_text,
    }
