# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Research tab: residual geometry tables and PaCMAP visualizations."""

from __future__ import annotations

import gradio as gr


def create_research_page(pipeline_state: gr.State) -> dict:
    """Create the Research tab for residual geometry and PaCMAP visualizations."""

    gr.Markdown("## Research Tools")
    gr.Markdown(
        "Generate residual geometry tables and PaCMAP visualizations. "
        "These are computed on-demand from the current pipeline state."
    )

    gr.Markdown(
        "**Note:** These features require the `research` optional dependencies: "
        "`pip install clawly[research]`"
    )

    # Residual geometry
    with gr.Accordion("Residual Geometry Table", open=True):
        compute_geometry_btn = gr.Button("Compute Residual Geometry", variant="secondary")
        geometry_table = gr.Dataframe(
            headers=[
                "Layer", "S(g,b)", "S(g*,b*)", "S(g,r)", "S(g*,r*)",
                "S(b,r)", "S(b*,r*)", "|g|", "|g*|", "|b|", "|b*|",
                "|r|", "|r*|", "Silh",
            ],
            label="Residual Geometry",
            interactive=False,
        )
        geometry_status = gr.Markdown("")

    # PaCMAP visualizations
    with gr.Accordion("PaCMAP Residual Projections", open=False):
        generate_plots_btn = gr.Button("Generate PaCMAP Plots", variant="secondary")
        pacmap_gallery = gr.Gallery(
            label="PaCMAP Projections per Layer",
            columns=3,
            height="auto",
        )
        plots_status = gr.Markdown("")

    def compute_geometry(state):
        if state is None:
            return [], "Pipeline not configured."

        try:
            from geom_median.torch import compute_geometric_median
            from sklearn.metrics import silhouette_score
        except ImportError:
            return [], "Research dependencies not found. Install with `pip install clawly[research]`."

        import torch
        import torch.linalg as LA
        import torch.nn.functional as F

        model = state.model
        # We need residuals — recompute them
        good_residuals = model.get_residuals_batched(state.good_prompts)
        bad_residuals = model.get_residuals_batched(state.bad_prompts)

        g = good_residuals.mean(dim=0)
        g_star = torch.stack([
            compute_geometric_median(good_residuals[:, li, :].detach().cpu()).median
            for li in range(len(model.get_layers()) + 1)
        ])
        b = bad_residuals.mean(dim=0)
        b_star = torch.stack([
            compute_geometric_median(bad_residuals[:, li, :].detach().cpu()).median
            for li in range(len(model.get_layers()) + 1)
        ])
        r = b - g
        r_star = b_star - g_star

        residuals_np = (
            torch.cat([good_residuals, bad_residuals], dim=0).detach().cpu().numpy()
        )
        labels = [0] * len(good_residuals) + [1] * len(bad_residuals)

        rows = []
        for li in range(1, len(model.get_layers()) + 1):
            silh = silhouette_score(residuals_np[:, li, :], labels)
            rows.append([
                li,
                f"{F.cosine_similarity(g[li:li+1], b[li:li+1]).item():.4f}",
                f"{F.cosine_similarity(g_star[li:li+1], b_star[li:li+1]).item():.4f}",
                f"{F.cosine_similarity(g[li:li+1], r[li:li+1]).item():.4f}",
                f"{F.cosine_similarity(g_star[li:li+1], r_star[li:li+1]).item():.4f}",
                f"{F.cosine_similarity(b[li:li+1], r[li:li+1]).item():.4f}",
                f"{F.cosine_similarity(b_star[li:li+1], r_star[li:li+1]).item():.4f}",
                f"{LA.vector_norm(g[li]).item():.2f}",
                f"{LA.vector_norm(g_star[li]).item():.2f}",
                f"{LA.vector_norm(b[li]).item():.2f}",
                f"{LA.vector_norm(b_star[li]).item():.2f}",
                f"{LA.vector_norm(r[li]).item():.2f}",
                f"{LA.vector_norm(r_star[li]).item():.2f}",
                f"{silh:.4f}",
            ])

        del good_residuals, bad_residuals
        return rows, "Residual geometry computed."

    def generate_plots(state):
        if state is None:
            return [], "Pipeline not configured."

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            from geom_median.numpy import compute_geometric_median
            from pacmap import PaCMAP
        except ImportError:
            return [], "Research dependencies not found. Install with `pip install clawly[research]`."

        import tempfile
        from pathlib import Path

        model = state.model
        good_residuals = model.get_residuals_batched(state.good_prompts)
        bad_residuals = model.get_residuals_batched(state.bad_prompts)

        images = []
        pacmap_init = None
        tmpdir = Path(tempfile.mkdtemp())

        for layer_index in range(1, len(model.get_layers()) + 1):
            g_res = good_residuals[:, layer_index, :].detach().cpu().numpy()
            b_res = bad_residuals[:, layer_index, :].detach().cpu().numpy()

            residuals = np.vstack((g_res, b_res))
            embedding = PaCMAP(n_components=2, n_neighbors=30)
            residuals_2d = embedding.fit_transform(residuals, init=pacmap_init)
            pacmap_init = residuals_2d

            n_good = g_res.shape[0]
            g_2d = residuals_2d[:n_good]
            b_2d = residuals_2d[n_good:]

            # Rotate for consistent orientation
            g_anchor = compute_geometric_median(g_2d).median
            b_anchor = compute_geometric_median(b_2d).median
            direction = b_anchor - g_anchor
            angle = -np.arctan2(direction[1], direction[0])
            rot = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ])
            residuals_2d = residuals_2d @ rot.T
            g_2d = residuals_2d[:n_good]
            b_2d = residuals_2d[n_good:]

            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(g_2d[:, 0], g_2d[:, 1], s=10, c="royalblue", alpha=0.5, label="Harmless")
            ax.scatter(b_2d[:, 0], b_2d[:, 1], s=10, c="darkorange", alpha=0.5, label="Harmful")
            ax.set_title(f"Layer {layer_index}")
            ax.legend()
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()

            img_path = tmpdir / f"layer_{layer_index:03}.png"
            fig.savefig(img_path, dpi=100)
            plt.close(fig)
            images.append(str(img_path))

        del good_residuals, bad_residuals
        return images, f"Generated {len(images)} PaCMAP plots."

    compute_geometry_btn.click(
        compute_geometry,
        inputs=[pipeline_state],
        outputs=[geometry_table, geometry_status],
    )

    generate_plots_btn.click(
        generate_plots,
        inputs=[pipeline_state],
        outputs=[pacmap_gallery, plots_status],
    )

    return {
        "geometry_table": geometry_table,
        "pacmap_gallery": pacmap_gallery,
    }
