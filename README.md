# Clawly: Automatic LLM abliteration with selective uncensoring

**Clawly** is a fork of [Heretic](https://github.com/p-e-w/heretic) that adds two major features:
a **Gradio Web UI** and **partial/composable abliteration** (category-based selective uncensoring).

It combines an advanced implementation of directional ablation ("abliteration")
with a TPE-based parameter optimizer powered by [Optuna](https://optuna.org/),
and wraps it all in a reusable Pipeline API that can be driven from the CLI,
a web browser, or a Jupyter notebook.

## What's new in Clawly

### Category-Based Selective Abliteration

Instead of removing all refusal behavior, Clawly lets you choose *which categories*
to uncensor and which to leave intact. For example, you can abliterate violence
and drug-related refusals while preserving self-harm refusal behavior:

```toml
# config.toml
category_preset = "default"
orthogonalize_categories = true
```

Each category has a weight from 0.0 (preserve refusal) to 1.0 (full abliteration).
The built-in `default` preset includes:

| Category | Default Weight | Description |
| :--- | :---: | :--- |
| Violence & Weapons | 1.0 | Full abliteration |
| Drugs & Substances | 1.0 | Full abliteration |
| Adult Content | 1.0 | Full abliteration |
| Self-Harm & Suicide | 0.0 | Preserved |

Under the hood, Clawly computes separate refusal directions per category,
optionally orthogonalizes them via Gram-Schmidt, and accumulates weighted
delta-W matrices that are SVD-decomposed into LoRA adapters.

### Gradio Web UI

Launch the web interface with:

```
clawly-web
```

The UI provides 5 tabs:

- **Configuration** -- Set model, quantization, trials, category weights, and advanced settings
- **Run** -- Live progress bar, Pareto front scatter plot, trial log
- **Results** -- Interactive Pareto chart, trial parameter inspection, per-category breakdown
- **Export** -- Save locally, upload to Hugging Face, or chat with the model
- **Research** -- Residual geometry tables and PaCMAP visualizations

### Pipeline API

Clawly exposes a `Pipeline` class for programmatic use in notebooks or scripts:

```python
from clawly import Pipeline, Settings

settings = Settings(model="Qwen/Qwen3-4B-Instruct-2507", n_trials=50)
pipeline = Pipeline(settings)

pipeline.load_model()
pipeline.load_datasets()
pipeline.auto_batch_size()
pipeline.detect_response_prefix()
pipeline.init_evaluator()
pipeline.compute_refusal_directions()

# Stream results
for result in pipeline.run_optimization_iter():
    print(f"Trial {result.index}: {result.refusals} refusals, KL={result.kl_divergence:.4f}")

# Export best trial
best = pipeline.get_pareto_front()[0]
pipeline.restore_trial(best)
pipeline.save_model("./output")
```


## Installation

```
pip install -U clawly
```

For the web UI:
```
pip install -U clawly[web]
```

For research features (PaCMAP plots, residual geometry):
```
pip install -U clawly[research]
```


## CLI Usage

```
clawly Qwen/Qwen3-4B-Instruct-2507
```

Replace with whatever model you want to decensor. The process is fully automatic.

Run `clawly --help` to see all options, or use `config.toml` for file-based configuration.
See [`config.default.toml`](config.default.toml) for the full list of parameters.

### Category mode from CLI

```
clawly --category-preset default --orthogonalize-categories Qwen/Qwen3-4B-Instruct-2507
```


## How it works

Clawly implements a parametrized variant of directional ablation. For each
supported transformer component (attention out-projection and MLP down-projection),
it identifies the associated matrices in each transformer layer and orthogonalizes
them with respect to the relevant "refusal direction", inhibiting the expression
of that direction.

Refusal directions are computed for each layer as a difference-of-means between
the first-token residuals for "harmful" and "harmless" example prompts.

The ablation process is controlled by several optimizable parameters:

* `direction_index`: Either the index of a refusal direction, or `per layer`
* `max_weight`, `max_weight_position`, `min_weight`, `min_weight_distance`:
  Shape and position of the ablation weight kernel over layers

In **category mode**, separate refusal directions are computed per category
(each against the same good-prompt means), optionally orthogonalized, and then
combined with per-category weights into a single LoRA adapter via SVD decomposition.


## Research features

Install with `pip install -U clawly[research]` to access:

- `--plot-residuals`: PaCMAP projections of residual vectors per layer
- `--print-residual-geometry`: Quantitative analysis table with cosine similarities,
  norms, and silhouette coefficients


## Based on Heretic

Clawly is built on [Heretic](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann.
The core abliteration engine, Optuna integration, and model handling are from the original project.

### Prior art

* [AutoAbliteration](https://huggingface.co/posts/mlabonne/714992455492422)
* [abliterator.py](https://github.com/FailSpy/abliterator)
* [wassname's Abliterator](https://github.com/wassname/abliterator)
* [ErisForge](https://github.com/Tsadoq/ErisForge)
* [deccp](https://github.com/AUGMXNT/deccp)

### Acknowledgments

* [Original abliteration paper (Arditi et al. 2024)](https://arxiv.org/abs/2406.11717)
* [Maxime Labonne's article on abliteration](https://huggingface.co/blog/mlabonne/abliteration)
* Jim Lai's articles on [projected abliteration](https://huggingface.co/blog/grimjim/projected-abliteration)
  and [norm-preserving biprojected abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)


## License

Copyright &copy; 2025-2026  Philipp Emanuel Weidmann (<pew@worldwidemann.com>) + contributors

GNU Affero General Public License v3 or later (AGPLv3+).
See [LICENSE](LICENSE) for details.
