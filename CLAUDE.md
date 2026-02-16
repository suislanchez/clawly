# Clawly — Project Guide

## Overview
Clawly is a fork of [Heretic](https://github.com/p-e-w/heretic) that adds category-based selective abliteration and a Gradio Web UI. It removes censorship from LLMs by computing "refusal directions" and orthogonalizing model weights against them, with per-category control.

## Project Structure
```
src/clawly/
├── __init__.py         # Public API exports (Pipeline, Settings, etc.)
├── config.py           # Pydantic Settings (CLI args, env vars, TOML)
├── pipeline.py         # Stateful Pipeline class — core orchestration
├── model.py            # Model loading, abliteration, LoRA application
├── evaluator.py        # Refusal counting, per-category tracking
├── categories.py       # AbliterationCategory, preset loading, classification
├── events.py           # ProgressCallback protocol, TrialResult dataclass
├── analyzer.py         # Residual geometry analysis (research feature)
├── utils.py            # Shared utilities, print helpers
├── main.py             # CLI entry point (thin wrapper around Pipeline)
├── presets/            # Built-in category preset JSON files
│   ├── __init__.py
│   ├── default.json
│   ├── minimal.json
│   └── creative_writing.json
└── web/                # Gradio Web UI
    ├── __init__.py
    ├── app.py          # Main app, tabs, CSS
    └── pages/
        ├── config_page.py
        ├── run_page.py
        ├── results_page.py
        ├── export_page.py
        ├── compare_page.py
        └── research_page.py
```

## Key Patterns

- **Settings**: All configuration flows through `config.py:Settings` (Pydantic BaseSettings). CLI args are auto-generated via `CliSettingsSource` with kebab-case. Env vars use `CLAWLY_` prefix. TOML via `config.toml`.

- **Pipeline**: `pipeline.py:Pipeline` holds all state (model, evaluator, study, directions). Both CLI (`main.py`) and Web UI (`web/`) drive the same Pipeline. Methods are called sequentially: `load_model()` → `load_datasets()` → `auto_batch_size()` → `detect_response_prefix()` → `init_evaluator()` → `compute_refusal_directions()` → `run_optimization()`.

- **Categories**: When `category_preset` is set, `pipeline.py` computes per-category refusal directions instead of a single global one. The evaluator tracks per-category refusals. `model.py:abliterate()` has a multi-category path that accumulates weighted delta-W matrices and SVD-decomposes them into LoRA adapters.

- **Events**: The `ProgressCallback` protocol decouples UI from logic. CLI uses `CLIProgressCallback`, Web UI reads from generators (`run_optimization_iter()`), headless uses `NullCallback`.

## Entry Points
- `clawly` → `clawly.main:main` (CLI)
- `clawly-web` → `clawly.web.app:main` (Gradio UI)

## Dependencies
- Core: torch, transformers, accelerate, peft, optuna, pydantic-settings, datasets, rich, questionary
- Web: gradio, plotly (`pip install clawly[web]`)
- Research: pacmap, geom-median, scikit-learn (`pip install clawly[research]`)

## Testing
No test suite yet. Verify manually:
```bash
clawly --help                    # CLI args work
clawly-web                       # Web UI launches
python -c "from clawly import Pipeline, Settings"  # Imports work
```
