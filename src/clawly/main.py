# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""CLI entry point for Clawly.

Provides the ``clawly`` command: a thin wrapper around :class:`Pipeline`
with interactive menus for model export, Hugging Face upload, and chat.
"""

import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import huggingface_hub
import torch
from optuna.trial import TrialState as _TrialState
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
)
from pydantic import ValidationError
from questionary import Choice
from rich.traceback import install

from .config import QuantizationMethod, Settings
from .events import TrialResult
from .model import get_model_class
from .pipeline import Pipeline
from .utils import (
    get_trial_parameters,
    print,
    prompt_password,
    prompt_path,
    prompt_select,
    prompt_text,
)


class CLIProgressCallback:
    """CLI-specific callback that prints progress to the console."""

    def on_state_change(self, state: str, message: str) -> None:
        pass  # Pipeline methods already print to console

    def on_trial_complete(self, result: TrialResult) -> None:
        pass  # Pipeline already prints trial results

    def on_log(self, message: str) -> None:
        print(message)

    def on_progress(self, current: int, total: int, desc: str) -> None:
        pass  # Pipeline already prints progress


def obtain_merge_strategy(settings: Settings) -> str | None:
    """
    Prompts the user for how to proceed with saving the model.
    Provides info to the user if the model is quantized on memory use.
    Returns "merge", "adapter", or None (if cancelled/invalid).
    """

    if settings.quantization == QuantizationMethod.BNB_4BIT:
        print()
        print(
            "Model was loaded with quantization. Merging requires reloading the base model."
        )
        print(
            "[yellow]WARNING: CPU merging requires dequantizing the entire model to system RAM.[/]"
        )
        print("[yellow]This can lead to system freezes if you run out of memory.[/]")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meta_model = get_model_class(settings.model).from_pretrained(
                    settings.model,
                    device_map="meta",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                footprint_bytes = meta_model.get_memory_footprint()
                footprint_gb = footprint_bytes / (1024**3)
                print(
                    f"[yellow]Estimated RAM required (excluding overhead): [bold]~{footprint_gb:.2f} GB[/][/]"
                )
        except Exception:
            print(
                "[yellow]Rule of thumb: You need approximately 3x the parameter count in GB RAM.[/]"
            )
            print(
                "[yellow]Example: A 27B model requires ~80GB RAM. A 70B model requires ~200GB RAM.[/]"
            )
        print()

        strategy = prompt_select(
            "How do you want to proceed?",
            choices=[
                Choice(
                    title="Merge LoRA into full model"
                    + (
                        ""
                        if settings.quantization == QuantizationMethod.NONE
                        else " (requires sufficient RAM)"
                    ),
                    value="merge",
                ),
                Choice(
                    title="Cancel",
                    value="cancel",
                ),
            ],
        )

        if strategy == "cancel":
            return None

        return strategy
    else:
        return "merge"


def _print_device_info():
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"Detected [bold]{count}[/] CUDA device(s):")
        for i in range(count):
            print(f"* GPU {i}: [bold]{torch.cuda.get_device_name(i)}[/]")
    elif is_xpu_available():
        count = torch.xpu.device_count()
        print(f"Detected [bold]{count}[/] XPU device(s):")
        for i in range(count):
            print(f"* XPU {i}: [bold]{torch.xpu.get_device_name(i)}[/]")
    elif is_mlu_available():
        count = torch.mlu.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] MLU device(s):")
        for i in range(count):
            print(f"* MLU {i}: [bold]{torch.mlu.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_sdaa_available():
        count = torch.sdaa.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] SDAA device(s):")
        for i in range(count):
            print(f"* SDAA {i}: [bold]{torch.sdaa.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_musa_available():
        count = torch.musa.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] MUSA device(s):")
        for i in range(count):
            print(f"* MUSA {i}: [bold]{torch.musa.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_npu_available():
        print(f"NPU detected (CANN version: [bold]{torch.version.cann}[/])")  # ty:ignore[unresolved-attribute]
    elif torch.backends.mps.is_available():
        print("Detected [bold]1[/] MPS device (Apple Metal)")
    else:
        print(
            "[bold yellow]No GPU or other accelerator detected. Operations will be slow.[/]"
        )


def run():
    Pipeline.configure_environment()

    # Modified "Pagga" font from https://budavariam.github.io/asciiart-text/
    print(f"[cyan]█▀▀░█░░░█▀█░█░█░█░░░█░█[/]  v{version('clawly')}")
    print("[cyan]█░░░█░░░█▀█░█▄█░█░░░░█░[/]")
    print(
        "[cyan]▀▀▀░▀▀▀░▀░▀░▀░▀░▀▀▀░░▀░[/]  [blue underline]https://github.com/suislanchez/clawly[/]"
    )
    print()

    if (
        len(sys.argv) > 1
        and "--model" not in sys.argv
        and not sys.argv[-1].startswith("-")
    ):
        sys.argv.insert(-1, "--model")

    try:
        settings = Settings()  # ty:ignore[missing-argument]
    except ValidationError as error:
        print(f"[red]Configuration contains [bold]{error.error_count()}[/] errors:[/]")

        for error in error.errors():
            print(f"[bold]{error['loc'][0]}[/]: [yellow]{error['msg']}[/]")

        print()
        print(
            "Run [bold]clawly --help[/] or see [bold]config.default.toml[/] for details about configuration parameters."
        )
        return

    _print_device_info()

    pipeline = Pipeline(settings, CLIProgressCallback())

    # Set up study storage and handle resume
    storage = pipeline.setup_study()
    existing_study = pipeline.get_existing_study()

    if existing_study is not None and settings.evaluate_model is None:
        choices = []

        if existing_study.user_attrs["finished"]:
            print()
            print(
                (
                    "[green]You have already processed this model.[/] "
                    "You can show the results from the previous run, allowing you to export models or to run additional trials. "
                    "Alternatively, you can ignore the previous run and start from scratch. "
                    "This will delete the checkpoint file and all results from the previous run."
                )
            )
            choices.append(
                Choice(
                    title="Show the results from the previous run",
                    value="continue",
                )
            )
        else:
            print()
            print(
                (
                    "[yellow]You have already processed this model, but the run was interrupted.[/] "
                    "You can continue the previous run from where it stopped. This will override any specified settings. "
                    "Alternatively, you can ignore the previous run and start from scratch. "
                    "This will delete the checkpoint file and all results from the previous run."
                )
            )
            choices.append(
                Choice(
                    title="Continue the previous run",
                    value="continue",
                )
            )

        choices.append(
            Choice(
                title="Ignore the previous run and start from scratch",
                value="restart",
            )
        )

        choices.append(
            Choice(
                title="Exit program",
                value="",
            )
        )

        print()
        choice = prompt_select("How would you like to proceed?", choices)

        if choice == "continue":
            settings = Settings.model_validate_json(
                existing_study.user_attrs["settings"]
            )
            pipeline.settings = settings
        elif choice == "restart":
            pipeline.restart_study()
        elif choice is None or choice == "":
            return

    # Core pipeline steps
    pipeline.load_model()
    pipeline.load_datasets()
    pipeline.auto_batch_size()
    pipeline.detect_response_prefix()
    pipeline.init_evaluator()

    if settings.evaluate_model is not None:
        pipeline.evaluate_external_model()
        return

    pipeline.compute_refusal_directions()
    pipeline.run_optimization()

    # Post-optimization interactive menu
    while True:
        completed_trials = [
            t
            for t in pipeline.study.trials
            if t.state == _TrialState.COMPLETE
        ]
        if not completed_trials:
            raise KeyboardInterrupt

        best_trials = pipeline.get_pareto_front()

        choices = [
            Choice(
                title=(
                    f"[Trial {trial.user_attrs['index']:>3}] "
                    f"Refusals: {trial.user_attrs['refusals']:>2}/{len(pipeline.evaluator.bad_prompts)}, "
                    f"KL divergence: {trial.user_attrs['kl_divergence']:.4f}"
                ),
                value=trial,
            )
            for trial in best_trials
        ]

        choices.append(
            Choice(
                title="Run additional trials",
                value="continue",
            )
        )

        choices.append(
            Choice(
                title="Exit program",
                value="",
            )
        )

        print()
        print("[bold green]Optimization finished![/]")
        print()
        print(
            (
                "The following trials resulted in Pareto optimal combinations of refusals and KL divergence. "
                "After selecting a trial, you will be able to save the model, upload it to Hugging Face, "
                "or chat with it to test how well it works. You can return to this menu later to select a different trial. "
                "[yellow]Note that KL divergence values above 1 usually indicate significant damage to the original model's capabilities.[/]"
            )
        )

        while True:
            print()
            trial = prompt_select("Which trial do you want to use?", choices)

            if trial == "continue":
                while True:
                    try:
                        n_additional_trials = prompt_text(
                            "How many additional trials do you want to run?"
                        )
                        if n_additional_trials is None or n_additional_trials == "":
                            n_additional_trials = 0
                            break
                        n_additional_trials = int(n_additional_trials)
                        if n_additional_trials > 0:
                            break
                        print("[red]Please enter a number greater than 0.[/]")
                    except ValueError:
                        print("[red]Please enter a number.[/]")

                if n_additional_trials == 0:
                    continue

                pipeline.run_additional_trials(n_additional_trials)
                break

            elif trial is None or trial == "":
                return

            pipeline.restore_trial(trial)

            while True:
                print()
                action = prompt_select(
                    "What do you want to do with the decensored model?",
                    [
                        "Save the model to a local folder",
                        "Upload the model to Hugging Face",
                        "Chat with the model",
                        "Return to the trial selection menu",
                    ],
                )

                if action is None or action == "Return to the trial selection menu":
                    break

                try:
                    match action:
                        case "Save the model to a local folder":
                            save_directory = prompt_path("Path to the folder:")
                            if not save_directory:
                                continue

                            strategy = obtain_merge_strategy(settings)
                            if strategy is None:
                                continue

                            pipeline.save_model(save_directory, strategy)

                        case "Upload the model to Hugging Face":
                            token = huggingface_hub.get_token()
                            if not token:
                                token = prompt_password("Hugging Face access token:")
                            if not token:
                                continue

                            user = huggingface_hub.whoami(token)
                            fullname = user.get(
                                "fullname",
                                user.get("name", "unknown user"),
                            )
                            email = user.get("email", "no email found")
                            print(f"Logged in as [bold]{fullname} ({email})[/]")

                            repo_id = prompt_text(
                                "Name of repository:",
                                default=f"{user['name']}/{Path(settings.model).name}-clawly",
                            )

                            visibility = prompt_select(
                                "Should the repository be public or private?",
                                [
                                    "Public",
                                    "Private",
                                ],
                            )
                            private = visibility == "Private"

                            strategy = obtain_merge_strategy(settings)
                            if strategy is None:
                                continue

                            pipeline.upload_model(
                                repo_id, token, private, strategy, trial
                            )

                        case "Chat with the model":
                            print()
                            print(
                                "[cyan]Press Ctrl+C at any time to return to the menu.[/]"
                            )

                            chat = [
                                {
                                    "role": "system",
                                    "content": settings.system_prompt,
                                },
                            ]

                            while True:
                                try:
                                    message = prompt_text(
                                        "User:",
                                        qmark=">",
                                        unsafe=True,
                                    )
                                    if not message:
                                        break
                                    chat.append(
                                        {"role": "user", "content": message}
                                    )

                                    print("[bold]Assistant:[/] ", end="")
                                    response = pipeline.model.stream_chat_response(
                                        chat
                                    )
                                    chat.append(
                                        {
                                            "role": "assistant",
                                            "content": response,
                                        }
                                    )
                                except (KeyboardInterrupt, EOFError):
                                    break

                except Exception as error:
                    print(f"[red]Error: {error}[/]")


def main():
    install()

    try:
        run()
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt) or isinstance(
            error.__context__, KeyboardInterrupt
        ):
            print()
            print("[red]Shutting down...[/]")
        else:
            raise
