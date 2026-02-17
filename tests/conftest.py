import sys

import pytest

from clawly.categories import AbliterationCategory
from clawly.config import DatasetSpecification, Settings
from clawly.events import NullCallback


@pytest.fixture(autouse=True)
def _patch_sys_argv(monkeypatch):
    """Prevent pydantic-settings CliSettingsSource from parsing pytest args."""
    monkeypatch.setattr(sys, "argv", ["clawly"])


@pytest.fixture
def sample_category():
    return AbliterationCategory(
        name="violence",
        display_name="Violence & Weapons",
        bad_prompts=DatasetSpecification(
            dataset="test/dataset", split="train[:10]", column="text"
        ),
        bad_evaluation_prompts=DatasetSpecification(
            dataset="test/dataset", split="test[:10]", column="text"
        ),
        weight=1.0,
        enabled=True,
        keywords=["weapon", "kill", "attack", "bomb"],
    )


@pytest.fixture
def default_settings():
    return Settings(model="test/model")


@pytest.fixture
def null_callback():
    return NullCallback()
