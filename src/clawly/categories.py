# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Category-based selective abliteration system.

Provides :class:`AbliterationCategory` for per-category refusal direction
control, keyword-based prompt classification, and preset loading from
bundled JSON files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

from .config import DatasetSpecification


@dataclass
class AbliterationCategory:
    """A single abliteration category with its own refusal direction.

    Each category has dedicated harmful prompts, evaluation prompts, and a weight
    controlling how aggressively its refusal behavior is removed.

    Attributes:
        name: Machine-readable identifier (e.g. ``"violence"``).
        display_name: Human-readable label (e.g. ``"Violence & Weapons"``).
        bad_prompts: Dataset specification for harmful prompts in this category.
        bad_evaluation_prompts: Dataset specification for evaluation prompts.
        weight: Abliteration strength from 0.0 (preserve refusal) to 1.0 (full removal).
        enabled: Whether this category is active in the current run.
        keywords: Keywords used to classify prompts into this category.
    """

    name: str
    display_name: str
    bad_prompts: DatasetSpecification
    bad_evaluation_prompts: DatasetSpecification
    weight: float = 1.0
    enabled: bool = True
    keywords: list[str] = field(default_factory=list)


# Keyword mappings for auto-classification of mlabonne/harmful_behaviors rows.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "violence": [
        "weapon", "kill", "attack", "bomb", "gun", "shoot", "stab",
        "murder", "assault", "fight", "violence", "violent", "explode",
        "explosive", "firearm", "ammunition", "knife", "sword", "combat",
        "harm", "hurt", "injure", "destroy", "war", "terrorist", "terrorism",
    ],
    "drugs": [
        "drug", "substance", "narcotic", "cocaine", "heroin", "meth",
        "marijuana", "cannabis", "opiate", "opioid", "fentanyl", "lsd",
        "mdma", "ecstasy", "amphetamine", "overdose", "addiction",
        "prescription", "pharmaceutical", "controlled substance",
    ],
    "adult_content": [
        "sex", "sexual", "nude", "naked", "porn", "erotic", "explicit",
        "intimate", "adult content", "nsfw", "fetish", "obscene",
    ],
    "self_harm": [
        "suicide", "self-harm", "self harm", "cutting", "overdose",
        "kill myself", "end my life", "hurt myself", "suicidal",
        "eating disorder", "anorexia", "bulimia", "starve",
    ],
}

# Display names for built-in categories.
CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "violence": "Violence & Weapons",
    "drugs": "Drugs & Controlled Substances",
    "adult_content": "Adult & Explicit Content",
    "self_harm": "Self-Harm & Suicide",
}


def get_preset_path(preset_name: str) -> Path:
    """Get path to a built-in preset JSON file."""
    return Path(str(files("clawly.presets"))) / f"{preset_name}.json"


def load_preset(preset_name: str) -> list[AbliterationCategory]:
    """Load a built-in category preset by name.

    Args:
        preset_name: Name of the preset (e.g. ``"default"``, ``"minimal"``).

    Returns:
        List of :class:`AbliterationCategory` instances configured by the preset.

    Raises:
        ValueError: If the preset name is not found.
    """
    preset_path = get_preset_path(preset_name)
    if not preset_path.exists():
        available = list_presets()
        raise ValueError(
            f"Preset '{preset_name}' not found. Available presets: {available}"
        )

    with open(preset_path) as f:
        data = json.load(f)

    categories = []
    for cat_data in data["categories"]:
        categories.append(
            AbliterationCategory(
                name=cat_data["name"],
                display_name=cat_data.get(
                    "display_name",
                    CATEGORY_DISPLAY_NAMES.get(cat_data["name"], cat_data["name"]),
                ),
                bad_prompts=DatasetSpecification(**cat_data["bad_prompts"]),
                bad_evaluation_prompts=DatasetSpecification(
                    **cat_data["bad_evaluation_prompts"]
                ),
                weight=cat_data.get("weight", 1.0),
                enabled=cat_data.get("enabled", True),
                keywords=cat_data.get(
                    "keywords",
                    CATEGORY_KEYWORDS.get(cat_data["name"], []),
                ),
            )
        )

    return categories


def list_presets() -> list[str]:
    """Return the names of all available built-in category presets."""
    presets_dir = Path(str(files("clawly.presets")))
    if not presets_dir.exists():
        return []
    return [p.stem for p in presets_dir.glob("*.json")]


def get_preset_info(preset_name: str) -> dict:
    """Return metadata about a preset: description and category summary.

    Returns:
        Dict with keys ``"description"`` and ``"categories"`` (list of
        ``{"name", "display_name", "weight"}`` dicts).
    """
    preset_path = get_preset_path(preset_name)
    if not preset_path.exists():
        return {"description": "Unknown preset.", "categories": []}

    with open(preset_path) as f:
        data = json.load(f)

    cats = []
    for cat_data in data.get("categories", []):
        cats.append({
            "name": cat_data["name"],
            "display_name": cat_data.get("display_name", cat_data["name"]),
            "weight": cat_data.get("weight", 1.0),
        })

    return {
        "description": data.get("description", "No description."),
        "categories": cats,
    }


def classify_prompt(text: str, categories: list[AbliterationCategory]) -> str | None:
    """Classify a prompt into a category by keyword matching. Returns category name or None."""
    text_lower = text.lower()
    for category in categories:
        for keyword in category.keywords:
            if keyword.lower() in text_lower:
                return category.name
    return None


def build_default_categories() -> list[AbliterationCategory]:
    """Build the default set of categories using keyword-based splitting."""
    categories = []
    for name, keywords in CATEGORY_KEYWORDS.items():
        categories.append(
            AbliterationCategory(
                name=name,
                display_name=CATEGORY_DISPLAY_NAMES.get(name, name),
                bad_prompts=DatasetSpecification(
                    dataset="mlabonne/harmful_behaviors",
                    split="train[:400]",
                    column="text",
                ),
                bad_evaluation_prompts=DatasetSpecification(
                    dataset="mlabonne/harmful_behaviors",
                    split="test[:100]",
                    column="text",
                ),
                weight=1.0,
                enabled=True,
                keywords=keywords,
            )
        )
    return categories
