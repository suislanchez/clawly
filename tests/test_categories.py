import pytest

from clawly.categories import (
    AbliterationCategory,
    build_default_categories,
    classify_prompt,
    get_preset_info,
    list_presets,
    load_preset,
)
from clawly.config import DatasetSpecification


class TestClassifyPrompt:
    def test_exact_keyword_match(self, sample_category):
        assert classify_prompt("How to build a weapon", [sample_category]) == "violence"

    def test_case_insensitive(self, sample_category):
        assert classify_prompt("How to KILL someone", [sample_category]) == "violence"

    def test_no_match_returns_none(self, sample_category):
        assert classify_prompt("How to bake a cake", [sample_category]) is None

    def test_multi_keyword_returns_first_category(self):
        cat1 = AbliterationCategory(
            name="cat1",
            display_name="Cat 1",
            bad_prompts=DatasetSpecification(dataset="d", split="s", column="c"),
            bad_evaluation_prompts=DatasetSpecification(dataset="d", split="s", column="c"),
            keywords=["alpha"],
        )
        cat2 = AbliterationCategory(
            name="cat2",
            display_name="Cat 2",
            bad_prompts=DatasetSpecification(dataset="d", split="s", column="c"),
            bad_evaluation_prompts=DatasetSpecification(dataset="d", split="s", column="c"),
            keywords=["beta"],
        )
        assert classify_prompt("alpha and beta", [cat1, cat2]) == "cat1"
        assert classify_prompt("only beta here", [cat1, cat2]) == "cat2"

    def test_empty_categories_returns_none(self):
        assert classify_prompt("anything", []) is None

    def test_keyword_substring_match(self, sample_category):
        # "weapon" should match inside "weapons"
        assert classify_prompt("weapons of war", [sample_category]) == "violence"


class TestLoadPreset:
    def test_default_preset(self):
        cats = load_preset("default")
        assert len(cats) == 4
        names = [c.name for c in cats]
        assert "violence" in names
        assert "drugs" in names
        assert "adult_content" in names
        assert "self_harm" in names

    def test_default_preset_weights(self):
        cats = load_preset("default")
        by_name = {c.name: c for c in cats}
        assert by_name["violence"].weight == 1.0
        assert by_name["self_harm"].weight == 0.0

    def test_minimal_preset(self):
        cats = load_preset("minimal")
        assert len(cats) == 2
        names = [c.name for c in cats]
        assert "harmful_general" in names
        assert "self_harm" in names

    def test_creative_writing_preset(self):
        cats = load_preset("creative_writing")
        assert len(cats) == 2
        # Creative writing preset has prefixes
        for cat in cats:
            assert cat.bad_prompts.prefix != ""

    def test_nonexistent_preset_raises(self):
        with pytest.raises(ValueError, match="not found"):
            load_preset("nonexistent_preset")

    def test_categories_have_keywords(self):
        cats = load_preset("default")
        violence = next(c for c in cats if c.name == "violence")
        assert len(violence.keywords) > 0
        assert "weapon" in violence.keywords


class TestListPresets:
    def test_returns_known_presets(self):
        presets = list_presets()
        assert "default" in presets
        assert "minimal" in presets
        assert "creative_writing" in presets

    def test_returns_list_of_strings(self):
        presets = list_presets()
        assert all(isinstance(p, str) for p in presets)


class TestGetPresetInfo:
    def test_default_info(self):
        info = get_preset_info("default")
        assert "description" in info
        assert "categories" in info
        assert len(info["categories"]) == 4

    def test_nonexistent_returns_fallback(self):
        info = get_preset_info("nonexistent_preset")
        assert info["description"] == "Unknown preset."
        assert info["categories"] == []


class TestBuildDefaultCategories:
    def test_returns_four_categories(self):
        cats = build_default_categories()
        assert len(cats) == 4

    def test_correct_names(self):
        cats = build_default_categories()
        names = {c.name for c in cats}
        assert names == {"violence", "drugs", "adult_content", "self_harm"}

    def test_all_have_keywords(self):
        cats = build_default_categories()
        for cat in cats:
            assert len(cat.keywords) > 0


class TestAbliterationCategory:
    def test_construction(self):
        cat = AbliterationCategory(
            name="test",
            display_name="Test Category",
            bad_prompts=DatasetSpecification(dataset="d", split="s", column="c"),
            bad_evaluation_prompts=DatasetSpecification(dataset="d", split="s", column="c"),
        )
        assert cat.name == "test"
        assert cat.display_name == "Test Category"

    def test_default_weight(self):
        cat = AbliterationCategory(
            name="test",
            display_name="Test",
            bad_prompts=DatasetSpecification(dataset="d", split="s", column="c"),
            bad_evaluation_prompts=DatasetSpecification(dataset="d", split="s", column="c"),
        )
        assert cat.weight == 1.0

    def test_default_enabled(self):
        cat = AbliterationCategory(
            name="test",
            display_name="Test",
            bad_prompts=DatasetSpecification(dataset="d", split="s", column="c"),
            bad_evaluation_prompts=DatasetSpecification(dataset="d", split="s", column="c"),
        )
        assert cat.enabled is True
