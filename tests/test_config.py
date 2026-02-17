import pytest
from pydantic import ValidationError

from clawly.config import (
    DatasetSpecification,
    QuantizationMethod,
    RowNormalization,
    Settings,
)


class TestSettingsValidation:
    def test_valid_with_defaults(self):
        s = Settings(model="test/model", )
        assert s.model == "test/model"

    def test_empty_model_raises(self):
        with pytest.raises(ValidationError):
            Settings(model="", )

    def test_whitespace_model_raises(self):
        with pytest.raises(ValidationError):
            Settings(model="   ", )

    def test_model_stripped(self):
        s = Settings(model="  test/model  ", )
        assert s.model == "test/model"

    def test_n_trials_zero_raises(self):
        with pytest.raises(ValidationError):
            Settings(model="test", n_trials=0, )

    def test_n_trials_negative_raises(self):
        with pytest.raises(ValidationError):
            Settings(model="test", n_trials=-5, )

    def test_n_trials_one_valid(self):
        s = Settings(model="test", n_trials=1, )
        assert s.n_trials == 1

    def test_winsorization_too_high_raises(self):
        with pytest.raises(ValidationError):
            Settings(model="test", winsorization_quantile=1.5, )

    def test_winsorization_negative_raises(self):
        with pytest.raises(ValidationError):
            Settings(model="test", winsorization_quantile=-0.1, )

    def test_winsorization_edge_valid(self):
        s = Settings(model="test", winsorization_quantile=0.0, )
        assert s.winsorization_quantile == 0.0
        s = Settings(model="test", winsorization_quantile=1.0, )
        assert s.winsorization_quantile == 1.0

    def test_category_lora_rank_zero_raises(self):
        with pytest.raises(ValidationError):
            Settings(model="test", category_lora_rank=0, )


class TestSettingsDefaults:
    def test_default_n_trials(self, default_settings):
        assert default_settings.n_trials == 200

    def test_default_batch_size(self, default_settings):
        assert default_settings.batch_size == 0

    def test_default_category_preset_none(self, default_settings):
        assert default_settings.category_preset is None

    def test_default_category_lora_rank(self, default_settings):
        assert default_settings.category_lora_rank == 8


class TestQuantizationMethod:
    def test_none_value(self):
        assert QuantizationMethod.NONE == "none"

    def test_bnb_4bit_value(self):
        assert QuantizationMethod.BNB_4BIT == "bnb_4bit"


class TestRowNormalization:
    def test_values(self):
        assert RowNormalization.NONE == "none"
        assert RowNormalization.PRE == "pre"
        assert RowNormalization.FULL == "full"


class TestDatasetSpecification:
    def test_valid_construction(self):
        ds = DatasetSpecification(dataset="test/ds", split="train", column="text")
        assert ds.dataset == "test/ds"
        assert ds.split == "train"
        assert ds.column == "text"

    def test_default_prefix(self):
        ds = DatasetSpecification(dataset="d", split="s", column="c")
        assert ds.prefix == ""

    def test_default_suffix(self):
        ds = DatasetSpecification(dataset="d", split="s", column="c")
        assert ds.suffix == ""

    def test_default_system_prompt_none(self):
        ds = DatasetSpecification(dataset="d", split="s", column="c")
        assert ds.system_prompt is None
