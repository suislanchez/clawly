from unittest.mock import MagicMock

from clawly.utils import Prompt, batchify, format_duration, get_trial_parameters


class TestFormatDuration:
    def test_zero(self):
        assert format_duration(0) == "0s"

    def test_seconds_only(self):
        assert format_duration(45) == "45s"

    def test_minutes_and_seconds(self):
        assert format_duration(125) == "2m 5s"

    def test_hours_and_minutes(self):
        assert format_duration(3665) == "1h 1m"

    def test_large_hours(self):
        assert format_duration(86400) == "24h 0m"

    def test_exact_minute(self):
        assert format_duration(60) == "1m 0s"

    def test_exact_hour(self):
        assert format_duration(3600) == "1h 0m"


class TestBatchify:
    def test_even_split(self):
        assert batchify([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_uneven_split(self):
        assert batchify([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

    def test_empty_list(self):
        assert batchify([], 5) == []

    def test_single_item(self):
        assert batchify([1], 10) == [[1]]

    def test_batch_size_one(self):
        assert batchify([1, 2, 3], 1) == [[1], [2], [3]]


class TestPrompt:
    def test_construction(self):
        p = Prompt(system="sys", user="usr")
        assert p.system == "sys"
        assert p.user == "usr"


class TestGetTrialParameters:
    def test_with_mocked_trial(self):
        trial = MagicMock()
        trial.user_attrs = {
            "direction_index": 15.50,
            "parameters": {
                "q_proj": {"max_weight": 1.2, "max_weight_position": 20.0, "min_weight": 0.5, "min_weight_distance": 10.0},
            },
        }
        params = get_trial_parameters(trial)
        assert params["direction_index"] == "15.50"
        assert "q_proj.max_weight" in params
        assert params["q_proj.max_weight"] == "1.20"

    def test_per_layer_direction(self):
        trial = MagicMock()
        trial.user_attrs = {
            "direction_index": None,
            "parameters": {},
        }
        params = get_trial_parameters(trial)
        assert params["direction_index"] == "per layer"
