from clawly.events import NullCallback, ProgressCallback, TrialResult


class TestTrialResult:
    def test_construction(self):
        r = TrialResult(
            index=1,
            score=(0.5, 0.3),
            kl_divergence=0.01,
            refusals=5,
            parameters={"weight": "1.00"},
        )
        assert r.index == 1
        assert r.score == (0.5, 0.3)
        assert r.kl_divergence == 0.01
        assert r.refusals == 5
        assert r.parameters == {"weight": "1.00"}

    def test_default_category_refusals_none(self):
        r = TrialResult(
            index=1,
            score=(0.0, 0.0),
            kl_divergence=0.0,
            refusals=0,
            parameters={},
        )
        assert r.category_refusals is None

    def test_with_category_refusals(self):
        r = TrialResult(
            index=1,
            score=(0.0, 0.0),
            kl_divergence=0.0,
            refusals=3,
            parameters={},
            category_refusals={"violence": 2, "drugs": 1},
        )
        assert r.category_refusals == {"violence": 2, "drugs": 1}


class TestNullCallback:
    def test_on_state_change_no_raise(self, null_callback):
        null_callback.on_state_change("test", "message")

    def test_on_trial_complete_no_raise(self, null_callback):
        result = TrialResult(
            index=1, score=(0.0, 0.0), kl_divergence=0.0, refusals=0, parameters={}
        )
        null_callback.on_trial_complete(result)

    def test_on_log_no_raise(self, null_callback):
        null_callback.on_log("test message")

    def test_on_progress_no_raise(self, null_callback):
        null_callback.on_progress(1, 10, "testing")

    def test_satisfies_protocol(self, null_callback):
        # Verify NullCallback implements all ProgressCallback methods
        assert callable(getattr(null_callback, "on_state_change", None))
        assert callable(getattr(null_callback, "on_trial_complete", None))
        assert callable(getattr(null_callback, "on_log", None))
        assert callable(getattr(null_callback, "on_progress", None))
