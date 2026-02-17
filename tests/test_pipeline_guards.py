import pytest

from clawly.config import Settings
from clawly.events import NullCallback
from clawly.pipeline import Pipeline


@pytest.fixture
def pipeline(default_settings):
    return Pipeline(default_settings, callback=NullCallback())


class TestPipelineGuards:
    def test_load_datasets_before_load_model(self, pipeline):
        with pytest.raises(RuntimeError, match="load_model"):
            pipeline.load_datasets()

    def test_compute_directions_before_load_model(self, pipeline):
        with pytest.raises(RuntimeError, match="load_model"):
            pipeline.compute_refusal_directions()

    def test_compute_directions_before_load_datasets(self, pipeline):
        # Simulate model loaded but no datasets
        pipeline.model = object()  # truthy sentinel
        with pytest.raises(RuntimeError, match="load_datasets"):
            pipeline.compute_refusal_directions()

    def test_run_optimization_before_directions(self, pipeline):
        with pytest.raises(RuntimeError, match="compute_refusal_directions"):
            pipeline.run_optimization()

    def test_run_optimization_iter_before_directions(self, pipeline):
        with pytest.raises(RuntimeError, match="compute_refusal_directions"):
            # Generator must be advanced to trigger the check
            next(pipeline.run_optimization_iter())

    def test_constructor_loads_categories_from_preset(self):
        settings = Settings(model="test/model", category_preset="default", )
        p = Pipeline(settings, callback=NullCallback())
        assert p.categories is not None
        assert len(p.categories) == 4

    def test_constructor_no_categories_by_default(self, pipeline):
        assert pipeline.categories is None
