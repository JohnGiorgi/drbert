import pytest

from ..modules.proportional_batch_sampler import ProportionalBatchSampler


class TestProportionalBatchSampler(object):
    """Collects all unit tests for
    `drbert.modules.proportional_batch_sampler.ProportionalBatchSampler`.
    """
    def test_value_error_invalid_tasks(self, tasks):
        with pytest.raises(ValueError):
            ProportionalBatchSampler([])
        with pytest.raises(ValueError):
            # Empty one of the dictionaries to make sure error is thrown
            tasks[0] = {}
            ProportionalBatchSampler(tasks)

    def test_value_error_invalid_partition(self, tasks):
        with pytest.raises(ValueError):
            ProportionalBatchSampler(tasks, partition='not valid')
