import pytest

from ..eval import evaluate


class TestEval(object):
    """Collects all unit tests for `drbert.eval`.
    """
    def test_evaluate_value_error(self, bert):
        with pytest.raises(ValueError):
            evaluate(args=None, model=None, dataset=None, task='not valid')
