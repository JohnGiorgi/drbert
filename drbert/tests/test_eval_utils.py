from ..utils import eval_utils


class TestEvalUtils(object):
    """Collects all unit tests for `drbert.utils.eval_utils`.
    """
    def test_reverse_dict_empty(self):
        test = {}
        expected = {}
        actual = eval_utils.reverse_dict(test)

        assert expected == actual

    def reverse_dict_simple(self):
        test = {'a': 1, 'b': 2, 'c': 3}

        expected = {1: 'a', 2: 'b', 3: 'c'}
        actual = eval_utils.reverse_dict(test)

        assert expected == actual
