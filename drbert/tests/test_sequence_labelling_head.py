import torch
from torch import nn

from ..modules.sequence_labelling_head import SequenceLabellingHead


class TestSequenceLabellingHead(object):
    """Collects all unit tests for `drbert.modules.SequenceLabellingHead`.
    """
    def test_attributes_after_initialization(self, sequence_labelling_head):
        batch_size, in_features, out_features, head = sequence_labelling_head

        assert head.in_features == in_features
        assert head.out_features == out_features
        assert isinstance(head.linear, (nn.Module, nn.Linear))

    def test_forward_pass(self, sequence_labelling_head):
        batch_size, in_features, out_features, head = sequence_labelling_head

        x = torch.randn(batch_size, in_features)
        output = head(x)

        assert output.size() == (batch_size, out_features)
