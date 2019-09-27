import torch


class SequenceLabellingHead(torch.nn.Module):
    """A simple head which can be placed out the output of a language model (such as BERT) to
    perform sequence labelling tasks.

    Args:
        in_features (int): The size of the feature dimension of the inputs.
        out_features (int): The size of the feature dimension of the output.

    Shape:
        - x_1: `(N, *, in_features)` where `N` is the batch dimension and `*` means any number of
          additional dimensisons.
        - x_2: `(N, *, in_features)`, where `N` is the batch dimension and `*` means any number of
          additional dimensions.
        - Output: `(N, *, out_features)`, where `N` is the batch dimension and `*` means any number
            of additional dimensions.

    Examples:
        >>> batch_size, in_features, out_features = 32, 100, 4
        >>> head = SequenceLabellingHead(in_features, out_features)
        >>> x = torch.randn(batch_size, in_features)
        >>> output = head(x)
        >>> print(output.size())
        torch.Size([32, 4])
    """
    def __init__(self, in_features, out_features):
        super(SequenceLabellingHead, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, input):
        return self.linear(input)

    def reset_parameters(self):
        self.linear.reset_parameters()
