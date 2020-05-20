import torch as th
import torch.nn as nn
from torch.nn.parameter import Parameter


class BatchNorm2d(nn.Module):
    """TBD"""

    def __init__(self, num_features, eps=1e-5):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.weight = Parameter(th.ones(num_features))
        self.bias = Parameter(th.zeros(num_features))

    def forward(self, input):
        self._check_input_dim(input)
        return nn.functional.batch_norm(
            input, None, None, self.weight, self.bias, None, None, self.eps
        )

    def _check_input_dim(self, input):
        if len(input.shape) != 4:
            raise ValueError(f"expected 4D input (got {len(input.shape)}D input)")
