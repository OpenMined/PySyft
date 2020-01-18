import torch.nn as nn
import sys


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.reset_parameters()

    def forward(self, input):

        _, y = input.shape
        if y != self.in_features:
            sys.exit(
                f"Wrong Input Features. Please use tensor with {self.in_features} Input Features"
            )

        weight = self.weight.t()
        prod = input @ weight
        if self.bias is not None:
            resized_bias = self.bias.unsqueeze(0)
            output = prod + resized_bias.expand(*prod.shape)
        else:
            output = prod
        return output

    def reset_parameters(self):
        dummy_for_init = nn.Linear(self.in_features, self.out_features, bias=self.use_bias)

        self.weight = dummy_for_init.weight
        self.bias = dummy_for_init.bias
