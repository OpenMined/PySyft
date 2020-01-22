import torch.nn as nn
import sys


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, verbose=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.verbose = verbose

        self.reset_parameters()

    def forward(self, input):

        _, y = input.shape
        if y != self.in_features:
            sys.exit(
                f"Wrong Input Features. Please use tensor with {self.in_features} Input Features"
            )
        if self.verbose:
            sys.stdout.write("Linear - transpose()      ")
        weight = self.weight.t()
        if self.verbose:
            sys.stdout.write("\rLinear - matmul()      ")
        prod = input @ weight
        if self.verbose:
            sys.stdout.write("\rLinear - +bias      ")
        if self.bias is not None:
            resized_bias = self.bias.unsqueeze(0)
            output = prod + resized_bias.expand(*prod.shape)
        else:
            output = prod
        if self.verbose:
            sys.stdout.write("\rLinear - done!         ")
        print()
        return output

    def reset_parameters(self):
        dummy_for_init = nn.Linear(self.in_features, self.out_features, bias=self.use_bias)

        self.weight = dummy_for_init.weight
        self.bias = dummy_for_init.bias

    def __repr__(self):
        return str(self)

    def __str__(self):
        out = "Linear-Handcrafted("
        out += "in_features=" + str(self.in_features) + ", "
        out += "out_features=" + str(self.out_features) + ", "
        out += "bias=" + str(self.bias is not None)
        out += ")"
        return out

    def torchcraft(self):
        """Converts this handcrafted module into a torch.nn.Linear module wherein all the
        module's features are executing in C++. This will increase performance at the cost of
        some of PySyft's more advanced features such as encrypted computation."""

        model = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        model.weight = self.weight
        model.bias = self.bias
        return model


def handcraft(self):
    """Converts a torch.nn.Linear module to a handcrafted one wherein all the
    module's features are executing in python. This is necessary for some of PySyft's
    more advanced features (like encrypted computation)."""

    model = Linear(self.in_features, self.out_features, bias=self.bias is not None)
    model.weight = self.weight
    model.bias = self.bias
    return model


nn.Linear.handcraft = handcraft
