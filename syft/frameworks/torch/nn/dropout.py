import torch as th
import torch.nn as nn


class Dropout2d(nn.Module):
    def __init__(self, p, inplace=False):
        super().__init__()
        self.p = p
        self.training = True
        self.inplace = inplace

    def forward(self, x):

        if self.training:
            mask = th.rand(x.shape) > self.p
            output = (x * mask) / (1 - self.p)
        else:
            output = x

        if self.inplace:
            x.set_(output)
            return x

        return output

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __repr__(self):
        return str(self)

    def __str__(self):
        out = "Dropout2d-Handcrafted("
        out += "p=" + str(self.p) + ", "
        out += "inplace=" + str(self.inplace)
        out += ")"
        return out

    def torchcraft(self):
        """Converts this handcrafted module into a torch.nn.Dropout2d module wherein all the
        module's features are executing in C++. This will increase performance at the cost of
        some of PySyft's more advanced features such as encrypted computation."""

        model = th.nn.Dropout2d(self.p, self.inplace)
        model.training = self.training
        return model


def handcraft(self):
    """Converts a torch.nn.Dropout2ds module to a handcrafted one wherein all the
    module's features are executing in python. This is necessary for some of PySyft's
    more advanced features (like encrypted computation)."""

    model = Dropout2d(self.p, self.inplace)
    model.training = self.training
    return model


th.nn.Dropout2d.handcraft = handcraft
