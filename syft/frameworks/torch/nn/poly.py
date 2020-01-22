import sys
import torch as th
import torch.nn as nn


class SimplePolynomial(nn.Module):
    def __init__(self, power=2, scalar=0.1, verbose=False):

        super().__init__()

        self.power = power
        self.scalar = scalar

        self.verbose = verbose

    def forward(self, x):
        shape = x.shape
        if self.verbose:
            sys.stdout.write("Nonlin - x^" + str(self.power) + " x " + str(self.scalar) + " - 0%")
        new_x = list()
        flattened = x.view(-1)
        for val_i in range(flattened.shape[0]):
            val = flattened[val_i : val_i + 1]
            if self.verbose:
                sys.stdout.write(
                    "\rNonlin - x^"
                    + str(self.power)
                    + " x "
                    + str(self.scalar)
                    + " - "
                    + str((val_i / flattened.shape[0]) * 100)[0:4]
                    + "%"
                )
            out = val
            for _ in range(self.power - 1):
                out = out * val
            new_x.append((out * self.scalar).unsqueeze(0))
        if self.verbose:
            sys.stdout.write(
                "\rNonlin - x^" + str(self.power) + " x " + str(self.scalar) + " - 100%"
            )
            print()
        return th.cat(new_x).view(shape)
