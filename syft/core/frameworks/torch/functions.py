import torch


class TorchFunctions:
    def add(self, x, y):
        """routes to correct addition."""

    def add_spdz(self, x, y):
        """stuff here."""

    def add_normal(self, x, y):
        """stuff here."""

    def add_fixed_prec(self, x, y):
        """stuff here."""


class SPDZTorch:
    @staticmethod
    def add(x, y):
        result = []
        shares = x.shares

        for i in range(shares):
            res = torch.add(x.shares[i], y.shares[i])
            result.append(res)

        # z = SPDZTensor(result)

        # return z
