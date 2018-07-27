import torch

class TorchFunctions(object):

    def add(self, x, y):

        "routes to correct addition"


    def add_mpc(self, x, y):
        "stuff here"

    def add_normal(self, x, y):
        "stuff here"

    def add_fixed_prec(self, x, y):
        "stuff here"

class MPCTorch(object):

    @staticmethod
    def add(x, y):

        result = []
        shares = x.shares

        for i in range(x.shares):
            res = torch.add(x.shares[i], y.shares[i])
            result.append(res)

        #z = MPCTensor(result)

        #return z
