import torch as th
import syft as sy
import torch.nn as nn

class Dropout2d(nn.Module):

    def __init__(self, p, inplace=False):
        super().__init__()
        self.p = p
        self.training = True
        self.inplace = inplace

    def forward(self, x):

        if (self.training):
            mask = th.rand(x.shape) > self.p
            output = (x * mask) / (1 - self.p)
        else:
            output = x

        if(self.inplace):
            x.set_(output)
            return x
        
        return output

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

