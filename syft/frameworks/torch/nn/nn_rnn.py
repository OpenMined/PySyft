import numpy as np
import torch
from torch import nn
from torch.nn import init

from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor
import syft.frameworks.torch.tensors.interpreters.precision
from syft.generic.pointers.pointer_tensor import PointerTensor


class RNNCellBase(nn.Module):
    """
    Cell to be used as base for all RNN cells, including GRU and LSTM
    This class overrides the torch.nn.RNNCellBase
    Only Linear and Dropout layers are used to be able to use MPC
    """

    def __init__(self, input_size, hidden_size, bias, num_chunks, nonlinearity=None, outer=None):
        super(RNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_chunks = num_chunks
        self.nonlinearity = nonlinearity
        self.fc_xh = nn.Linear(input_size, self.num_chunks * hidden_size, bias=bias)
        self.fc_hh = nn.Linear(hidden_size, self.num_chunks * hidden_size, bias=bias)
        self.reset_parameters()
        print("yeah!!")
        print(type(outer))

    def reset_parameters(self):
        """
        This method initializes or reset all the parameters of the cell.
        The paramaters are initiated following a uniform distribution.
        """
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            init.uniform_(w, -std, std)

    def init_hidden(self, input):
        """
        This method initializes a hidden state when no hidden state is provided
        in the forward method. It creates a hidden state with zero values.
        """
        h = torch.zeros(input.shape[0], self.hidden_size, dtype=input.dtype, device=input.device)
        if input.has_child() and isinstance(input.child, PointerTensor):
            h = h.send(input.child.location)
        if input.has_child() and isinstance(
            input.child, syft.frameworks.torch.tensors.interpreters.precision.FixedPrecisionTensor
        ):
            h = h.fix_precision()
            child = input.child
            if isinstance(child.child, AdditiveSharingTensor):
                crypto_provider = child.child.crypto_provider
                owners = child.child.locations
                h = h.share(*owners, crypto_provider=crypto_provider)
        return h
