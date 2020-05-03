import numpy as np
import torch
from torch import nn
from torch.nn import init


from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor
from syft.frameworks.torch.tensors.interpreters import precision
from syft.generic.pointers.pointer_tensor import PointerTensor


class RNNCellBase(nn.Module):
    """
    Cell to be used as base for all RNN cells, including GRU and LSTM
    This class overrides the torch.nn.RNNCellBase
    Only Linear and Dropout layers are used to be able to use MPC
    """

    def __init__(self, input_size, hidden_size, bias, num_chunks, nonlinearity=None):
        super(RNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_chunks = num_chunks
        self.nonlinearity = nonlinearity
        self.fc_xh = nn.Linear(input_size, self.num_chunks * hidden_size, bias=bias)
        self.fc_hh = nn.Linear(hidden_size, self.num_chunks * hidden_size, bias=bias)
        self.reset_parameters()

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
        if input.has_child() and isinstance(input.child, precision.FixedPrecisionTensor):
            h = h.fix_precision()
            child = input.child
            if isinstance(child.child, AdditiveSharingTensor):
                crypto_provider = child.child.crypto_provider
                owners = child.child.locations
                h = h.share(*owners, crypto_provider=crypto_provider)
        return h


class RNNCell(RNNCellBase):
    """
    Python implementation of RNNCell with tanh or relu non-linearity for MPC
    This class overrides the torch.nn.RNNCell
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super(RNNCell, self).__init__(input_size, hidden_size, bias, num_chunks=1)

        if nonlinearity == "tanh":
            self.nonlinearity = torch.tanh
        elif nonlinearity == "relu":
            self.nonlinearity = torch.relu
        else:
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

    def forward(self, x, h=None):

        if h is None:
            h = self.init_hidden(x)
        h_ = self.nonlinearity(self.fc_xh(x) + self.fc_hh(h))

        return h_


class GRUCell(RNNCellBase):
    """
    Python implementation of GRUCell for MPC
    This class overrides the torch.nn.GRUCell
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=None):
        super(GRUCell, self).__init__(input_size, hidden_size, bias, num_chunks=3)

    def forward(self, x, h=None):

        if h is None:
            h = self.init_hidden(x)

        gate_x = self.fc_xh(x)
        gate_h = self.fc_hh(h)
        x_r, x_z, x_n = gate_x.chunk(self.num_chunks, 1)
        h_r, h_z, h_n = gate_h.chunk(self.num_chunks, 1)

        resetgate = torch.sigmoid(x_r + h_r)
        updategate = torch.sigmoid(x_z + h_z)
        newgate = torch.tanh(x_n + (resetgate * h_n))

        h_ = newgate + updategate * (h - newgate)

        return h_


class LSTMCell(RNNCellBase):
    """
    Python implementation of LSTMCell for MPC
    This class overrides the torch.nn.LSTMCell
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=None):
        super(LSTMCell, self).__init__(input_size, hidden_size, bias, num_chunks=4)

    def reset_parameters(self):
        super(LSTMCell, self).reset_parameters()

        # Bias of forget gate should be initialize with 1 or 2
        # Ref: http://proceedings.mlr.press/v37/jozefowicz15.pdf
        incr_bias = 1.0 / self.hidden_size
        init.constant_(self.fc_xh.bias[self.hidden_size : 2 * self.hidden_size], incr_bias)
        init.constant_(self.fc_hh.bias[self.hidden_size : 2 * self.hidden_size], incr_bias)

    def forward(self, x, hc=None):

        if hc is None:
            hc = (self.init_hidden(x), self.init_hidden(x))
        h, c = hc

        gate_x = self.fc_xh(x)
        gate_h = self.fc_hh(h)

        x_i, x_f, x_c, x_o = gate_x.chunk(self.num_chunks, 1)
        h_i, h_f, h_c, h_o = gate_h.chunk(self.num_chunks, 1)

        inputgate = torch.sigmoid(x_i + h_i)
        forgetgate = torch.sigmoid(x_f + h_f)
        cellgate = torch.tanh(x_c + h_c)
        outputgate = torch.sigmoid(x_o + h_o)

        c_ = torch.mul(forgetgate, c) + torch.mul(inputgate, cellgate)

        h_ = torch.mul(outputgate, torch.tanh(c_))

        return h_, c_


class RNNBase(nn.Module):
    """
    Module to be used as base for all RNN modules, including GRU and LSTM
    This class overrides the torch.nn.RNNBase
    Only Linear and Dropout layers are used to be able to use MPC
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias,
        batch_first,
        dropout,
        bidirectional,
        base_cell,
        nonlinearity=None,
    ):
        super(RNNBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.is_lstm = base_cell is LSTMCell
        self.nonlinearity = nonlinearity

        # Dropout layers
        # TODO: implement a nn.Dropout class for PySyft
        # Link to issue: https://github.com/OpenMined/PySyft/issues/2500

        # Build RNN layers
        sizes = [input_size, *[hidden_size] * (self.num_layers - 1)]
        self.rnn_forward = nn.ModuleList(
            (base_cell(sz, hidden_size, bias, nonlinearity) for sz in sizes)
        )
        if self.bidirectional:
            self.rnn_backward = nn.ModuleList(
                (base_cell(sz, hidden_size, bias, nonlinearity) for sz in sizes)
            )

    def forward(self, x, hc=None):
        # Treat hc[0] as the hidden state regardless of whether it is an LSTM or GRU
        if not self.is_lstm and hc is not None:
            hc = [hc]

        # If batch_first == True, swap batch with seq_len
        # At the end of the process we swap it back to the original structure
        if self.batch_first:
            x = torch.transpose(x, 0, 1)
            if hc is not None:
                hc = [torch.transpose(item, 0, 1) for item in hc]

        batch_size = x.shape[1]
        seq_len = x.shape[0]

        # Initiate states if needed
        if hc is None:
            hc = [self._init_hidden(x) for i in range(2 if self.is_lstm else 1)]

        # If bidirectional==True, split states in two, one for each direction
        if self.bidirectional:
            hc = [
                item.contiguous().view(self.num_layers, 2, batch_size, self.hidden_size)
                for item in hc
            ]
            hc_fwd = [item[:, 0, :, :] for item in hc]
            hc_back = [item[:, 1, :, :] for item in hc]
        else:
            hc_fwd = hc

        # Run through rnn in the forward direction
        output = x.new(seq_len, batch_size, self.hidden_size).zero_()
        for t in range(seq_len):
            hc_fwd = self._apply_time_step(x, hc_fwd, t)
            output[t, :, :] = hc_fwd[0][-1, :, :]

        # Run through rnn in the backward direction if bidirectional==True
        if self.bidirectional:
            output_back = x.new(seq_len, batch_size, self.hidden_size).zero_()
            for t in range(seq_len - 1, -1, -1):
                hc_back = self._apply_time_step(x, hc_back, t, reverse_direction=True)
                output_back[t, :, :] = hc_back[0][-1, :, :]

            # Concatenate both directions
            output = torch.cat((output, output_back), dim=-1)
            hidden = [
                torch.cat((hid_item, back_item), dim=0)
                for hid_item, back_item in zip(hc_fwd, hc_back)
            ]
        else:
            hidden = hc_fwd

        # If batch_first == True, swap axis back to get original structure
        if self.batch_first:
            output = torch.transpose(output, 0, 1)
            hidden = [torch.transpose(item, 0, 1) for item in hidden]

        # Reshape hidden to the original shape of h
        hidden = tuple(hidden) if self.is_lstm else hidden[0]

        return output, hidden

    def _init_hidden(self, input):
        """
        This method initializes a hidden state when no hidden state is provided
        in the forward method. It creates a hidden state with zero values for each
        layer of the network.
        """
        h = torch.zeros(
            self.num_layers * self.num_directions,
            input.shape[1],
            self.hidden_size,
            dtype=input.dtype,
            device=input.device,
        )
        if input.has_child() and isinstance(input.child, PointerTensor):
            h = h.send(input.child.location)
        if input.has_child() and isinstance(input.child, precision.FixedPrecisionTensor):
            h = h.fix_precision()
            child = input.child
            if isinstance(child.child, AdditiveSharingTensor):
                crypto_provider = child.child.crypto_provider
                owners = child.child.locations
                h = h.share(*owners, crypto_provider=crypto_provider)
        return h

    def _apply_time_step(self, x, h, t, reverse_direction=False):
        """
        Apply RNN layers at time t, given input and previous hidden states
        """
        rnn_layers = self.rnn_backward if reverse_direction else self.rnn_forward

        h = torch.stack([*h])
        h_next = torch.stack([item.new(item.shape).zero_() for item in h])

        for layer in range(self.num_layers):
            inp = x[t, :, :] if layer == 0 else h_next[0][layer - 1, :, :].clone()

            if self.is_lstm:
                h_next[:, layer, :, :] = torch.stack(rnn_layers[layer](inp, h[:, layer, :, :]))
            else:
                h_next[0][layer, :, :] = rnn_layers[layer](inp, h[0][layer, :, :])

        return h_next


class RNN(RNNBase):
    """
    Python implementation of RNN for MPC
    This class overrides the torch.nn.RNN
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        nonlinearity="tanh",
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
    ):

        super(RNN, self).__init__(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            RNNCell,
            nonlinearity,
        )


class GRU(RNNBase):
    """
    Python implementation of GRU for MPC
    This class overrides the torch.nn.GRU
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
    ):

        super(GRU, self).__init__(
            input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, GRUCell
        )


class LSTM(RNNBase):
    """
    Python implementation of LSTM for MPC
    This class overrides the torch.nn.LSTM
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
    ):

        super(LSTM, self).__init__(
            input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, LSTMCell
        )
