import numpy as np
import torch
from torch import nn
from torch.nn import init

import syft
from syft.frameworks.torch.tensors.interpreters.precision import FixedPrecisionTensor
from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor


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
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            init.uniform_(w, -std, std)


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
            raise RuntimeError("Unknown nonlinearity: {}".format(nonlinearity))

    def forward(self, x, h=None):

        if h is None:
            h = init_hidden(x, self.hidden_size)
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
            h = init_hidden(x, self.hidden_size)

        gate_x = self.fc_xh(x)
        gate_h = self.fc_hh(h)
        x_r, x_z, x_n = gate_x.chunk(3, 1)
        h_r, h_z, h_n = gate_h.chunk(3, 1)

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
            hc = (init_hidden(x, self.hidden_size), init_hidden(x, self.hidden_size))
        h, c = hc

        gate_x = self.fc_xh(x)
        gate_h = self.fc_hh(h)

        x_i, x_f, x_c, x_o = gate_x.chunk(4, 1)
        h_i, h_f, h_c, h_o = gate_h.chunk(4, 1)

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
        # TO DO: implement a nn.Dropout class for PySyft

        # Build RNN layers
        self.rnn_forward = nn.ModuleList()
        for layer in range(self.num_layers):
            if layer == 0:
                self.rnn_forward.append(base_cell(input_size, hidden_size, bias, nonlinearity))
            else:
                self.rnn_forward.append(base_cell(hidden_size, hidden_size, bias, nonlinearity))

        if self.bidirectional:
            self.rnn_backward = nn.ModuleList()
            for layer in range(self.num_layers):
                if layer == 0:
                    self.rnn_backward.append(base_cell(input_size, hidden_size, bias, nonlinearity))
                else:
                    self.rnn_backward.append(
                        base_cell(hidden_size, hidden_size, bias, nonlinearity)
                    )

    def forward(self, x, h=None):

        if self.batch_first:
            x = torch.transpose(x, 0, 1)
            if h is not None:
                if self.is_lstm:
                    h, c = h
                    h = torch.transpose(h, 0, 1)
                    c = torch.transpose(c, 0, 1)
                else:
                    h = torch.transpose(h, 0, 1)

        batch_size = x.shape[1]
        seq_len = x.shape[0]

        if h is None:
            h = init_hidden(x, self.hidden_size, self.num_layers * self.num_directions)

            if self.is_lstm:
                c = init_hidden(x, self.hidden_size, self.num_layers * self.num_directions)

        elif self.is_lstm:
            h, c = h

        if self.bidirectional:
            h = h.contiguous().view(self.num_layers, 2, batch_size, self.hidden_size)
            h_forward = h[:, 0, :, :]
            h_backward = h[:, 1, :, :]
            if self.is_lstm:
                c = c.contiguous().view(self.num_layers, 2, batch_size, self.hidden_size)
                c_forward = c[:, 0, :, :]
                c_backward = c[:, 1, :, :]

        else:
            h_forward = h
            if self.is_lstm:
                c_forward = c

        # Forward direction
        output_forward = x.new(seq_len, batch_size, self.hidden_size).zero_()
        for t in range(seq_len):
            h_next = h_forward.new(h_forward.shape).zero_()
            if self.is_lstm:
                c_next = c_forward.new(c_forward.shape).zero_()

            for layer in range(self.num_layers):
                if layer == 0:
                    if self.is_lstm:
                        h_next[layer, :, :], c_next[layer, :, :] = self.rnn_forward[layer](
                            x[t, :, :], (h_forward[layer, :, :], c_forward[layer, :, :])
                        )
                    else:
                        h_next[layer, :, :] = self.rnn_forward[layer](
                            x[t, :, :], h_forward[layer, :, :]
                        )

                else:
                    if self.is_lstm:
                        h_next[layer, :, :], c_next[layer, :, :] = self.rnn_forward[layer](
                            h_next[layer - 1, :, :],
                            (h_forward[layer, :, :], c_forward[layer, :, :]),
                        )
                    else:
                        h_next[layer, :, :] = self.rnn_forward[layer](
                            h_next[layer - 1, :, :], h_forward[layer, :, :]
                        )
            output_forward[t, :, :] = h_next[layer, :, :]
            h_forward = h_next
            if self.is_lstm:
                c_forward = c_next

        output = output_forward
        hidden = h_forward
        if self.is_lstm:
            cell = c_forward

        # Backward direction
        if self.bidirectional:
            output_backward = x.new(seq_len, batch_size, self.hidden_size).zero_()
            for t in range(seq_len - 1, -1, -1):
                h_next = h_backward.new(h_backward.shape).zero_()
                if self.is_lstm:
                    c_next = c_backward.new(c_backward.shape).zero_()

                for layer in range(self.num_layers):
                    if layer == 0:
                        if self.is_lstm:
                            h_next[layer, :, :], c_next[layer, :, :] = self.rnn_backward[layer](
                                x[t, :, :], (h_backward[layer, :, :], c_backward[layer, :, :])
                            )
                        else:
                            h_next[layer, :, :] = self.rnn_backward[layer](
                                x[t, :, :], h_backward[layer, :, :]
                            )

                    else:
                        if self.is_lstm:
                            h_next[layer, :, :], c_next[layer, :, :] = self.rnn_backward[layer](
                                h_next[layer - 1, :, :],
                                (h_backward[layer, :, :], c_backward[layer, :, :]),
                            )
                        else:
                            h_next[layer, :, :] = self.rnn_backward[layer](
                                h_next[layer - 1, :, :], h_backward[layer, :, :]
                            )
                output_backward[t, :, :] = h_next[layer, :, :]
                h_backward = h_next
                if self.is_lstm:
                    c_backward = c_next

            output = torch.cat((output, output_backward), dim=-1)
            hidden = torch.cat((hidden, h_backward), dim=0)
            if self.is_lstm:
                cell = torch.cat((cell, c_backward), dim=0)

        if self.batch_first:
            output = torch.transpose(output, 0, 1)
            hidden = torch.transpose(hidden, 0, 1)
            if self.is_lstm:
                cell = torch.transpose(cell, 0, 1)

        if self.is_lstm:
            hidden = (hidden, cell)

        return output, hidden


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


def init_hidden(input, hidden_size, n_layers=None):
    if n_layers is None:
        h = torch.zeros(input.shape[0], hidden_size, dtype=input.dtype, device=input.device)
    else:
        h = torch.zeros(
            n_layers, input.shape[1], hidden_size, dtype=input.dtype, device=input.device
        )
    if input.has_child() and isinstance(input.child, FixedPrecisionTensor):
        h = h.fix_precision()
        child = input.child
        if isinstance(child.child, AdditiveSharingTensor):
            crypto_provider = child.child.crypto_provider
            owners = child.child.locations
            h = h.share(*owners, crypto_provider=crypto_provider)
    return h
