import numpy as np

from torch import nn
import torch.nn.functional as F


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # reset gate
        self.fc_ir = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hr = nn.Linear(hidden_size, hidden_size, bias=bias)

        # update gate
        self.fc_iz = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hz = nn.Linear(hidden_size, hidden_size, bias=bias)

        # new gate
        self.fc_in = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hn = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.init_parameters()

    def init_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h):

        x = x.view(-1, x.shape[1])

        i_r = self.fc_ir(x)
        h_r = self.fc_hr(h)
        i_z = self.fc_iz(x)
        h_z = self.fc_hz(h)
        i_n = self.fc_in(x)
        h_n = self.fc_hn(h)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_z + h_z)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (h - newgate)

        return hy


class GRU(nn.Module):
    def __init__(
        self, vocab_size, output_size=1, embedding_dim=50, hidden_dim=10, bias=True, dropout=0.2
    ):
        super(GRU, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_size = output_size

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # GRU Cell
        self.gru_cell = GRUCell(embedding_dim, hidden_dim)
        # Fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        # Sigmoid layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):

        batch_size = x.shape[0]

        # Deal with cases were the current batch_size is different from general batch_size
        # It occurrs at the end of iteration with the Dataloaders
        if h.shape[0] != batch_size:
            h = h[:batch_size, :].contiguous()

        # Apply embedding
        x = self.embedding(x)

        # GRU cells
        for t in range(x.shape[1]):
            h = self.gru_cell(x[:, t, :], h)

        # Output corresponds to the last hidden state
        out = h.contiguous().view(-1, self.hidden_dim)

        # Dropout and fully-connected layers
        out = self.dropout(out)
        sig_out = self.sigmoid(self.fc(out))

        return sig_out, h
