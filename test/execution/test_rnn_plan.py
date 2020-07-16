import numpy as np
import torch as th
import torch.nn as nn

import syft as sy
from syft.execution.plan import Plan


# Modified from handcrafted_GRU.py
class CustomGruCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(CustomGruCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Reset Gate
        self.fc_ir = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hr = nn.Linear(hidden_size, hidden_size, bias=bias)

        # Update Gate
        self.fc_iz = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hz = nn.Linear(hidden_size, hidden_size, bias=bias)

        # New Gate
        self.fc_in = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hn = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.init_parameters()

    def init_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h):
        i_r = self.fc_ir(x)
        h_r = self.fc_hr(h)
        i_z = self.fc_iz(x)
        h_z = self.fc_hz(h)
        i_n = self.fc_in(x)
        h_n = self.fc_hn(h)

        # Activation functions need to be on the object (not functional)
        # for PySyft gradient stuff to work.
        resetgate = (i_r + h_r).sigmoid()
        inputgate = (i_z + h_z).sigmoid()
        newgate = (i_n + (resetgate * h_n)).tanh()

        hy = newgate + inputgate * (h - newgate)

        return hy


class CustomGru(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(CustomGru, self).__init__()
        self.hidden_size = hidden_size
        self.gru_cell = CustomGruCell(input_size, hidden_size, bias)

    def init_hidden(self, batch_size):
        return th.zeros(batch_size, self.hidden_size)

    def forward(self, x, hidden=None):
        # For the plan to work, the sequence length must always be the same.
        sequence_length = x.shape[0]

        if hidden is None:
            batch_size = x.shape[1]
            hidden = self.init_hidden(batch_size)
        for t in range(sequence_length):
            hidden = self.gru_cell(x[t, :, :], hidden)
        # Just return the result of the final cell
        # since some PySyft autograd features seem like they have issues with 3D tensors.
        output = hidden
        return output, hidden


def test_rnn_plan_example():
    """
    Prepares simple static federated learning training plan example that use an RNN.
    """
    # Disable translators
    Plan._build_translators = []

    class Net(nn.Module):
        def __init__(self, vocab_size, embedding_size, hidden_size):
            super(Net, self).__init__()
            self.vocab_size = vocab_size
            self.encoder = nn.Embedding(self.vocab_size, embedding_size)
            self.rnn = CustomGru(embedding_size, hidden_size)
            self.decoder = nn.Linear(hidden_size, self.vocab_size)

        def init_hidden(self, batch_size):
            if hasattr(self, 'rnn'):
                return self.rnn.init_hidden(batch_size)
            else:
                return None

        def forward(self, x, hidden=None):
            emb = self.encoder(x)
            output, hidden = self.rnn(emb, hidden)
            output = self.decoder(output)
            return output, hidden

    def set_model_params(module, params_list, start_param_idx=0):
        """
        Set params list into model recursively.
        """
        param_idx = start_param_idx

        for name, param in module._parameters.items():
            # A param can be None if it is not used.
            if param is not None:
                module._parameters[name] = params_list[param_idx]
                param_idx += 1

        for name, child in module._modules.items():
            if child is not None:
                param_idx = set_model_params(child, params_list, param_idx)

        return param_idx

    def softmax_cross_entropy_with_logits(logits, targets, batch_size):
        # numstable logsoftmax
        norm_logits = logits - logits.max()
        log_probs = norm_logits - norm_logits.exp().sum(dim=1, keepdim=True).log()
        return -(targets * log_probs).sum() / batch_size

    def naive_sgd(param, **kwargs):
        if param.grad is None:
            # A grad can be None if you used operations that are not supported
            # by PySyft's autograd features or the param
            # isn't trainable (e.g. nn.Embedding was used).
            return param
        return param - kwargs["lr"] * param.grad

    model = Net(100, 64, 64)

    @sy.func2plan()
    def train(data, initial_hidden, targets, lr, batch_size, model_parameters):
        set_model_params(model, model_parameters)

        logits, hidden = model(data, initial_hidden)

        loss = softmax_cross_entropy_with_logits(logits, targets, batch_size)

        loss.backward()

        num_none_grads = len(list(filter(lambda param: param.grad is None, model_parameters)))
        s = f"{num_none_grads}/{len(model_parameters)} model params have None grad(s)."
        print(s)
        # Only the grad for the embeddings will be None.
        assert num_none_grads == 1, s

        updated_params = [naive_sgd(param, lr=lr) for param in model_parameters]

        pred = th.argmax(logits, dim=1)
        targets_idx = th.argmax(targets, dim=1)
        acc = pred.eq(targets_idx).sum().float() / batch_size

        return (loss, acc, *updated_params)

    # Dummy inputs
    batch_size = 3
    sequence_length = 3
    vocab_size = model.vocab_size
    # Data has the index of the word in a vocabulary.
    data = th.randint(0, vocab_size, (sequence_length, batch_size))

    # The model can initialize the hidden state if it is not set
    # but this might not work within a Plan.
    initial_hidden = model.init_hidden(batch_size)

    # Predicting the next word for each sequence.
    targets = th.randint(0, vocab_size, (batch_size,))
    targets = nn.functional.one_hot(targets, vocab_size)

    lr = th.tensor([0.1])
    batch_size = th.tensor([float(batch_size)])
    model_state = list(model.parameters())

    # Build Plan
    build_result = train.build(data, initial_hidden, targets, lr, batch_size, model_state,
                               trace_autograd=True)
    loss, acc = build_result[:2]
    assert loss is not None
    assert loss.shape == th.Size([1])
    assert loss.item() > 0
    assert acc is not None
    assert acc.shape == th.Size([1])
    assert acc.item() >= 0
