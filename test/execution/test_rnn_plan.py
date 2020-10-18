import traceback

import numpy as np
import torch as th
import torch.nn as nn

import syft as sy
from syft.execution.plan import Plan
from syft.execution.translation.threepio import PlanTranslatorTfjs
from syft.execution.translation.torchscript import PlanTranslatorTorchscript


# Modified from handcrafted_GRU.py
class CustomGruCell(nn.Module):
    """
    A forward only GRU cell.
    Input should be: (sequence length x batch size x input_size).
    The output is the output of the final forward call.
    It's not clear if it would be possible to use the output from each cell in a Plan
    because of the assumptions of 2D tensors in backprop.
    """

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

    def forward(self, x, hidden=None, sequence_length=None):
        if hidden is None:
            batch_size = x.shape[1]
            hidden = self.init_hidden(batch_size)
        if sequence_length is None:
            sequence_length = x.shape[0]
        else:
            # The sequence length should always be the same size when running the Plan.
            # But maybe we can one day be more dynamic and use it.
            sequence_length = sequence_length.item()

        for t in range(sequence_length):
            # `x.select(0, t)` == `x[t, :, :]` but it can be converted to Tensorflow.js.
            hidden = self.gru_cell(x.select(0, t), hidden)
        # Just return the result of the final cell
        # since some PySyft autograd features seem like they have issues with 3D tensors.
        output = hidden
        return output, hidden


def test_rnn_plan_example():
    """
    Prepares simple model-centric federated learning training plan example that use an RNN.
    """
    # Disable translators
    Plan._build_translators = []

    class Net(nn.Module):
        def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx=0):
            super(Net, self).__init__()
            self.vocab_size = vocab_size
            self.padding_idx = padding_idx

            # It would be nice use PyTorch's nn.Embedding and let them be trainable:
            # `self.encoder = nn.Embedding(self.vocab_size, embedding_size)`
            # but gradients didn't get computed for its weights when building the Plan.
            # We could make our own custom embedding layer but it didn't work yet in PySyft:
            # * Doing
            #   `output = self.weight[embedding_indices]`
            #   doesn't work with PySyft's implementation for __getitem__
            #   because the indices have too many dimensions.
            # * Doing the lookup "manually" with loops gave grad=None for the embeddings too.
            embeddings = th.zeros(self.vocab_size, embedding_size)
            self.encoder: nn.Embedding = nn.Embedding.from_pretrained(
                embeddings, padding_idx=self.padding_idx
            )
            self.encoder.reset_parameters()

            self.rnn = CustomGru(embedding_size, hidden_size)
            self.decoder = nn.Linear(hidden_size, self.vocab_size)

        def init_hidden(self, batch_size):
            return self.rnn.init_hidden(batch_size)

        def forward(self, x, hidden=None, sequence_length=None):
            embeddings = self.encoder(x)
            output, hidden = self.rnn(embeddings, hidden, sequence_length)
            output = self.decoder(output)
            return output, hidden

    def set_model_params(module, params_list, start_param_idx=0):
        """
        Set params list into model recursively.
        """
        param_idx = start_param_idx

        for name, param in module._parameters.items():
            # A param can be None if it is not trainable.
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

    model = Net(10, 8, 8)

    @sy.func2plan()
    def train(data, initial_hidden, targets, lr, batch_size, sequence_length, model_parameters):
        set_model_params(model, model_parameters)

        logits, hidden = model(data, initial_hidden, sequence_length)

        loss = softmax_cross_entropy_with_logits(logits, targets, batch_size)

        loss.backward()

        num_none_grads = len(list(filter(lambda param: param.grad is None, model_parameters)))
        # Only the grad for the embeddings will be None.
        assert (
            num_none_grads == 1
        ), f"{num_none_grads}/{len(model_parameters)} model params have None grad(s)."
        assert model_parameters[0].grad is None, "The grad for the embeddings should be None."

        updated_params = [naive_sgd(param, lr=lr) for param in model_parameters]

        pred = th.argmax(logits, dim=1)
        targets_idx = th.argmax(targets, dim=1)
        acc = pred.eq(targets_idx).sum().float() / batch_size

        return (loss, acc, *updated_params)

    # Set up dummy inputs.
    # These size must always be the same when using the model.
    batch_size = 3
    # Changing the sequence length from 2 to 3 can make the Torchscript take much much longer.
    # (a few seconds to 1min when testing on a decent machine).
    # The sequence length should always be the same size when running the Plan.
    # But maybe we can one day be more dynamic and use it.
    # For non-example purposes, increase the sequence length and use padding on the data.
    sequence_length = 2
    vocab_size = model.vocab_size
    # Data has the index of the word in a vocabulary.
    # Start token indices after padding index.
    token_start_index = max(model.padding_idx + 1, 1)
    data = th.randint(token_start_index, vocab_size, (sequence_length, batch_size))

    # Test the model with no default hidden state.
    output, hidden = model(data)
    assert output.shape == th.Size([batch_size, vocab_size])
    assert hidden.shape == th.Size([batch_size, model.rnn.hidden_size])

    # The model can initialize the hidden state if it is not set
    # but this might not work within a Plan.
    initial_hidden = model.init_hidden(batch_size)

    # Predicting the next word for each sequence.
    targets = th.randint(0, vocab_size, (batch_size,))
    targets = nn.functional.one_hot(targets, vocab_size)

    lr = th.tensor([0.1])
    batch_size = th.tensor([batch_size])
    sequence_length = th.tensor([sequence_length])
    model_state = list(model.parameters())

    # Build Plan
    train.build(
        data,
        initial_hidden,
        targets,
        lr,
        batch_size,
        sequence_length,
        model_state,
        trace_autograd=True,
    )

    # Original forward func (Torch autograd)
    loss_torch, acc_torch, *params_torch = train(
        data,
        initial_hidden,
        targets,
        lr,
        batch_size,
        sequence_length,
        model_state,
    )

    # Traced forward func (traced autograd)
    train.forward = None
    loss_syft, acc_syft, *params_syft = train(
        data, initial_hidden, targets, lr, batch_size, sequence_length, model_state
    )

    print("Translating Plan to Tfjs.")
    try:
        train.add_translation(PlanTranslatorTfjs)
    except Exception as e:
        print(
            "Failed to translate the Plan to Tensorflow.js."
            " A later version of 3p0 (>0.2) that might not be released yet is required."
        )
        print(traceback.format_exc())

    # Translate Plan to Torchscript
    print("Translating Plan to Torchscript")
    train.add_translation(PlanTranslatorTorchscript)
    print("Running Torchscript Plan")
    loss_ts, acc_ts, *params_ts = train.torchscript(
        data, initial_hidden, targets, lr, batch_size, sequence_length, model_state
    )

    # Tests
    loss, acc = loss_torch, acc_torch
    assert loss is not None
    assert loss.shape == th.Size([1])
    assert loss.item() > 0
    assert acc is not None
    assert acc.shape == th.Size([1])
    assert acc.item() >= 0

    # Outputs from each type of run should be equal.
    assert th.allclose(loss_torch, loss_syft), f"loss torch {loss_torch} != loss syft {loss_syft}"
    assert th.allclose(acc_torch, acc_syft), "acc torch/syft"
    assert th.allclose(loss_torch, loss_ts), f"loss torch {loss_torch} != loss ts {loss_ts}"
    assert th.allclose(acc_torch, acc_ts), "acc torch/ts"
    # Skip embedding weights comparison (1st param).
    for i, param_torch in enumerate(params_torch[1:], start=1):
        assert th.allclose(
            param_torch, params_syft[i]
        ), f"param {i} (out_{i + 3}) torch/syft. {th.abs(param_torch - params_syft[i]).max()}"
        assert th.allclose(
            param_torch, params_ts[i]
        ), f"param {i} (out_{i + 3}) torch/ts. {th.abs(param_torch - params_ts[i]).max()}"
