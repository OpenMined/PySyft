import pytest
import torch as th
from torch import nn
from torch import optim
import syft as sy
from syft.frameworks.torch.nn import RNN
from syft.frameworks.torch.nn import GRU
from syft.frameworks.torch.nn import LSTM


@pytest.mark.parametrize("module", ["lstm", "gru", "rnn_tanh", "rnn_relu"])
def test_rnn_mpc(module, hook, workers):
    """
    Testing RNN modules with MPC
    """
    th.manual_seed(42)  # Truncation might not always work so we set the random seed
    bob = workers["bob"]
    alice = workers["alice"]
    crypto_prov = workers["james"]

    # model hyperparameters
    batch_size = 2
    input_size = 2
    hidden_size = 2
    seq_len = 2
    num_layers = 2
    bidirectional = True
    dropout = 0.1
    directions = 2 if bidirectional else 1

    if module == "lstm":
        model = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
    elif module == "gru":
        model = GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
    elif module == "rnn_tanh":
        model = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            bidirectional=bidirectional,
        )
    else:
        model = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="relu",
            batch_first=True,
            bidirectional=bidirectional,
        )

    model = model.fix_precision().share(alice, bob, crypto_provider=crypto_prov)

    x = (
        th.randn(batch_size, seq_len, input_size)
        .fix_precision()
        .share(alice, bob, crypto_provider=crypto_prov)
    )

    if module == "lstm":
        # Testing with initialization of hidden state internally
        out, (h, c) = model(x)
        assert out.get().float_precision().shape == th.Size(
            [batch_size, seq_len, directions * hidden_size]
        )
        assert h.get().float_precision().shape == th.Size(
            [batch_size, directions * num_layers, hidden_size]
        )
        assert c.get().float_precision().shape == th.Size(
            [batch_size, directions * num_layers, hidden_size]
        )

        # Testing with initialization of hidden state externally
        h0 = (
            th.randn(batch_size, directions * num_layers, hidden_size)
            .fix_precision()
            .share(alice, bob, crypto_provider=crypto_prov)
        )
        c0 = (
            th.randn(batch_size, directions * num_layers, hidden_size)
            .fix_precision()
            .share(alice, bob, crypto_provider=crypto_prov)
        )
        out, (h, c) = model(x, (h0, c0))
        assert out.get().float_precision().shape == th.Size(
            [batch_size, seq_len, directions * hidden_size]
        )
        assert h.get().float_precision().shape == th.Size(
            [batch_size, directions * num_layers, hidden_size]
        )
        assert c.get().float_precision().shape == th.Size(
            [batch_size, directions * num_layers, hidden_size]
        )

    else:
        # Testing with initialization of hidden state internally
        out, h = model(x)
        assert out.get().float_precision().shape == th.Size(
            [batch_size, seq_len, directions * hidden_size]
        )
        assert h.get().float_precision().shape == th.Size(
            [batch_size, directions * num_layers, hidden_size]
        )

        # Testing with initialization of hidden state externally
        h0 = (
            th.randn(batch_size, directions * num_layers, hidden_size)
            .fix_precision()
            .share(alice, bob, crypto_provider=crypto_prov)
        )
        out, h = model(x, h0)
        assert out.get().float_precision().shape == th.Size(
            [batch_size, seq_len, directions * hidden_size]
        )
        assert h.get().float_precision().shape == th.Size(
            [batch_size, directions * num_layers, hidden_size]
        )


@pytest.mark.parametrize("module", ["lstm", "gru", "rnn_tanh", "rnn_relu"])
def test_rnn_federated(module, hook, workers):
    """
    Testing Federated Learning with RNN modules
    """

    bob = workers["bob"]
    alice = workers["alice"]

    # model hyperparameters
    data_size = 2
    input_size = 2
    hidden_size = 2
    seq_len = 2
    num_layers = 2
    bidirectional = True

    # Create toy dataset and federated loader
    x = th.randn(data_size, seq_len, input_size)
    y = th.rand(data_size)

    idx = int(data_size / 2)
    bob_dataset = sy.BaseDataset(x[:idx], y[:idx]).send(bob)
    alice_dataset = sy.BaseDataset(x[idx:], y[idx:]).send(alice)
    federated_dataset = sy.FederatedDataset([bob_dataset, alice_dataset])
    federated_loader = sy.FederatedDataLoader(federated_dataset, shuffle=True, batch_size=2)

    # Build Model
    if module == "lstm":
        model = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
    if module == "gru":
        model = GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
    if module == "rnn_tanh":
        model = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            bidirectional=bidirectional,
            batch_first=True,
        )
    if module == "rnn_relu":
        model = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="relu",
            bidirectional=bidirectional,
            batch_first=True,
        )

    # Training
    opt = optim.SGD(model.parameters(), lr=0.01)
    losses = []
    for e in range(10):
        current_losses = []
        for data, target in federated_loader:
            model.send(data.location)
            opt.zero_grad()
            output, _ = model(data)
            pred = output.sum(dim=1).sum(dim=1)
            loss = ((pred - target) ** 2).sum()
            loss.backward()
            opt.step()
            current_losses.append(loss.get())
            model = model.get()
        losses.append(sum(current_losses))

    assert losses[0] > losses[-1]
