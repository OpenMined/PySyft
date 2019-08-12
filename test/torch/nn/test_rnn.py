import pytest
import torch as th
import syft as sy
from syft.frameworks.torch.nn import RNN
from syft.frameworks.torch.nn import GRU
from syft.frameworks.torch.nn import LSTM


@pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
def test_rnn(nonlinearity, hook, workers):
    """
    Testing RNN modules with MPC
    """

    bob = workers["bob"]
    alice = workers["alice"]
    crypto_prov = sy.VirtualWorker(hook, id="crypto_prov")

    # model hyperparameters
    batch_size = 4
    input_size = 3
    hidden_size = 5
    seq_len = 6
    num_layers = 2
    bidirectional = True
    directions = 2 if bidirectional else 1

    x = (
        th.randn(batch_size, seq_len, input_size)
        .fix_precision()
        .share(alice, bob, crypto_provider=crypto_prov)
    )
    model = (
        RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
            bidirectional=bidirectional,
        )
        .fix_precision()
        .share(alice, bob, crypto_provider=crypto_prov)
    )

    out, h = model(x)
    assert out.get().float_precision().shape == th.Size(
        [batch_size, seq_len, directions * hidden_size]
    )
    assert h.get().float_precision().shape == th.Size(
        [batch_size, directions * num_layers, hidden_size]
    )

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


def test_gru(hook, workers):
    bob = workers["bob"]
    alice = workers["alice"]
    crypto_prov = sy.VirtualWorker(hook, id="crypto_prov")

    # model hyperparameters
    batch_size = 4
    input_size = 3
    hidden_size = 5
    seq_len = 6
    num_layers = 2
    bidirectional = True
    directions = 2 if bidirectional else 1

    x = (
        th.randn(batch_size, seq_len, input_size)
        .fix_precision()
        .share(alice, bob, crypto_provider=crypto_prov)
    )
    model = (
        GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        .fix_precision()
        .share(alice, bob, crypto_provider=crypto_prov)
    )

    out, h = model(x)
    assert out.get().float_precision().shape == th.Size(
        [batch_size, seq_len, directions * hidden_size]
    )
    assert h.get().float_precision().shape == th.Size(
        [batch_size, directions * num_layers, hidden_size]
    )

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


def test_lstm(hook, workers):
    bob = workers["bob"]
    alice = workers["alice"]
    crypto_prov = sy.VirtualWorker(hook, id="crypto_prov")

    # model hyperparameters
    batch_size = 4
    input_size = 3
    hidden_size = 5
    seq_len = 6
    num_layers = 2
    bidirectional = True
    directions = 2 if bidirectional else 1

    x = (
        th.randn(batch_size, seq_len, input_size)
        .fix_precision()
        .share(alice, bob, crypto_provider=crypto_prov)
    )
    model = (
        LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        .fix_precision()
        .share(alice, bob, crypto_provider=crypto_prov)
    )

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

    h0 = (
        th.randn(batch_size, directions * num_layers, hidden_size)
        .fix_precision()
        .share(alice, bob, crypto_provider=crypto_prov)
    )
    out, (h, c) = model(x, h0)
    assert out.get().float_precision().shape == th.Size(
        [batch_size, seq_len, directions * hidden_size]
    )
    assert h.get().float_precision().shape == th.Size(
        [batch_size, directions * num_layers, hidden_size]
    )
    assert c.get().float_precision().shape == th.Size(
        [batch_size, directions * num_layers, hidden_size]
    )
