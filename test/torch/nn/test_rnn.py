import pytest
import torch as th
import syft as sy
from syft.frameworks.torch.nn import RNN
from syft.frameworks.torch.nn import GRU
from syft.frameworks.torch.nn import LSTM


@pytest.mark.parametrize("module", ["lstm", "gru", "rnn_tanh", "rnn_relu"])
def test_rnn(module, hook, workers):
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
