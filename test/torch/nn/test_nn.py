import torch as th
import torch.nn as nn

import syft.frameworks.torch.nn as nn2


def test_Conv2d():
    """
    Test the Conv2d module to ensure that it produces the exact same
    output as the primary torch implementation, in the same order.
    """

    # Disable mkldnn to avoid rounding errors due to difference in implementation
    mkldnn_enabled_init = th._C._get_mkldnn_enabled()
    th._C._set_mkldnn_enabled(False)

    model2 = nn2.Conv2d(1, 32, 3, bias=True)

    model = nn.Conv2d(
        in_channels=1,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    )

    model2.weight = model.weight
    model2.bias = model.bias

    data = th.rand(10, 1, 28, 28)

    out = model(data)

    out2 = model2(data)

    # Reset mkldnn to the original state
    th._C._set_mkldnn_enabled(mkldnn_enabled_init)

    assert th.allclose(out, out2, atol=1e-2)


def test_pool2d():
    """
    Test the Pool2d module to ensure that it produces the exact same
    output as the primary torch implementation, in the same order.
    """

    model = nn.Conv2d(
        in_channels=1,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    )

    pool = nn.AvgPool2d(2)
    pool2 = nn2.AvgPool2d(2)

    data = th.rand(10, 1, 8, 8)

    model_out = model(data)
    out = pool(model_out)
    out2 = pool2(model_out)

    assert th.eq(out, out2).all()


def test_RNNCell():
    """
    Test the RNNCell module to ensure that it produces the exact same
    output as the primary torch implementation, in the same order.
    """

    # Disable mkldnn to avoid rounding errors due to difference in implementation
    mkldnn_enabled_init = th._C._get_mkldnn_enabled()
    th._C._set_mkldnn_enabled(False)

    batch_size = 5
    input_size = 10
    hidden_size = 50

    test_input = th.rand(batch_size, input_size)
    test_hidden = th.rand(batch_size, hidden_size)

    # RNNCell implemented in pysyft
    rnn_syft = nn2.RNNCell(input_size, hidden_size, True, "tanh")

    # RNNCell implemented in original pytorch
    rnn_torch = th.nn.RNNCell(input_size, hidden_size, True, "tanh")

    # Make sure the weights of both RNNCell are identical
    rnn_syft.fc_xh.weight = rnn_torch.weight_ih
    rnn_syft.fc_hh.weight = rnn_torch.weight_hh
    rnn_syft.fc_xh.bias = rnn_torch.bias_ih
    rnn_syft.fc_hh.bias = rnn_torch.bias_hh

    output_syft = rnn_syft(test_input, test_hidden)
    output_torch = rnn_torch(test_input, test_hidden)

    # Reset mkldnn to the original state
    th._C._set_mkldnn_enabled(mkldnn_enabled_init)

    assert th.all(th.lt(th.abs(output_syft - output_torch), 1e-6))


def test_GRUCell():
    """
    Test the GRUCell module to ensure that it produces the exact same
    output as the primary torch implementation, in the same order.
    """

    # Disable mkldnn to avoid rounding errors due to difference in implementation
    mkldnn_enabled_init = th._C._get_mkldnn_enabled()
    th._C._set_mkldnn_enabled(False)

    batch_size = 5
    input_size = 10
    hidden_size = 50

    test_input = th.rand(batch_size, input_size)
    test_hidden = th.rand(batch_size, hidden_size)

    # GRUCell implemented in pysyft
    rnn_syft = nn2.GRUCell(input_size, hidden_size, True)

    # GRUCell implemented in original pytorch
    rnn_torch = th.nn.GRUCell(input_size, hidden_size, True)

    # Make sure the weights of both GRUCell are identical
    rnn_syft.fc_xh.weight = rnn_torch.weight_ih
    rnn_syft.fc_hh.weight = rnn_torch.weight_hh
    rnn_syft.fc_xh.bias = rnn_torch.bias_ih
    rnn_syft.fc_hh.bias = rnn_torch.bias_hh

    output_syft = rnn_syft(test_input, test_hidden)
    output_torch = rnn_torch(test_input, test_hidden)

    # Reset mkldnn to the original state
    th._C._set_mkldnn_enabled(mkldnn_enabled_init)

    assert th.all(th.lt(th.abs(output_syft - output_torch), 1e-6))


def test_LSTMCell():
    """
    Test the LSTMCell module to ensure that it produces the exact same
    output as the primary torch implementation, in the same order.
    """

    # Disable mkldnn to avoid rounding errors due to difference in implementation
    mkldnn_enabled_init = th._C._get_mkldnn_enabled()
    th._C._set_mkldnn_enabled(False)

    batch_size = 5
    input_size = 10
    hidden_size = 50

    test_input = th.rand(batch_size, input_size)
    test_hidden_state = th.rand(batch_size, hidden_size)
    test_cell_state = th.rand(batch_size, hidden_size)

    # LSTMCell implemented in pysyft
    rnn_syft = nn2.LSTMCell(input_size, hidden_size, True)

    # LSTMCell implemented in original pytorch
    rnn_torch = th.nn.LSTMCell(input_size, hidden_size, True)

    # Make sure the weights of both LSTMCell are identical
    rnn_syft.fc_xh.weight = rnn_torch.weight_ih
    rnn_syft.fc_hh.weight = rnn_torch.weight_hh
    rnn_syft.fc_xh.bias = rnn_torch.bias_ih
    rnn_syft.fc_hh.bias = rnn_torch.bias_hh

    hidden_syft, cell_syft = rnn_syft(test_input, (test_hidden_state, test_cell_state))
    hidden_torch, cell_torch = rnn_torch(test_input, (test_hidden_state, test_cell_state))

    # Reset mkldnn to the original state
    th._C._set_mkldnn_enabled(mkldnn_enabled_init)

    # Assert the hidden_state and cell_state of both models are identical separately
    assert th.all(th.lt(th.abs(hidden_syft - hidden_torch), 1e-6))
    assert th.all(th.lt(th.abs(cell_syft - cell_torch), 1e-6))


def test_RNN():
    """
    Test the RNN module to ensure that it produces the exact same
    output as the primary torch implementation, in the same order.
    """

    # Disable mkldnn to avoid rounding errors due to difference in implementation
    mkldnn_enabled_init = th._C._get_mkldnn_enabled()
    th._C._set_mkldnn_enabled(False)

    batch_size = 5
    input_size = 10
    hidden_size = 50
    num_layers = 1
    seq_len = 8

    test_input = th.rand(seq_len, batch_size, input_size)
    test_hidden_state = th.rand(num_layers, batch_size, hidden_size)

    # RNN implemented in pysyft
    rnn_syft = nn2.RNN(input_size, hidden_size, num_layers)

    # RNN implemented in original pytorch
    rnn_torch = th.nn.RNN(input_size, hidden_size, num_layers)

    # Make sure the weights of both RNN are identical
    rnn_syft.rnn_forward[0].fc_xh.weight = rnn_torch.weight_ih_l0
    rnn_syft.rnn_forward[0].fc_xh.bias = rnn_torch.bias_ih_l0
    rnn_syft.rnn_forward[0].fc_hh.weight = rnn_torch.weight_hh_l0
    rnn_syft.rnn_forward[0].fc_hh.bias = rnn_torch.bias_hh_l0

    output_syft, hidden_syft = rnn_syft(test_input, test_hidden_state)
    output_torch, hidden_torch = rnn_torch(test_input, test_hidden_state)

    # Reset mkldnn to the original state
    th._C._set_mkldnn_enabled(mkldnn_enabled_init)

    # Assert the hidden_state and output of both models are identical separately
    assert th.all(th.lt(th.abs(output_syft - output_torch), 1e-6))
    assert th.all(th.lt(th.abs(hidden_syft - hidden_torch), 1e-6))


def test_GRU():
    """
    Test the GRU module to ensure that it produces the exact same
    output as the primary torch implementation, in the same order.
    """

    # Disable mkldnn to avoid rounding errors due to difference in implementation
    mkldnn_enabled_init = th._C._get_mkldnn_enabled()
    th._C._set_mkldnn_enabled(False)

    batch_size = 5
    input_size = 10
    hidden_size = 50
    num_layers = 1
    seq_len = 8

    test_input = th.rand(seq_len, batch_size, input_size)
    test_hidden_state = th.rand(num_layers, batch_size, hidden_size)

    # GRU implemented in pysyft
    rnn_syft = nn2.GRU(input_size, hidden_size, num_layers)

    # GRU implemented in original pytorch
    rnn_torch = th.nn.GRU(input_size, hidden_size, num_layers)

    # Make sure the weights of both GRU are identical
    rnn_syft.rnn_forward[0].fc_xh.weight = rnn_torch.weight_ih_l0
    rnn_syft.rnn_forward[0].fc_xh.bias = rnn_torch.bias_ih_l0
    rnn_syft.rnn_forward[0].fc_hh.weight = rnn_torch.weight_hh_l0
    rnn_syft.rnn_forward[0].fc_hh.bias = rnn_torch.bias_hh_l0

    output_syft, hidden_syft = rnn_syft(test_input, test_hidden_state)
    output_torch, hidden_torch = rnn_torch(test_input, test_hidden_state)

    # Reset mkldnn to the original state
    th._C._set_mkldnn_enabled(mkldnn_enabled_init)

    # Assert the hidden_state and output of both models are identical separately
    assert th.all(th.lt(th.abs(output_syft - output_torch), 1e-6))
    assert th.all(th.lt(th.abs(hidden_syft - hidden_torch), 1e-6))


def test_LSTM():
    """
    Test the LSTM module to ensure that it produces the exact same
    output as the primary torch implementation, in the same order.
    """

    # Disable mkldnn to avoid rounding errors due to difference in implementation
    mkldnn_enabled_init = th._C._get_mkldnn_enabled()
    th._C._set_mkldnn_enabled(False)

    batch_size = 5
    input_size = 10
    hidden_size = 50
    num_layers = 1
    seq_len = 8

    test_input = th.rand(seq_len, batch_size, input_size)
    test_hidden_state = th.rand(num_layers, batch_size, hidden_size)
    test_cell_state = th.rand(num_layers, batch_size, hidden_size)

    # LSTM implemented in pysyft
    rnn_syft = nn2.LSTM(input_size, hidden_size, num_layers)

    # LSTM implemented in original pytorch
    rnn_torch = th.nn.LSTM(input_size, hidden_size, num_layers)

    # Make sure the weights of both LSTM are identical
    rnn_syft.rnn_forward[0].fc_xh.weight = rnn_torch.weight_ih_l0
    rnn_syft.rnn_forward[0].fc_xh.bias = rnn_torch.bias_ih_l0
    rnn_syft.rnn_forward[0].fc_hh.weight = rnn_torch.weight_hh_l0
    rnn_syft.rnn_forward[0].fc_hh.bias = rnn_torch.bias_hh_l0

    output_syft, (hidden_syft, cell_syft) = rnn_syft(
        test_input, (test_hidden_state, test_cell_state)
    )
    output_torch, (hidden_torch, cell_torch) = rnn_torch(
        test_input, (test_hidden_state, test_cell_state)
    )

    # Reset mkldnn to the original state
    th._C._set_mkldnn_enabled(mkldnn_enabled_init)

    # Assert the hidden_state, cell_state and output of both models are identical separately
    assert th.all(th.lt(th.abs(output_syft - output_torch), 1e-6))
    assert th.all(th.lt(th.abs(hidden_syft - hidden_torch), 1e-6))
    assert th.all(th.lt(th.abs(cell_syft - cell_torch), 1e-6))
