import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import syft.frameworks.torch.nn as syft_nn


def test_nn_linear(workers):
    torch.manual_seed(121)  # Truncation might not always work so we set the random seed
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    t = torch.tensor([[1.0, 2]])
    x = t.fix_prec().share(bob, alice, crypto_provider=james)
    model = nn.Linear(2, 1)
    model.weight = nn.Parameter(torch.tensor([[-1.0, 2]]))
    model.bias = nn.Parameter(torch.tensor([[-1.0]]))
    model.fix_precision().share(bob, alice, crypto_provider=james)

    y = model(x)

    assert len(alice.object_store._objects) == 4  # x, y, weight, bias
    assert len(bob.object_store._objects) == 4
    assert y.get().float_prec() == torch.tensor([[2.0]])


def test_conv2d(workers):
    """
    Test the nn.Conv2d module to ensure that it produces the exact same
    output as the primary torch implementation, in the same order.
    """
    torch.manual_seed(121)  # Truncation might not always work so we set the random seed

    # Disable mkldnn to avoid rounding errors due to difference in implementation
    mkldnn_enabled_init = torch._C._get_mkldnn_enabled()
    torch._C._set_mkldnn_enabled(False)

    # Direct Import from Syft
    model = syft_nn.Conv2d(1, 2, 3, bias=True)
    model_1 = nn.Conv2d(1, 2, 3, bias=True)
    model.weight = model_1.weight.fix_prec()
    model.bias = model_1.bias.fix_prec()
    data = torch.rand(10, 1, 28, 28)  # eg. mnist data

    out = model(data.fix_prec()).float_prec()
    out_1 = model_1(data)

    assert torch.allclose(out, out_1, atol=1e-2)

    # Fixed Precision Tensor
    model_2 = model_1.copy().fix_prec()
    out_2 = model_2(data.fix_prec()).float_prec()

    # Note: absolute tolerance can be reduced by increasing precision_fractional of fix_prec()
    assert torch.allclose(out_1, out_2, atol=1e-2)

    # Additive Shared Tensor
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])
    shared_data = data.fix_prec().share(bob, alice, crypto_provider=james)

    mode_3 = model_2.share(bob, alice, crypto_provider=james)
    out_3 = mode_3(shared_data).get().float_prec()

    assert torch.allclose(out_1, out_3, atol=1e-2)

    # Reset mkldnn to the original state
    torch._C._set_mkldnn_enabled(mkldnn_enabled_init)


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

    pool = syft_nn.AvgPool2d(2)
    pool_1 = nn.AvgPool2d(2)
    pool_2 = pool_1.copy().fix_prec()

    data = torch.rand(10, 1, 8, 8)

    model_out = model(data)
    out = pool(model_out)
    out_1 = pool_1(model_out)
    out_2 = pool_2(model_out)

    assert torch.eq(out, out_1).all()
    assert torch.eq(out_1, out_2).all()


def test_cnn_model(workers):
    torch.manual_seed(121)  # Truncation might not always work so we set the random seed
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            # TODO: uncomment maxpool2d operations
            # once it is supported with smpc.
            x = F.relu(self.conv1(x))
            # x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            # x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4 * 4 * 50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = Net()
    sh_model = copy.deepcopy(model).fix_precision().share(alice, bob, crypto_provider=james)

    data = torch.zeros((1, 1, 28, 28))
    sh_data = torch.zeros((1, 1, 28, 28)).fix_precision().share(alice, bob, crypto_provider=james)

    assert torch.allclose(sh_model(sh_data).get().float_prec(), model(data), atol=1e-2)


def test_RNNCell():
    """
    Test the RNNCell module to ensure that it produces the exact same
    output as the primary torch implementation, in the same order.
    """

    # Disable mkldnn to avoid rounding errors due to difference in implementation
    mkldnn_enabled_init = torch._C._get_mkldnn_enabled()
    torch._C._set_mkldnn_enabled(False)

    batch_size = 5
    input_size = 10
    hidden_size = 50

    test_input = torch.rand(batch_size, input_size)
    test_hidden = torch.rand(batch_size, hidden_size)

    # RNNCell implemented in pysyft
    rnn_syft = syft_nn.RNNCell(input_size, hidden_size, True, "tanh")

    # RNNCell implemented in original pytorch
    rnn_torch = nn.RNNCell(input_size, hidden_size, True, "tanh")

    # Make sure the weights of both RNNCell are identical
    rnn_syft.fc_xh.weight = rnn_torch.weight_ih
    rnn_syft.fc_hh.weight = rnn_torch.weight_hh
    rnn_syft.fc_xh.bias = rnn_torch.bias_ih
    rnn_syft.fc_hh.bias = rnn_torch.bias_hh

    output_syft = rnn_syft(test_input, test_hidden)
    output_torch = rnn_torch(test_input, test_hidden)

    assert torch.allclose(output_syft, output_torch, atol=1e-2)

    # Reset mkldnn to the original state
    torch._C._set_mkldnn_enabled(mkldnn_enabled_init)


def test_GRUCell():
    """
    Test the GRUCell module to ensure that it produces the exact same
    output as the primary torch implementation, in the same order.
    """

    # Disable mkldnn to avoid rounding errors due to difference in implementation
    mkldnn_enabled_init = torch._C._get_mkldnn_enabled()
    torch._C._set_mkldnn_enabled(False)

    batch_size = 5
    input_size = 10
    hidden_size = 50

    test_input = torch.rand(batch_size, input_size)
    test_hidden = torch.rand(batch_size, hidden_size)

    # GRUCell implemented in pysyft
    rnn_syft = syft_nn.GRUCell(input_size, hidden_size, True)

    # GRUCell implemented in original pytorch
    rnn_torch = nn.GRUCell(input_size, hidden_size, True)

    # Make sure the weights of both GRUCell are identical
    rnn_syft.fc_xh.weight = rnn_torch.weight_ih
    rnn_syft.fc_hh.weight = rnn_torch.weight_hh
    rnn_syft.fc_xh.bias = rnn_torch.bias_ih
    rnn_syft.fc_hh.bias = rnn_torch.bias_hh

    output_syft = rnn_syft(test_input, test_hidden)
    output_torch = rnn_torch(test_input, test_hidden)

    # Reset mkldnn to the original state
    torch._C._set_mkldnn_enabled(mkldnn_enabled_init)

    assert torch.all(torch.lt(torch.abs(output_syft - output_torch), 1e-6))


def test_LSTMCell():
    """
    Test the LSTMCell module to ensure that it produces the exact same
    output as the primary torch implementation, in the same order.
    """

    # Disable mkldnn to avoid rounding errors due to difference in implementation
    mkldnn_enabled_init = torch._C._get_mkldnn_enabled()
    torch._C._set_mkldnn_enabled(False)

    batch_size = 5
    input_size = 10
    hidden_size = 50

    test_input = torch.rand(batch_size, input_size)
    test_hidden_state = torch.rand(batch_size, hidden_size)
    test_cell_state = torch.rand(batch_size, hidden_size)

    # LSTMCell implemented in pysyft
    rnn_syft = syft_nn.LSTMCell(input_size, hidden_size, True)

    # LSTMCell implemented in original pytorch
    rnn_torch = nn.LSTMCell(input_size, hidden_size, True)

    # Make sure the weights of both LSTMCell are identical
    rnn_syft.fc_xh.weight = rnn_torch.weight_ih
    rnn_syft.fc_hh.weight = rnn_torch.weight_hh
    rnn_syft.fc_xh.bias = rnn_torch.bias_ih
    rnn_syft.fc_hh.bias = rnn_torch.bias_hh

    hidden_syft, cell_syft = rnn_syft(test_input, (test_hidden_state, test_cell_state))
    hidden_torch, cell_torch = rnn_torch(test_input, (test_hidden_state, test_cell_state))

    # Reset mkldnn to the original state
    torch._C._set_mkldnn_enabled(mkldnn_enabled_init)

    # Assert the hidden_state and cell_state of both models are identical separately
    assert torch.all(torch.lt(torch.abs(hidden_syft - hidden_torch), 1e-6))
    assert torch.all(torch.lt(torch.abs(cell_syft - cell_torch), 1e-6))


def test_RNN():
    """
    Test the RNN module to ensure that it produces the exact same
    output as the primary torch implementation, in the same order.
    """

    # Disable mkldnn to avoid rounding errors due to difference in implementation
    mkldnn_enabled_init = torch._C._get_mkldnn_enabled()
    torch._C._set_mkldnn_enabled(False)

    batch_size = 5
    input_size = 10
    hidden_size = 50
    num_layers = 1
    seq_len = 8

    test_input = torch.rand(seq_len, batch_size, input_size)
    test_hidden_state = torch.rand(num_layers, batch_size, hidden_size)

    # RNN implemented in pysyft
    rnn_syft = syft_nn.RNN(input_size, hidden_size, num_layers)

    # RNN implemented in original pytorch
    rnn_torch = nn.RNN(input_size, hidden_size, num_layers)

    # Make sure the weights of both RNN are identical
    rnn_syft.rnn_forward[0].fc_xh.weight = rnn_torch.weight_ih_l0
    rnn_syft.rnn_forward[0].fc_xh.bias = rnn_torch.bias_ih_l0
    rnn_syft.rnn_forward[0].fc_hh.weight = rnn_torch.weight_hh_l0
    rnn_syft.rnn_forward[0].fc_hh.bias = rnn_torch.bias_hh_l0

    output_syft, hidden_syft = rnn_syft(test_input, test_hidden_state)
    output_torch, hidden_torch = rnn_torch(test_input, test_hidden_state)

    # Reset mkldnn to the original state
    torch._C._set_mkldnn_enabled(mkldnn_enabled_init)

    # Assert the hidden_state and output of both models are identical separately
    assert torch.all(torch.lt(torch.abs(output_syft - output_torch), 1e-6))
    assert torch.all(torch.lt(torch.abs(hidden_syft - hidden_torch), 1e-6))


def test_GRU():
    """
    Test the GRU module to ensure that it produces the exact same
    output as the primary torch implementation, in the same order.
    """

    # Disable mkldnn to avoid rounding errors due to difference in implementation
    mkldnn_enabled_init = torch._C._get_mkldnn_enabled()
    torch._C._set_mkldnn_enabled(False)

    batch_size = 5
    input_size = 10
    hidden_size = 50
    num_layers = 1
    seq_len = 8

    test_input = torch.rand(seq_len, batch_size, input_size)
    test_hidden_state = torch.rand(num_layers, batch_size, hidden_size)

    # GRU implemented in pysyft
    rnn_syft = syft_nn.GRU(input_size, hidden_size, num_layers)

    # GRU implemented in original pytorch
    rnn_torch = nn.GRU(input_size, hidden_size, num_layers)

    # Make sure the weights of both GRU are identical
    rnn_syft.rnn_forward[0].fc_xh.weight = rnn_torch.weight_ih_l0
    rnn_syft.rnn_forward[0].fc_xh.bias = rnn_torch.bias_ih_l0
    rnn_syft.rnn_forward[0].fc_hh.weight = rnn_torch.weight_hh_l0
    rnn_syft.rnn_forward[0].fc_hh.bias = rnn_torch.bias_hh_l0

    output_syft, hidden_syft = rnn_syft(test_input, test_hidden_state)
    output_torch, hidden_torch = rnn_torch(test_input, test_hidden_state)

    # Reset mkldnn to the original state
    torch._C._set_mkldnn_enabled(mkldnn_enabled_init)

    # Assert the hidden_state and output of both models are identical separately
    assert torch.all(torch.lt(torch.abs(output_syft - output_torch), 1e-6))
    assert torch.all(torch.lt(torch.abs(hidden_syft - hidden_torch), 1e-6))


def test_LSTM():
    """
    Test the LSTM module to ensure that it produces the exact same
    output as the primary torch implementation, in the same order.
    """

    # Disable mkldnn to avoid rounding errors due to difference in implementation
    mkldnn_enabled_init = torch._C._get_mkldnn_enabled()
    torch._C._set_mkldnn_enabled(False)

    batch_size = 5
    input_size = 10
    hidden_size = 50
    num_layers = 1
    seq_len = 8

    test_input = torch.rand(seq_len, batch_size, input_size)
    test_hidden_state = torch.rand(num_layers, batch_size, hidden_size)
    test_cell_state = torch.rand(num_layers, batch_size, hidden_size)

    # LSTM implemented in pysyft
    rnn_syft = syft_nn.LSTM(input_size, hidden_size, num_layers)

    # LSTM implemented in original pytorch
    rnn_torch = nn.LSTM(input_size, hidden_size, num_layers)

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
    torch._C._set_mkldnn_enabled(mkldnn_enabled_init)

    # Assert the hidden_state, cell_state and output of both models are identical separately
    assert torch.all(torch.lt(torch.abs(output_syft - output_torch), 1e-6))
    assert torch.all(torch.lt(torch.abs(hidden_syft - hidden_torch), 1e-6))
    assert torch.all(torch.lt(torch.abs(cell_syft - cell_torch), 1e-6))
