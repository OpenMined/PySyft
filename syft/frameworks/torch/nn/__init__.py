from syft.frameworks.torch.nn.conv import Conv2d
from syft.frameworks.torch.nn.functional import conv2d
from syft.frameworks.torch.nn.functional import dropout
from syft.frameworks.torch.nn.functional import linear
from syft.frameworks.torch.nn.pool import AvgPool2d

# from syft.frameworks.torch.nn.rnn import GRU
# from syft.frameworks.torch.nn.rnn import GRUCell
# from syft.frameworks.torch.nn.rnn import LSTM
# from syft.frameworks.torch.nn.rnn import LSTMCell
# from syft.frameworks.torch.nn.rnn import RNN
# from syft.frameworks.torch.nn.rnn import RNNCell
from syft.generic.frameworks.overload import overloaded


@overloaded.module
def nn(module):
    """
    The syntax is the same, so @overloaded.module handles recursion
    Note that we don't need to add the @staticmethod decorator
    """

    @overloaded.module
    def functional(module):
        module.conv2d = conv2d
        module.dropout = dropout
        module.linear = linear

    module.functional = functional
