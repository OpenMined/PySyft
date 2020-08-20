from syft.frameworks.torch.nn.conv import Conv2d
from syft.frameworks.torch.nn.functional import conv2d
from syft.frameworks.torch.nn.functional import max_pool2d
from syft.frameworks.torch.nn.functional import avg_pool2d
from syft.frameworks.torch.nn.functional import adaptive_avg_pool2d
from syft.frameworks.torch.nn.functional import dropout
from syft.frameworks.torch.nn.functional import linear
from syft.frameworks.torch.nn.functional import batch_norm
from syft.frameworks.torch.nn.pool import AvgPool2d
from syft.frameworks.torch.nn.rnn import GRU
from syft.frameworks.torch.nn.rnn import GRUCell
from syft.frameworks.torch.nn.rnn import LSTM
from syft.frameworks.torch.nn.rnn import LSTMCell
from syft.frameworks.torch.nn.rnn import RNN
from syft.frameworks.torch.nn.rnn import RNNBase
from syft.frameworks.torch.nn.rnn import RNNCell
from syft.frameworks.torch.nn.rnn import RNNCellBase
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
        module.max_pool2d = max_pool2d
        module.avg_pool2d = avg_pool2d
        module.adaptive_avg_pool2d = adaptive_avg_pool2d
        module.batch_norm = batch_norm

    module.functional = functional

    module.Conv2d = Conv2d
    module.AvgPool2d = AvgPool2d
    module.GRU = GRU
    module.GRUCell = GRUCell
    module.LSTM = LSTM
    module.LSTMCell = LSTMCell
    module.RNN = RNN
    module.RNNBase = RNNBase
    module.RNNCell = RNNCell
    module.RNNCellBase = RNNCellBase
