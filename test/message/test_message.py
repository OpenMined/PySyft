import syft as sy
import syft

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from syft.serde.serde import deserialize
from syft.serde.serde import serialize
import time

import pytest
import unittest.mock as mock

from syft.workers import WebsocketClientWorker
from syft.workers import WebsocketServerWorker


def test_message_serde(hook):

    x = sy.Message(0, [1,2,3])
    x_bin = sy.serde.serialize(x)
    y = sy.serde.deserialize(x_bin, sy.local_worker)

    assert x.contents == y.contents
    assert x.msg_type == y.msg_type