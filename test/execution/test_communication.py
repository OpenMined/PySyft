import pytest

import syft as sy

from syft.execution.communication import CommunicationAction


def test_communication_methods_accepted():
    c = CommunicationAction("send", None, (), {}, ())

    assert c.name == "send"


def test_computation_methods_rejected():
    with pytest.raises(ValueError):
        CommunicationAction("__add__", None, (), {}, ())
