import pytest
import torch as th
import syft as sy

from syft.frameworks.torch.tensors.decorators import LoggingTensor


def test_tensors_not_collated_exception(workers):
    """
    Ensure that the sy.combine_pointers works as expected
    """

    bob = workers["bob"]
    alice = workers["alice"]

    x = th.tensor([1, 2, 3, 4, 5]).send(bob)
    y = th.tensor([1, 2, 3, 4, 5]).send(alice)

    try:

        b = x + y
        assert False
    except sy.exceptions.TensorsNotCollocatedException as e:
        assert True

    x = th.tensor([1, 2, 3, 4, 5]).send(alice)
    y = th.tensor([1, 2, 3, 4, 5]).send(bob)

    try:

        b = x + y
        assert False
    except sy.exceptions.TensorsNotCollocatedException as e:
        assert True

    x = th.tensor([1, 2, 3, 4, 5]).send(alice)
    y = th.tensor([1, 2, 3, 4, 5])

    try:

        b = x + y
        assert False
    except sy.exceptions.TensorsNotCollocatedException as e:
        assert True

    x = th.tensor([1, 2, 3, 4, 5])
    y = th.tensor([1, 2, 3, 4, 5]).send(alice)

    try:

        b = x + y
        assert False
    except sy.exceptions.TensorsNotCollocatedException as e:
        assert True
