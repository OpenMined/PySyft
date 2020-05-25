import pytest
import torch as th
import syft as sy


def test_tensors_not_collated_exception(workers):
    """
    Ensure that the sy.combine_pointers works as expected
    """

    bob = workers["bob"]
    alice = workers["alice"]

    x = th.tensor([1, 2, 3, 4, 5]).send(bob)
    y = th.tensor([1, 2, 3, 4, 5]).send(alice)

    with pytest.raises(sy.exceptions.TensorsNotCollocatedException):
        b = x + y

    x = th.tensor([1, 2, 3, 4, 5]).send(alice)
    y = th.tensor([1, 2, 3, 4, 5]).send(bob)

    with pytest.raises(sy.exceptions.TensorsNotCollocatedException):
        b = x + y

    x = th.tensor([1, 2, 3, 4, 5]).send(alice)
    y = th.tensor([1, 2, 3, 4, 5])

    with pytest.raises(sy.exceptions.TensorsNotCollocatedException):
        b = x + y

    x = th.tensor([1, 2, 3, 4, 5])
    y = th.tensor([1, 2, 3, 4, 5]).send(alice)

    with pytest.raises(sy.exceptions.TensorsNotCollocatedException):
        b = x + y
