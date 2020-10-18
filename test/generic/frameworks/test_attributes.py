import pytest
import torch as th

from syft.test import my_awesome_computation
from syft.generic.utils import remote


@pytest.mark.parametrize("return_value", [True, False])
def test_remote(workers, return_value):
    alice = workers["alice"]

    x = th.tensor([1.0])
    expected = my_awesome_computation(x)

    p = x.send(alice)
    args = (p,)
    results = remote(my_awesome_computation, location=alice)(
        *args, return_value=return_value, return_arity=2
    )

    if not return_value:
        results = tuple(result.get() for result in results)

    assert results == expected


@pytest.mark.parametrize("return_value", [True, False])
def test_remote_wrong_arity(workers, return_value):
    """
    Identical to test_remote except the use didn't set return_arity to
    be the correct number of return values.
    Here it should be 2, not 1.
    """
    alice = workers["alice"]

    x = th.tensor([1.0])
    expected = my_awesome_computation(x)

    p = x.send(alice)
    args = (p,)
    results = remote(my_awesome_computation, location=alice)(
        *args, return_value=return_value, return_arity=1
    )

    if not return_value:
        results = tuple(result.get() for result in results)

    assert results == expected
