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
