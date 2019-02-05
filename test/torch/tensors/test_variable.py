"""All the tests of things which are exclusively gradient focused. If
you are working on gradients being used by other abstractions, don't
use this class. Use the abstraction's test class instead. (I.e., if you
are testing gradients with PointerTensor, use test_pointer.py.)"""

import random

import torch
import syft as sy
from syft.frameworks.torch.tensors.decorators import LoggingTensor


def test_gradient_serde():
    # create a tensor
    x = torch.tensor([1, 2, 3, 4.0], requires_grad=True)

    # create gradient on tensor
    x.sum().backward(torch.ones(1))

    # save gradient
    orig_grad = x.grad

    # serialize
    blob = sy.serde.serialize(x)

    # deserialize
    t = sy.serde.deserialize(blob)

    # check that gradient was properly serde
    assert (t.grad == orig_grad).all()
