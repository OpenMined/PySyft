import importlib

import torch as th
from torch import nn

from syft.generic.constructor.object_constructor import ObjectConstructor


def test_object_constructors():
    class LowercaseTensorConstructor(ObjectConstructor):
        # Step 1: Store the attribute name that this constructor is replacing
        constructor_name = "tensor"

        # Step 2: Store a reference to the location on which this constructor currently lives.
        # This is also the location that this custom constructor will live once installed using
        # self.install_inside_library()
        constructor_location = th

    LowercaseTensorConstructor().install_inside_library()

    class UppercaseTensorConstructor(ObjectConstructor):
        # Step 1: Store the attribute name that this constructor is replacing
        constructor_name = "Tensor"

        # Step 2: Store a reference to the location on which this constructor currently lives.
        # This is also the location that this custom constructor will live once installed using
        # self.install_inside_library()
        constructor_location = th

    UppercaseTensorConstructor().install_inside_library()

    class ParameterConstructor(ObjectConstructor):
        # Step 1: Store the attribute name that this constructor is replacing
        constructor_name = "Parameter"

        # Step 2: Store a reference to the location on which this constructor currently lives.
        # This is also the location that this custom constructor will live once installed using
        # self.install_inside_library()
        constructor_location = nn

    ParameterConstructor().install_inside_library()

    assert th.tensor.syft == True

    x = th.tensor([1, 2, 3, 4])

    assert isinstance(x, th.Tensor)
    assert (x == th.Tensor([1, 2, 3, 4])).all()

    assert th.Tensor.syft == True

    x = th.Tensor([1, 2, 3, 4])

    assert isinstance(x, th.Tensor)
    assert (x == th.Tensor([1, 2, 3, 4])).all()

    assert nn.Parameter.syft == True

    x = th.tensor([1, 2, 3, 4.0])
    p = nn.Parameter(x)

    assert isinstance(x, th.Tensor)
    assert isinstance(p, nn.Parameter)
    assert (p.data == th.Tensor([1, 2, 3, 4])).all()
