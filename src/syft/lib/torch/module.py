# third party
import torch as th

# syft relative
from ..generic import ObjectConstructor


class ModuleConstructor(ObjectConstructor):

    __name__ = "ModuleConstructor"

    # Step 1: Store the attribute name that this constructor is replacing
    constructor_name = "Module"

    # Step 2: Store a reference to the location on which this constructor currently lives.
    # This is also the location that this custom constructor will live once installed using
    # self.install_inside_library()
    constructor_location = th.nn

    original_type = th.nn.Module


# Step 3: create constructor and install it in the library
ModuleConstructor().install_inside_library()
