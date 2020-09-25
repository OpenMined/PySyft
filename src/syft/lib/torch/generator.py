# third party
import torch as th

# syft relative
from ..generic import ObjectConstructor


class GeneratorConstructor(ObjectConstructor):

    __name__ = "GeneratorConstructor"

    # Step 1: Store the attribute name that this constructor is replacing
    constructor_name = "Generator"

    # Step 2: Store a reference to the location on which this constructor currently lives.
    # This is also the location that this custom constructor will live once installed using
    constructor_location = th

    original_type = th.Generator


# Step 3: create constructor and install it in the library
GeneratorConstructor().install_inside_library()
