from syft.generic import ObjectConstructor

from torch import nn


class ParameterConstructor(ObjectConstructor):
    # Step 1: Store the attribute name that this constructor is replacing
    constructor_name = "Parameter"

    # Step 2: Store a reference to the location on which this constructor currently lives.
    # This is also the location that this custom constructor will live once installed using
    # self.install_inside_library()
    constructor_location = nn


# Step 3: create constructor and install it in the library
constructor = ParameterConstructor().install_inside_library()
