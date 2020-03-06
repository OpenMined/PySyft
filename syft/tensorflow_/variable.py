from syft.generic import ObjectConstructor

import tensorflow as tf


class VariableConstructor(ObjectConstructor):
    # Step 1: Store the attribute name that this constructor is replacing
    constructor_name = "Variable"

    # Step 2: Store a reference to the location on which this constructor currently lives.
    # This is also the location that this custom constructor will live once installed using
    # self.install_inside_library()
    constructor_location = tf


# Step 3: create constructor and install it in the library
VariableConstructor().install_inside_library()
