from ..generic import ObjectConstructor

import numpy as np


class NdArrayConstructor(ObjectConstructor):

    # Step 1: Store the attribute name that this constructor is replacing
    constructor_name = "ndarray"

    # Step 2: Store a reference to the location on which this constructor currently lives.
    # This is also the location that this custom constructor will live once installed using
    # self.install_inside_library()
    constructor_location = np

    original_type = np.ndarray


# Step 3: create constructor and install it in the library
NdArrayConstructor().install_inside_library()
