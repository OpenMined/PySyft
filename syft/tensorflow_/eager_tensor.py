from syft.generic import ObjectConstructor
from syft import check

from tensorflow.python.framework import ops

class EagerTensorConstructor(ObjectConstructor):
    # Step 1: Store the attribute name that this constructor is replacing
    constructor_name = "EagerTensor"

    # Step 2: Store a reference to the location on which this constructor currently lives.
    # This is also the location that this custom constructor will live once installed using
    # self.install_inside_library()
    constructor_location = ops


# Step 3: create constructor and install it in the library
EagerTensorConstructor().install_inside_library()
