# third party
import torch as th

# syft relative
from ..generic import ObjectConstructor


class DeviceConstructor(ObjectConstructor):

    __name__ = "DeviceConstructor"

    # Step 1: Store the attribute name that this constructor is replacing
    constructor_name = "device"

    # Step 2: Store a reference to the location on which this constructor currently lives.
    # This is also the location that this custom constructor will live once installed using
    constructor_location = th

    original_type = th.device


# Step 3: create constructor and install it in the library
DeviceConstructor().install_inside_library()
