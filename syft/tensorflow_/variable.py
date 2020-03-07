from syft.generic import ObjectConstructor
from syft import check

import tensorflow as tf


class VariableConstructor(ObjectConstructor):
    # Step 1: Store the attribute name that this constructor is replacing
    constructor_name = "Variable"

    # Step 2: Store a reference to the location on which this constructor currently lives.
    # This is also the location that this custom constructor will live once installed using
    # self.install_inside_library()
    constructor_location = tf

    @check.type_hints
    def post_init(self, obj: object, *args, **kwargs):
        """Execute functionality after object has been created.

        This method executes functionality which can only be run after an object has been initailized. It is
        particularly useful for logic which registers the created object into an appropriate registry. It is
        also useful for adding arbitrary attributes to the object after initialization.

        Args:
            *args (tuple): the arguments being used to initialize the object
            **kwargs (dict): the kwarguments eeing used to initialize the object
        Returns:
            out (SyftObject): returns the underlying syft object.
        """

        # obj = self.assign_id(obj)

        return obj

# Step 3: create constructor and install it in the library
VariableConstructor().install_inside_library()
