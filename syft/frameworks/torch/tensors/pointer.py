from .abstract import AbstractTensor


class PointerTensor(AbstractTensor):
    def __init__(
        self, parent=None, location=None, id_at_location=None, register=None, owner=None, id=None
    ):
        self.location = location
        self.id_at_location = id_at_location
        self.owner = owner
        self.id = id

    def __str__(self):
        type_name = type(self).__name__
        return (
            f"["
            f"{type_name} - "
            f"id:{self.id} "
            f"owner:{self.owner.id} "
            f"loc:{self.location.id} "
            f"id@loc:{self.id_at_location}"
            f"]"
        )

    def __repr__(self):
        return self.__str__()

    def get(self, deregister_ptr=True):
        """Get back from a remote worker the chain this pointer
        is pointing at."""

        # if the pointer happens to be pointing to a local object,
        # just return that object (this is an edge case)
        if self.location == self.owner:
            tensor = self.owner.get_obj(self.id_at_location).child
        else:
            # get tensor from remote machine
            tensor = self.owner.request_obj(self.id_at_location, self.location)

        # Register the result
        assigned_id = self.id_at_location
        self.owner.register_obj(tensor, assigned_id)

        # Remove this pointer by default
        if deregister_ptr:
            self.owner.de_register_obj(self)

        return tensor
