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
