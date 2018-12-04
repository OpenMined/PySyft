import random

from syft.frameworks.torch.tensors import PointerTensor


class TorchTensor:
    """
    This tensor is simply a more convenient way to add custom functions to
    all Torch tensor types.
    """

    def __init__(self):
        self.id = None
        self.owner = None

    def create_pointer(
        self, location=None, id_at_location=None, register=False, owner=None, ptr_id=None
    ):

        if owner is None:
            owner = self.owner

        if location is None:
            location = self.owner.id

        owner = self.owner.get_worker(owner)
        location = self.owner.get_worker(location)

        if id_at_location is None:
            id_at_location = self.id

        if ptr_id is None:
            if location.id != self.owner.id:
                ptr_id = self.id
            else:
                ptr_id = int(10e10 * random.random())

        # previous_pointer = owner.get_pointer_to(location, id_at_location)
        previous_pointer = None

        if previous_pointer is None:
            ptr = PointerTensor(
                parent=self,
                location=location,
                id_at_location=id_at_location,
                register=register,
                owner=owner,
                id=ptr_id,
            )

        return ptr
