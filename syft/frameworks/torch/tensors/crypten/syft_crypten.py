from crypten.mpc import MPCTensor

from syft.generic.frameworks.hook import hook_args
from syft.generic.tensor import AbstractTensor


class SyftCrypTensor(AbstractTensor):
    def __init__(
        self,
        owner=None,
        id=None,
        tensor: MPCTensor = None,
        tags: set = None,
        description: str = None,
    ):
        super().__init__(id=id, owner=owner, tags=tags, description=description)
        self.tensor = tensor

    def get_class_attributes(self):
        """
        Specify all the attributes need to build a wrapper correctly when returning a response,
        """
        return {"tensor": self.tensor}

    @property
    def data(self):
        return self

    @data.setter
    def data(self, new_data):
        self.child = new_data.child
        return self

    def get_plain_text(self, dst=None):
        """Decrypts the tensor."""
        return self.tensor.get_plain_text(dst=dst)


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(SyftCrypTensor)
