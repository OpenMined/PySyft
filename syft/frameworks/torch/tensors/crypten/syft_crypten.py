from syft.generic.frameworks.hook import hook_args
from syft.generic.tensor import AbstractTensor

from syft.generic.frameworks.hook.trace import tracer


class SyftCrypTensor(AbstractTensor):
    def __init__(
        self,
        owner=None,
        id=None,
        tensor=None,
        tags: set = None,
        description: str = None,
    ):
        super().__init__(id=id, owner=owner, tags=tags, description=description)
        self.tensor = tensor

    def get_class_attributes(self):
        """
        Specify all the attributes need to build a wrapper correctly when returning a response,
        """
        # TODO: what we should return specific for this one?
        return {}

    @property
    def data(self):
        return self

    @data.setter
    def data(self, new_data):
        self.tensor = new_data.child
        return self

    @tracer(method_name="add")
    def add(self, other):
        return SyftCrypTensor(tensor=self.tensor)

    __add__ = add
    __radd__ = add

    @tracer(method_name="get_plain_text")
    def get_plain_text(self, dst=None):
        return SyftCrypTensor(tensor=self.tensor)


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(SyftCrypTensor)
