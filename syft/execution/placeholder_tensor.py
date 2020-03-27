from syft.generic.frameworks.hook import hook_args
from syft.generic.tensor import AbstractTensor


class PlaceholderTensor(AbstractTensor):
    pass


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(PlaceholderTensor)
