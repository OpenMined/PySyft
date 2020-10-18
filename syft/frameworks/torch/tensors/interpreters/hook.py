from syft.generic.frameworks.hook import hook_args
from syft.generic.abstract.tensor import AbstractTensor


class HookedTensor(AbstractTensor):
    """HookedTensor is an abstraction which should not be used directly on its own. Its purpose
    is only to allow other tensors to extend it so that they automatically have all of the Torch
    method hooked without having to add it to the hook.py file.
    """

    def __init__(self, owner=None, id=None, tags=None, description=None, verbose=False):
        """Initializes a HookedTensor.

        Args:
            owner (BaseWorker): An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id (str or int): An optional string or integer id of the LargePrecisionTensor.
            tags (list): list of tags for searching.
            description (str): a description of this tensor.
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description)
        self.verbose = verbose


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(HookedTensor)
