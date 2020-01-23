import syft as sy
from syft.generic.frameworks.hook import hook_args
from syft.generic.frameworks.overload import overloaded
from syft.generic.tensor import AbstractTensor


class LazyTensor(AbstractTensor):
    def __init__(self, owner=None, id=None, tags=None, description=None, verbose=False):
        """Initializes a LazyTensor.

        Args:
            owner (BaseWorker): An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id (str or int): An optional string or integer id of the LargePrecisionTensor.
            tags (list): list of tags for searching.
            description (str): a description of this tensor.
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description)
        self.todos = list()

        hook = sy.hook
        if not hasattr(hook, "hooked_lazy_tensor"):
            tensor_type = hook.torch.Tensor

            # Use a pre-defined list to select the methods to overload
            for attr in hook.to_auto_overload[tensor_type]:
                if attr not in dir(LazyTensor):
                    new_method = get_lazy_method(attr)
                    setattr(LazyTensor, attr, new_method)

            hook.hooked_lazy_tensor = True

    def execute(self):
        result = self
        for method_name, args, kwargs in self.todos:
            result = self.child.__getattribute__(method_name)(*args, **kwargs)
        return result


def get_lazy_method(attr):
    def new_method(self, *args, **kwargs):
        self.todos.append((attr, args, kwargs))
        return self

    return new_method


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(LazyTensor)
