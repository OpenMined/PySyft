import random
import torch

class BaseService(object):

    def __init__(self, worker):
        self.worker = worker
        self.api = self.worker.api

        ## Torch-specific
        self.tensor_types = [torch.FloatTensor,
                torch.DoubleTensor,
                torch.HalfTensor,
                torch.ByteTensor,
                torch.CharTensor,
                torch.ShortTensor,
                torch.IntTensor,
                torch.LongTensor]
        self.var_types = [torch.autograd.variable.Variable, torch.nn.Parameter]
        self.tensorvar_types = self.tensor_types + [torch.autograd.variable.Variable]
        self.tensorvar_types_strs = [x.__name__ for x in self.tensorvar_types]

        # Any commands that don't appear in the following two lists
        # will not execute
        self.torch_funcs = dir(torch)
        # TODO: Consider changing the following to a dictionary with each
        #       type in tensorvar_types mapped to a list of appropriate
        #       methods
        self.tensorvar_methods = list(
            set(
                [method
                    for tensorvar in self.tensorvar_types
                    for method in dir(tensorvar)]
                )
            )


    def register_object_(self, obj, **kwargs):
        """
        Registers an object with the current worker node.
        Selects an id for the object, assigns a list of owners,
        and establishes whether it's a pointer or not.

        Args:
            obj: a Torch instance, e.g. Tensor or Variable
        Default kwargs:
            id: random integer between 0 and 1e10
            owners: list containing local worker's IPFS id
            is_pointer: False
        """
        # TODO: Assign default id more intelligently (low priority)
        #       Consider popping id from long list of unique integers
        keys = kwargs.keys()
        obj.id = (kwargs['id']
            if 'id' in keys
            else random.randint(0, 1e10))
        obj.owners = (kwargs['owners']
            if 'owners' in keys
            else [self.worker.id])
        obj.is_pointer = (kwargs['is_pointer']
            if 'is_pointer' in keys
            else False)
        mal_points_away = obj.is_pointer and self.worker.id in obj.owners
        # The following was meant to assure that we didn't try to
        # register objects we didn't have. We end up needing to register
        # objects with non-local owners on the worker side before sending
        # things off, so it's been relaxed.  Consider using a 'strict'
        # kwarg for strict checking of this stuff
        mal_points_here = False
        # mal_points_here = not obj.is_pointer and self.worker.id not in obj.owners
        if mal_points_away or mal_points_here:
            raise RuntimeError(
                'Invalid registry: is_pointer is {} but owners is {}'.format(
                    obj.is_pointer, obj.owners))
        self.worker.objects[obj.id] = obj
        return obj
