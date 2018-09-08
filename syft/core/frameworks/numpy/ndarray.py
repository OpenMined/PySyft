import json
import random
import numpy as np
import syft as sy

from .encode import NumpyEncoder
from ..torch import utils
import torch

class abstractarray(np.ndarray):

    def __new__(cls, input_array, id=None, owner=None):

        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)

        obj = obj.init(input_array, id, owner)

        return obj

    def init(self, input_array, id, owner):

        # add the new attribute to the created instance
        if (id is None):
            id = random.randint(0, 1e10)
        self.id = id

        if (owner is None):
            # cache the local_worker object locally which we will
            # use for all outgoing communications
            if not hasattr(sy, 'local_worker'):
                hook = sy.TorchHook()
            owner = sy.local_worker

        self.owner = owner

        # Finally, we must return the newly created object:
        return self

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.info = getattr(obj, 'info', None)

    @classmethod
    def handle_call(cls, command, owner):

        attr = command['command']
        args = command['args']
        kwargs = command['kwargs']
        has_self = command['has_self']

        if has_self and cls.is_overloaded_method(attr):
            self_ = command['self']
            result = getattr(self_, attr)(*args, **kwargs)
            return result
        elif not has_self and cls.is_overloaded_function(attr):
            overload_function = cls.overloaded_functions.get(attr)
            result = overload_function(*args, **kwargs)
            return result
        else:
            print("do something else")

    @classmethod
    def is_overloaded_method(cls, attr):
        """
        State if a function name corresponds to a Syft Tensor method which
        overloads a torch method
        """
        # exclude = ['on', '__init__', 'native___init__', '__repr__', '__str__', 'create_pointer',
        #            'ser', 'deser', 'handle_call']
        # if attr in exclude:
        #     return False
        # if hasattr(getattr(cls, attr), '__module__') \
        #         and getattr(cls, attr).__module__ == 'syft.core.frameworks.numpy.array':
        #     return True
        # return False
        return True # TODO: finish this

    @classmethod
    def is_overloaded_function(cls, attr):
        """
        State if a function name corresponds to an overloaded function by the Syft
        tensor, which declared the corresponding overloading function in
        cls.overload_functions
        """
        return True #TODO: finish this
        # attr = attr.split('.')[-1]
        # overloaded_functions = [
        #     func for func in dir(cls.overload_functions)
        #     if re.match(r'__(.*)__', func) is None
        #        and func != 'get'
        # ]
        # return attr in overloaded_functions


class array(abstractarray):

    def ser(self, to_json=False):
        if (to_json):
            return json.dumps(self, cls=NumpyEncoder)
        else:
            out = {}
            out['type'] = "numpy.array"
            out['id'] = self.id
            out['data'] = self.tolist()
            return out

    def send(self, worker, ptr_id=None):

        if isinstance(worker, (int, str)):
            worker = self.owner.get_worker(worker)

        if ptr_id is None:
            ptr_id = random.randint(0, 10e10)

        obj_id = self.id

        self.owner.send_obj(self, obj_id, worker)

        ptr = self.create_pointer(id=ptr_id,
                                  location=worker,
                                  id_at_location=obj_id)
        return ptr

    def create_pointer(self, id, location, id_at_location):

        return array_ptr(None,
                         id=id,
                         owner=self.owner,
                         location=location,
                         id_at_location=id_at_location)

    def torch(self):
        return torch.FloatTensor(self)


class array_ptr(abstractarray):

    def __new__(cls, _,
                id=None,
                owner=None,
                location=None,
                id_at_location=None):

        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(["data is remote"]).view(cls)

        obj = obj.init(["data is remote"], id, owner)

        obj.location = location
        obj.id_at_location = id_at_location

        return obj

    def get(self, deregister_ptr=True):
        """
            Get a chain back from a remote worker that his pointer is pointing at
        """

        # Remove this pointer - TODO: call deregister function instead of doing it by hand
        if (deregister_ptr):
            self.owner.rm_obj(self.id)

        # if the pointer happens to be pointing to a local object,
        # just return that object (this is an edge case)
        if self.location == self.owner:
            return self.owner.get_obj(self.id_at_location)

        obj = self.owner.request_obj(self.id_at_location, self.location)
        obj.id = self.id
        self.owner.register(obj)
        return obj

    def ser(self, to_json=False):
        if (to_json):
            return json.dumps(self.ser(True))
        else:
            out = {}
            out['type'] = "numpy.array_ptr"
            out['id'] = self.id
            out['data'] = self.tolist()
            out['location'] = self.location.id
            out['id_at_location'] = self.id_at_location
            return out

    def torch(self):
        cmd, locations, owners = utils.compile_command(attr="torch",
                                                       args={},
                                                       kwargs={},
                                                       has_self=True,
                                                       self=self)

        return self.owner.send_command(recipient=self.location,
                                       message=cmd,
                                       framework="numpy")





