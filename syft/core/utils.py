"""Framework agnostic static utility functions."""
import json
import re
import types
import functools
import logging
import torch
import syft
import syft as sy

from .frameworks.torch import encode

def is_in_place_method(attr):
    """
    Determines if the method is in-place (ie modifies the self)
    TODO: Can you do better?
    """
    pat = re.compile('__(.+)__')
    return pat.search(attr) is None and attr[-1] == '_'

class PythonEncoder():
    """
        Encode python and torch objects to be JSON-able
        In particular, (hooked) Torch objects are replaced by their id.
        Note that a python object is returned, not JSON.
    """
    def __init__(self, retrieve_tensorvar=False):
        self.retrieve_tensorvar = retrieve_tensorvar
        self.found_tensorvar = []
        self.tensorvar_types = tuple([torch.autograd.Variable,
                                     torch.nn.Parameter,
                                     torch.FloatTensor,
                                     torch.DoubleTensor,
                                     torch.HalfTensor,
                                     torch.ByteTensor,
                                     torch.CharTensor,
                                     torch.ShortTensor,
                                     torch.IntTensor,
                                     torch.LongTensor])

    def encode(self, obj, retrieve_tensorvar=None):
        """
            Performs encoding, and retrieves if requested all the tensors and
            Variables found
        """
        if retrieve_tensorvar is not None:
            self.retrieve_tensorvar = retrieve_tensorvar
        if self.retrieve_tensorvar:
            return (self.python_encode(obj), self.found_tensorvar)
        else:
            return self.python_encode(obj)

    def python_encode(self, obj):
        # Case of basic types
        if isinstance(obj, (int, float, str)) or obj is None:
            return obj
        # Tensors and Variable encoded with their id
        elif isinstance(obj, self.tensorvar_types):
            if self.retrieve_tensorvar:
                self.found_tensorvar.append(obj)
            key = '__'+type(obj).__name__+'__'
            return { key: '_fl.{}'.format(obj.id) }
        # Lists
        elif isinstance(obj, list):
            return [self.python_encode(i) for i in obj]
        # Iterables non json-serializable
        elif isinstance(obj, (tuple, set, bytearray, range)):
            key = '__'+type(obj).__name__+'__'
            return {key:[self.python_encode(i) for i in obj]}
        # Slice
        elif isinstance(obj, slice):
            key = '__'+type(obj).__name__+'__'
            return { key: { 'args': [obj.start, obj.stop, obj.step]}}
        # Dict
        elif isinstance(obj, dict):
            return {
                k: self.python_encode(v)
                for k, v in obj.items()
            }
        # Generator (transformed to list)
        elif isinstance(obj, types.GeneratorType):
            logging.warning("Generator args can't be transmitted")
            return []
        # Else log the error
        else:
            raise ValueError('Unhandled type', type(obj))

class PythonJSONDecoder(json.JSONDecoder):
    """
        Decode JSON and reinsert python types when needed
        Retrieve Torch objects replaced by their id
    """
    def __init__(self, worker, *args, **kwargs):
        super(PythonJSONDecoder, self).__init__(*args,
            object_hook=self.custom_obj_hook, **kwargs)
        self.worker = worker
        self.tensorvar_types = tuple([torch.autograd.Variable,
                                     torch.nn.Parameter,
                                     torch.FloatTensor,
                                     torch.DoubleTensor,
                                     torch.HalfTensor,
                                     torch.ByteTensor,
                                     torch.CharTensor,
                                     torch.ShortTensor,
                                     torch.IntTensor,
                                     torch.LongTensor])

    def custom_obj_hook(self, dct):
        """
            Is called on every dict found. We check if some keys correspond
            to special keywords referring to a type we need to re-cast
            (e.g. tuple, or torch Variable).
            Note that in the case we have such a keyword, we will have created
            at encoding a dict with a single key value pair, so the return if
            the for loop is valid.
        """
        pat = re.compile('__(.+)__')
        for key, obj in dct.items():
            try:
                obj_type = pat.search(key).group(1)
                # Case of a tensor or a Variable
                if obj_type in map(lambda x: x.__name__, self.tensorvar_types):
                    pattern_var = re.compile('_fl.(.*)')
                    id = int(pattern_var.search(obj).group(1))
                    return self.worker.get_obj(id)
                # Case of a iter type non json serializable
                elif obj_type in ('tuple', 'set', 'bytearray', 'range'):
                    return eval(obj_type)(obj)
                # Case of a slice
                elif obj_type == 'slice':
                    return slice(*obj['args'])
                else:
                    return obj
            except AttributeError:
                pass
        return dct

def map_tuple(hook, args, func):
    if hook:
        return tuple(func(hook, x) for x in args)
    else:
        return tuple(func(x) for x in args)


def map_dict(hook, kwargs, func):
    if hook:
        return {key: func(hook, val) for key, val in kwargs.items()}
    else:
        return {key: func(val) for key, val in kwargs.items()}


def pass_method_args(method):
    """Wrapper gathering partialmethod object from method call."""

    @functools.wraps(method)
    def pass_args(*args, **kwargs):
        return functools.partialmethod(method, *args, **kwargs)

    return pass_args


def pass_func_args(func):
    """Wrapper gathering partial object from function call."""

    @functools.wraps(func)
    def pass_args(*args, **kwargs):
        # Return a new partial object which when called will behave like func called with the
        # positional arguments args and keyword arguments keywords. If more arguments are
        # supplied to the call, they are appended to args. If additional keyword arguments
        # are supplied, they extend and override keywords.
        # The partial() is used for partial function application which "freezes" some
        # portion of a function's arguments and/or keywords resulting in a new object
        # with a simplified signature.
        return functools.partial(func, *args, **kwargs)

    return pass_args
