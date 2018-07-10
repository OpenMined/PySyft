"""Framework agnostic static utility functions."""
import functools

import json
import re
import torch

class PythonEncoder():
    """
        Encode python and torch objects to be JSON-able
        Torch objects are replaced by an id.
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
        if retrieve_tensorvar is not None:
            self.retrieve_tensorvar = retrieve_tensorvar
        if self.retrieve_tensorvar:
            return (self.python_encode(obj), self.found_tensorvar)
        else:
            return self.python_encode(obj)

    def python_encode(self, obj):
        if isinstance(obj, (int, float, str)) or obj is None:
            return obj
        elif isinstance(obj, self.tensorvar_types):
            if self.retrieve_tensorvar:
                self.found_tensorvar.append(obj)
            key = '__'+type(obj).__name__+'__'
            return { key: '_fl.{}'.format(obj.id) }
        elif isinstance(obj, (tuple, set, bytearray)):
            key = '__'+type(obj).__name__+'__'
            return {key:[self.python_encode(i) for i in obj]}
        elif isinstance(obj, list):
            return [self.python_encode(i) for i in obj]
        elif isinstance(obj, dict):
            return {
                k: self.python_encode(v)
                for k, v in obj.items()
            }
        else:
            try:
                return { '__eval__': str(obj) }
            except:
                print('Unknown type', type(obj))
            return None

class PythonJSONDecoder(json.JSONDecoder):
    """
        Decode JSON and replace python types when need
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
        pat = re.compile('__(.+)__')
        for key, obj in dct.items():
            try:
                obj_type = pat.search(key).group(1)
                if obj_type in map(lambda x: x.__name__, self.tensorvar_types):
                    pattern_var = re.compile('_fl.(.*)')
                    id = int(pattern_var.search(obj).group(1))
                    return self.worker.get_obj(id)
                elif obj_type in ('tuple', 'set', 'bytearray', 'eval'):
                    return eval(obj_type)(obj)
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

        # The partial() is used for partial function application which “freezes” some
        # portion of a function’s arguments and/or keywords resulting in a new object
        # with a simplified signature.
        return functools.partial(func, *args, **kwargs)
    return pass_args
