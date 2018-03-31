import os
import json
import re

from pathlib import Path

from . import utils
import torch


# Helpers for HookService and TorchService
def check_workers(self, workers):
    if type(workers) is str:
        workers = [workers]
    elif not hasattr(workers, '__iter__'):
        raise TypeError(
            """Can only send {} to a string worker ID or an iterable of
            string worker IDs, not {}""".format(self.__name__, type(owners))
            )
    return workers


def get_tensorvars(self, command):
    args = command['args']
    kwargs = command['kwargs']
    arg_types = command['arg_types']
    kwarg_types = command['kwarg_types']
    tensorvar_args = [args[i] for i in range(len(args)) if arg_types[i] in self.tensorvar_types_strs]
    tensorvar_kwvals = [kwargs[i][1] for i in range(len(kwargs)) if kwarg_types[i] in self.tensorvar_types_strs]
    return tensorvar_args + tensorvar_kwvals


def check_remote(tensorvars):
    return any([tensorvar.is_pointer for tensorvar in tensorvars])


def get_owners(tensorvars):
    owners = list(set([owner
        for tensorvar in tensorvars
        for owner in tensorvar.owners]))
    multiple_owners = len(owners) > 1
    return multiple_owners, owners


def replace_tensorvar(x):
    if hasattr(torch, 'old_is_tensor'):
        check = torch.old_is_tensor
    else:
        check = torch.is_tensor
    try:
        if check(x) or isinstance(x, torch.autograd.Variable):
            return '_fl.{}'.format(x.id)
        else:
            [replace_tensorvar(i) for i in x]
    except (AttributeError, TypeError):
        return x


def replace_in_command(command_msg):
    command_msg['args'] = map_tuple(
        None, command_msg['args'], replace_tensorvar)
    command_msg['kwargs'] = map_dict(
        None, command_msg['kwargs'], replace_tensorvar)
    try:
        command_msg['self'] = replace_tensorvar(command_msg['self'])
    except KeyError:
        pass
    return command_msg

# Client needs to identify a tensor before sending commands that use it
def id_tensorvar(x):
    pat = re.compile('_fl.(.*)')
    try:
        if isinstance(x, str):
            return int(pat.search(x).group(1))
        else:
            return [id_tensorvar(i) for i in x]
    except AttributeError:
        return x


# Safety checks for serializing and deserializing torch objects
# Desperately needs stress testing before going out in the wild
map_torch_type = {
    'torch.FloatTensor':torch.FloatTensor,
    'torch.DoubleTensor':torch.DoubleTensor,
    'torch.HalfTensor':torch.HalfTensor,
    'torch.ByteTensor':torch.ByteTensor,
    'torch.CharTensor':torch.CharTensor,
    'torch.ShortTensor':torch.ShortTensor,
    'torch.IntTensor':torch.IntTensor,
    'torch.LongTensor':torch.LongTensor,
    'torch.autograd.variable.Variable':torch.autograd.variable.Variable,
    'torch.nn.parameter.Parameter':torch.nn.parameter.Parameter
}


def types_guard(obj_type):
    return map_torch_type[obj_type]


def tensor_contents_guard(contents):
    # TODO: check to make sure the incoming list isn't dangerous to use for
    #       constructing a tensor (likely non-trivial)
    return contents


def command_guard(command, allowed):
    if command not in allowed:
        raise RuntimeError(
            'Command "{}" is not a supported Torch operation.'.format(command))
    return command


# Worker needs to retrieve tensor by ID before computing with it
def retrieve_tensor(self, x):
    try:
        return self.worker.objects[id_tensorvar(x)]
    except TypeError:
        try:
            return [self.worker.objects[i] for i in id_tensorvar(x)]
        except TypeError:
            return x
    except KeyError:
        return x


def map_tuple(service, args, func):
    if service:
        return tuple(func(service, x) for x in args)
    else:
        return tuple(func(x) for x in args)


def map_dict(service, kwargs, func):
    if service:
        return {key:func(service, val) for key, val in kwargs.items()}
    else:
        return {key:func(val) for key, val in kwargs.items()}


def hook_tensor_ser(service_self, tensor_type):
    def ser(self, include_data=True):
        """Serializes a {} object to JSON.""".format(tensor_type)
        tensor_msg = {}
        tensor_msg['torch_type'] = self.type()
        if (include_data):
            tensor_msg['data'] = self.tolist()
        tensor_msg['id'] = self.id
        tensor_msg['owners'] = self.owners
        return json.dumps(tensor_msg)

    tensor_type.ser = ser


def hook_var_ser(service_self):
    # TODO
    pass
