"""Framework agnostic static utility functions."""
import json
import re
import types
import functools
import logging
from copy import deepcopy

import torch
import syft
import syft as sy


def encode(message, retrieve_pointers=False, private_local=True):
    """
    Help function to call easy the PythonEncoder
    :param message:
    :param retrieve_pointers: If true, return a list of all the _PointerTensor
    :param private_local: If true, ask to hide all the sensitive data (ie keep all the
    metadata like the structure of the chain)
    :return: The message encoded in a json-able dict
    """
    encoder = PythonEncoder()
    response = encoder.encode(message,
                              retrieve_pointers=retrieve_pointers,
                              private_local=private_local)
    return response


class PythonEncoder:
    """
        Encode python and torch objects to be JSON-able
        In particular, (hooked) Torch objects are replaced by their id.
        Note that a python object is returned, not JSON.
    """
    def __init__(self):
        self.retrieve_pointers = False
        self.found_pointers = []
        self.found_next_child_types = []
        self.tensorvar_types = tuple(torch.tensorvar_types)

    def encode(self, obj, retrieve_pointers=False, private_local=True):
        """
            Performs encoding, and retrieves if requested all the tensors and
            Variables found
        """
        self.retrieve_pointers = retrieve_pointers

        serialized_obj = self.python_encode(obj, private_local)

        serialized_msg = {'obj': serialized_obj}
        # Give instruction to the decoder, should he acquire the tensor or register them
        if private_local:  # It's private, you can't access directly the data, so you suscribe to it with a pointer
            serialized_msg['mode'] = 'subscribe'
        else:  # It's public, you can acquire the data directly
            serialized_msg['mode'] = 'acquire'

        response = [serialized_msg]
        if self.retrieve_pointers:
            response.append(self.found_pointers)

        if len(response) == 1:
            return response[0]
        else:
            return tuple(response)

    def python_encode(self, obj, private_local):
        # Case of basic types
        if isinstance(obj, (int, float, str)) or obj is None:
            return obj
        # Tensors and Variable encoded with their id
        elif is_tensor(obj) or is_variable(obj):
            tail_object = find_tail_of_chain(obj)
            if self.retrieve_pointers and isinstance(tail_object, sy._PointerTensor):
                self.found_pointers.append(tail_object)
            return obj.ser(private=private_local)
        # sy._SyftTensor (Pointer, Local) [Note: shouldn't be called on regular chain with end=tensorvar]
        elif is_syft_tensor(obj):
            tail_object = find_tail_of_chain(obj)
            if self.retrieve_pointers and isinstance(tail_object, sy._PointerTensor):
                self.found_pointers.append(tail_object)
            return obj.ser(private=private_local)
            # raise TypeError('Syft Tensors should always be wrapped with a Torch Tensor')
        # List
        elif isinstance(obj, list):
            return [self.python_encode(i, private_local) for i in obj]
        # Iterables non json-serializable
        elif isinstance(obj, (tuple, set, bytearray, range)):
            key = '__'+type(obj).__name__+'__'
            return {key: [self.python_encode(i, private_local) for i in obj]}
        # Slice
        elif isinstance(obj, slice):
            key = '__'+type(obj).__name__+'__'
            return {key: {'args': [obj.start, obj.stop, obj.step]}}
        # Dict
        elif isinstance(obj, dict):
            return {
                k: self.python_encode(v, private_local)
                for k, v in obj.items()
            }
        # Generator (transformed to list)
        elif isinstance(obj, types.GeneratorType):
            logging.warning("Generator args can't be transmitted")
            return []
        # worker
        elif isinstance(obj, (sy.SocketWorker, sy.VirtualWorker)):
            return {'__worker__': obj.id}
        # Else log the error
        else:
            raise ValueError('Unhandled type', type(obj))


def decode(message, worker, acquire=None):
    """
    Determine whether the mode should be 'acquire' or 'suscribe', and
    Decode the message with this policy
    1. mode='acquire' means that every SyftTensor seen will be copied to the worker
       mode='suscribe' means that we point to every SyftTensor to keep track of it
    2. The mode is an indication of how to properly decode the message. Even if the
       worker is malicious it won't be able to get any sensitive info by following
       an adversial mode, since private data is remove at encoding. This indication
       is need in cases where you get _Pointer for instance, should you store it or
       point at it?
    3. This function basically extract the mode of the message, and send the message
       to be decoded
    :param message: The message to decode
    :param worker: The worker performing the decode operation
    :return: The message decoded
    """
    decoder = PythonJSONDecoder(worker=worker)

    # Handle when the message is a bytestring
    if isinstance(message, bytes):
        message = message.decode('utf-8')

    dict_message = json.loads(message)

    # If acquire is speficied, then know how we want to decode, and implicitely
    # We want to decode everything of the message
    if acquire is not None:
        return decoder.python_decode(message)


    # TODO It would be good to have a standardized place to put the 'mode' argument
    # Depending of the structure of the message, the mode argument is not at the same place
    if 'message' in dict_message:
        message = dict_message['message']
    else:
        message = dict_message

    if isinstance(message, dict) and 'mode' in message:
        decoder.acquire = True if message['mode'] == 'acquire' else False
        message = decoder.python_decode(message['obj'])

    if 'message' in dict_message:
        dict_message['message'] = message
    else:
        dict_message = message
    return dict_message


class PythonJSONDecoder:
    """
    Decode JSON and reinsert python types when needed, as well as
    SyftTensors

    For example, the message to decode could be:
    {'__FloatTensor__': {
        'type': 'syFloatTensor',
        'torch_type': 'syft.FloatTensor',
        'data': [[1.0, 2.0], [3.0, 4.0]],
        'child': {
            '___LocalTensor__': {
                'owner': 0,
                'id': 1000,
                'torch_type': 'syft.FloatTensor'
    }}}}
    """
    def __init__(self, worker, acquire=False):
        self.worker = worker
        self.tensor_types = tuple(torch.tensor_types)
        self.acquire = acquire

    def python_decode(self, dct):
        """
            Is called on every dict found. We check if some keys correspond
            to special keywords referring to a type we need to re-cast
            (e.g. tuple, or torch Variable).

        """
        # TODO: Stop with this prior that data should be a dict, which require the following lines to fix with real cases
        if isinstance(dct, (int, str)):
            return dct
        if isinstance(dct, (list, )):
            return [self.python_decode(o) for o in dct]
        if dct is None:
            return None
        if not isinstance(dct, dict):
            raise TypeError('Type not handled', dct)

        pat = re.compile('__(.+)__')
        for key, obj in dct.items():
            if pat.search(key) is not None:
                obj_type = pat.search(key).group(1)
                # Case of a tensor
                if is_tensor(obj_type):
                    o = eval('sy.'+obj_type).deser({key: obj}, self.worker, self.acquire)
                    return o
                # Case of a Variable
                elif is_variable(obj_type):
                    return sy.Variable.deser({key: obj}, self.worker, self.acquire)
                # Case of a Syft tensor
                elif is_syft_tensor(obj_type):
                    return sy._SyftTensor({key: obj}, self.worker, self.acquire)
                # Case of a iter type non json serializable
                elif obj_type in ('tuple', 'set', 'bytearray', 'range'):
                    return eval(obj_type)([self.python_decode(o) for o in obj])
                # Case of a slice
                elif obj_type == 'slice':
                    return slice(*obj['args'])
                # Case of a worker
                elif obj_type == 'worker':
                    return self.worker.get_worker(obj)
                else:
                    raise TypeError('The special object type', obj_type, 'is not supported')
            else:
                dct[key] = self.python_decode(obj)
        return dct


def extract_type_and_obj(dct):
    pat = re.compile('__(.+)__')
    for key, obj in dct.items():
        if pat.search(key) is not None:
            obj_type = pat.search(key).group(1)
            return obj_type, obj
        else:
            raise TypeError('Key', key, 'is not recognized.')


def chain_print(obj, display=True):
    types = [obj.__class__.__name__]
    i = 0
    while hasattr(obj, 'child'):
        types.append(obj.child.__class__.__name__)
        if isinstance(obj.child, (sy._LocalTensor, sy._PointerTensor)):
            break
        obj = obj.child
        i += 1
        if i >= 12:
            types.append('(...)')
            break
    if display:
        print(' > '.join(types))
    else:
        return ' > '.join(types)


def get_child_command(obj, child_types=[]):
    """
    Analyse a Python object (generally dict with a command and arguments,
    And for all tensors, variables, syft tensors, replace with their first
    child and retrieve its type
    :param obj:
    :param child_types:
    :return:
    """
    # Torch tensor or variable, or sy._SyftTensor
    if is_tensor(obj) or is_variable(obj) or is_syft_tensor(obj):
        return obj.child, [type(obj.child)]
    # List or iterables which could contain tensors
    elif isinstance(obj, (list, tuple, set, bytearray, range)):
        children = []
        types = []
        for o in obj:
            c, t = get_child_command(o, child_types)
            children.append(c)
            types += t
        return type(obj)(children), types
    # Dict
    elif isinstance(obj, dict):
        children = {}
        types = []
        for k, o in obj.items():
            c, t = get_child_command(o, child_types)
            children[k] = c
            types += t
        return children, types
    else:
        return obj, []


def prepare_child_command(command, replace_tensorvar_with_child=False):

    next_command, next_child_types = get_child_command(command)

    # Check that the next child type of all tensorvar is the same
    if len(next_child_types) == 0:
        ref_child_type = sy._LocalTensor
    else:
        if all([child_type in torch.tensor_types for child_type in next_child_types]):
            ref_child_type = sy.FloatTensor
        else:
            ref_child_type = next_child_types[0]
            for next_child_type in next_child_types:
                if next_child_type != ref_child_type:
                    raise NotImplementedError('All arguments should share the same child type.', next_child_types)

    if replace_tensorvar_with_child:
        return next_command, ref_child_type
    else:
        return command, ref_child_type


def enforce_owner(obj, owner):
    """
        Reassign every elements of the chain to an owner (in a Virtual worker context)
    """
    obj.owner = owner
    if not is_tensor(obj.child):
        enforce_owner(obj.child, owner)


def wrap_command(obj):
    """
        To a Syft command, add a torch wrapper
    """
    # Torch tensor or variable, or sy._SyftTensor
    if is_tensor(obj) or is_variable(obj):
        raise TypeError('Expecting syft tensors')
    elif is_syft_tensor(obj):
        wrapper = eval(obj.torch_type)()
        wrapper.child = obj
        obj.parent = wrapper
        return wrapper
    # List or iterables which could contain tensors
    elif isinstance(obj, (list, tuple, set, bytearray, range)):
        wrappers = []
        for o in obj:
            wrapper = wrap_command(o)
            wrappers.append(wrapper)
        return type(obj)(wrappers)
    # Dict
    elif isinstance(obj, dict):
        wrappers = {}
        for k, o in obj.items():
            wrapper = wrap_command(o)
            wrappers[k] = wrapper
        return wrappers
    else:
        return obj


def compile_command(attr, args, kwargs, has_self=False, self=None):
    command = {
        'command': attr,
        'has_self': has_self,
        'args': args,
        'kwargs': kwargs
    }
    if has_self:
        command['self'] = self

    command, pointers = encode(command, retrieve_pointers=True)

    # Get information about the location and owner of the pointers
    locations = []
    owners = []
    for pointer in pointers:
        locations.append(pointer.location)
        owners.append(pointer.owner)
    locations = list(set(locations))
    owners = list(set(owners))

    if len(locations) > 1:
        raise NotImplementedError('All pointers should point to the same worker')
    if len(owners) > 1:
        raise NotImplementedError('All pointers should share the same owner.')

    return command, locations, owners


def assert_has_only_torch_tensorvars(obj):
    """
    A check function that an object has only torch Tensors or Variable
    at his 'roots'
    Is useful for development.
    """
    if isinstance(obj, (int, float, str)):
        return True
    elif is_tensor(obj):
        return True
    elif is_variable(obj):
        return True
    elif isinstance(obj, (list, tuple)):
        rep = [assert_has_only_torch_tensorvars(o) for o in obj]
        return all(rep)
    elif isinstance(obj, dict):
        rep = [assert_has_only_torch_tensorvars(o) for o in obj.values()]
        return all(rep)
    elif isinstance(obj, slice):
        return True
    elif obj is None:
        return True
    else:
        assert False, ('Obj is not tensorvar', type(obj))


def assert_has_only_syft_tensors(obj):
    if isinstance(obj, (int, float, str)):
        return True
    elif issubclass(obj.__class__, sy._SyftTensor):
        return True
    elif isinstance(obj, (list, tuple)):
        rep = [assert_has_only_syft_tensors(o) for o in obj]
        return all(rep)
    elif isinstance(obj, dict):
        rep = [assert_has_only_syft_tensors(o) for o in obj.values()]
        return all(rep)
    elif isinstance(obj, slice):
        return True
    else:
        assert False, ('Obj is not syft tensor', type(obj))


def get_syft_chain(obj):
    """
    Return the chain of syft object types
    """
    next_node = obj.child
    syft_chain = []
    while next_node is not None and not (is_tensor(next_node) or is_variable(next_node)):
        syft_chain.append(type(next_node))
        next_node = next_node.child

    return syft_chain


def assert_is_chain_well_formed(obj, downward=True, start_id=None, start_type=None, end_chain=None):
    """
    Performs an analysis that a chain is correctly built:
    A local chain should be something that terminates with a _LocalTensor,
    e.g. `FloatTensor -> _LogTensor -> _LocalTensor`. In this setting the
    child and parent are obvious on the middle elements, and on the edge
    there is a "loop", _LocalTensor.child = FloatTensor and FloatTensor.parent
    = _LocalTensor.
    A non-local chain in something that terminates with a _PointerTensor for
    instance, e.g. `FloatTensor -> _LogTensor -> _PointerTensor`. In this case
    the edges of the chains shoudn't be connected because it makes no sense,
    and the remote protocol send/get/etc. is the equivalent of the child
    attribute which is missing for the pointer.

    In practice, we also check for unexpected loops.
    """
    # Is only executed at the first iteration
    if start_id is None:
        start_id = obj.id
        start_type = type(obj)
        if is_variable(obj):
            # We don't care about the return, as the main object has to return true anyway
            # All we care is about Exception raising
            assert_is_chain_well_formed(obj.data)
    else:
        if start_id == obj.id and start_type == type(obj):
            raise StopIteration('The chain looped downward=', downward,'on id', obj.child.id, 'with obj', obj.child)
    if end_chain is not None \
      and (is_variable(obj) or is_tensor(obj)):
        if isinstance(end_chain, sy._PointerTensor):
            assert obj.parent is None, "Tensorvar linked to Pointer should not have a parent"
            assert end_chain.child is None, "Pointer shouldnt have a child"
            return True
        elif isinstance(end_chain, sy._LocalTensor):
            assert obj.parent.id == end_chain.id, "TensorVar parent should be the tail LocalTensor" + str(obj.parent.id) + ',' + str(end_chain.id)
            assert end_chain.child.id == obj.id, "Tail LocalTensor child should be the Tensor Var"
            return True
        else:
            raise TypeError('Unsupported end_chain type:', obj)

    elif isinstance(obj, sy._PointerTensor):
        downward = False
        end_chain = obj
        start_id = obj.id
        start_type = type(obj)
    elif isinstance(obj, sy._LocalTensor):
        downward = False
        end_chain = obj
        start_id = obj.id
        start_type = type(obj)

    if downward:
        if obj.child is None:
            raise AttributeError('Chain broken downward without a Pointer at the end, but', obj)
        else:
            return assert_is_chain_well_formed(obj.child, downward, start_id, start_type, end_chain)
    else:
        if obj.parent is None:
            raise AttributeError('Chain broken upward, at', obj)
        else:
            return assert_is_chain_well_formed(obj.parent, downward, start_id, start_type, end_chain)


def find_tail_of_chain(obj, start_id=None, start_type=None):
    """
    Returns the last element of a chain, and perform basic sanity checks
    on the chain like unexpected loops
    """
    if start_id is None:
        start_id = obj.id
        start_type = type(obj)
    else:
        if start_id == obj.id and start_type == type(obj):
            raise StopIteration('The chain looped downward on id', obj.child.id, 'with obj', obj.child)

    if isinstance(obj, (sy._LocalTensor, sy._PointerTensor)):
        return obj
    else:
        if obj.child is None:
            raise AttributeError('Chain is broken on', obj)
        else:
            return find_tail_of_chain(obj.child, start_id, start_type)


def fix_chain_ends(obj):
    """
    Performs BASIC fixes on a chain, typically useful when decoding
    a JSON-ified chain to fix the child and parents attributes
    NOTE that this doesn't guarantee that the chain will be well-formed,
    so calling after `assert_is_chain_well_formed` will be a good idea.
    """
    end_obj = find_tail_of_chain(obj)
    if isinstance(end_obj, sy._LocalTensor):
        end_obj.child = obj
        obj.parent = end_obj
    elif isinstance(end_obj, sy._PointerTensor):
        end_obj.child = None
        obj.parent = None
    else:
        raise TypeError('Unsupported end of chain:', end_obj)

    if is_variable(obj):
        fix_chain_ends(obj.data)


def is_tensor_empty(obj):
    # TODO Will break with PyTorch >= 0.4
    return obj.dim() == 0


def is_syft_tensor(obj):
    """
    Determines whether the arg is a subclass of a SyftTensor
    or is the name of a subclass of a SyftTensor
    """
    if isinstance(obj, str):
        if obj in map(lambda x: x.__name__, sy._SyftTensor.__subclasses__()):
            return True
    else:
        if issubclass(obj.__class__, sy._SyftTensor):
            return True
    return False


def is_tensor(obj):
    """
    Determines whether the arg is a subclass of a Torch Tensor
    or is the name of a subclass of a Torch Tensor
    """
    if isinstance(obj, str):
        if obj in map(lambda x: x.__name__, torch.tensor_types):
            return True
    else:
        if isinstance(obj, tuple(torch.tensor_types)):
            return True
    return False


def is_variable(obj):
    """
    Determines whether the arg is a Variable
    or is the (part of the) name of a class Variable
    """
    if isinstance(obj, str):
        if obj in list(map(lambda x: x.__name__, torch.var_types)) + ['syft.Variable', 'syft.Parameter']:
            return True
    else:
        if isinstance(obj, tuple(torch.var_types)):
            return True
    return False

def is_in_place_method(attr):
    """
    Determines if the method is in-place (ie modifies the self)
    TODO: Can you do better?
    """
    pat = re.compile('__(.+)__')
    return pat.search(attr) is None and attr[-1] == '_'


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
