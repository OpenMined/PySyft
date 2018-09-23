"""Torch utility functions related to encoding and decoding in a JSON-serializable fashion """
import json
import re
import types
import functools
import logging
import torch
import syft
import syft as sy
import numpy as np

from syft.core import utils
from syft.core.frameworks.torch import utils as torch_utils
from .numpy import array, array_ptr


def encode(message, retrieve_pointers=False, private_local=True):
    """
    Help function to call the PythonEncoder
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
        Encode python and torch objects to be JSON-able.
        In particular, tensors of all types are replaced by single key dict, which
        encode the type of the tensor, and the related value includes associated
        content.
        Note that a python object is returned, not JSON.

        Example:
            Input:
                sy.FloatTensor([1,2,3,4])
            Output:
                ...: {'__FloatTensor__': {
                            'child': {'___LocalTensor__': {...},
                            'data': [1, 2, 3, 4],
                            'torch_type': 'syft.FloatTensor',
                            ...}
                     }
    """

    def __init__(self):
        self.retrieve_pointers = False
        self.found_pointers = []
        self.found_next_child_types = []
        self.tensorvar_types = tuple(torch.tensorvar_types)

    def encode(self, obj, retrieve_pointers=False, private_local=True):
        """
            Performs encoding, and retrieves if requested all pointers found
        """
        self.retrieve_pointers = retrieve_pointers

        serialized_obj = self.python_encode(obj, private_local)

        serialized_msg = {'obj': serialized_obj}
        # Give instruction to the decoder, should he acquire the tensor or register them
        # If it's private, you can't access directly the data, so you subscribe to it with a pointer
        if private_local:
            serialized_msg['mode'] = 'subscribe'
        else:  # If it's public, you can acquire the data directly
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
        elif isinstance(obj, np.ndarray):
            return obj.ser(private=private_local, to_json=False)
        # Tensors and Variable encoded with their id
        elif torch_utils.is_tensor(obj) or torch_utils.is_variable(obj):
            tail_object = torch_utils.find_tail_of_chain(obj)
            if self.retrieve_pointers and isinstance(tail_object, sy._PointerTensor):
                self.found_pointers.append(tail_object)
            if torch_utils.is_variable(obj):
                return obj.ser(private=private_local, is_head=True)
            else:
                return obj.ser(private=private_local)
        # sy._SyftTensor (Pointer, Local)
        # [Note: shouldn't be called on regular chain with end=tensorvar]
        elif torch_utils.is_syft_tensor(obj):
            tail_object = torch_utils.find_tail_of_chain(obj)
            if self.retrieve_pointers and isinstance(tail_object, sy._PointerTensor):
                self.found_pointers.append(tail_object)
            return obj.ser(private=private_local)
        # List
        elif isinstance(obj, list):
            return [self.python_encode(i, private_local) for i in obj]
        # Iterables non json-serializable
        elif isinstance(obj, (tuple, set, bytearray, range)):
            key = '__' + type(obj).__name__ + '__'
            return {key: [self.python_encode(i, private_local) for i in obj]}
        # Slice
        elif isinstance(obj, slice):
            key = '__' + type(obj).__name__ + '__'
            return {key: {'args': [obj.start, obj.stop, obj.step]}}
        # Dict
        elif isinstance(obj, dict):
            return {
                k: self.python_encode(v, private_local)
                for k, v in obj.items()
            }
        # Generator
        elif isinstance(obj, types.GeneratorType):
            logging.warning("Generator args can't be transmitted")
            return []
        # worker
        elif isinstance(obj, (sy.SocketWorker, sy.VirtualWorker)):
            return {'__worker__': obj.id}
        # Else log the error
        else:
            raise ValueError('Unhandled type', type(obj))


def decode(message, worker, acquire=None, message_is_dict=False):
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
    :param acquire: Should we copy the data of point at it
    :param message_is_dict: Is the message a dictionary already or a JSON string needing decoding?
    :return: The message decoded
    """
    decoder = PythonJSONDecoder(worker=worker, acquire=acquire)

    # Handle when the message is a bytestring
    if isinstance(message, bytes):
        message = message.decode('utf-8')

    if(message_is_dict):
        dict_message = message
    else:
        dict_message = json.loads(message)

    # If acquire is specified, then know how we want to decode, and implicitly
    # we want to decode everything of the message
    if acquire is not None:
        return decoder.python_decode(message)

    # TODO It would be good to have a standardized place to put the 'mode' argument
    # Extract the mode: depending of the structure of the message, the mode argument
    # is not at the same place
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

        # PLAN A: See if the dct object is not actually a dictionary and address
        # each case.

        if isinstance(dct, (int, str, float)):
            return dct
        if isinstance(dct, (list,)):
            return [self.python_decode(o) for o in dct]
        if dct is None:
            return None
        if not isinstance(dct, dict):
            print(type(dct))
            raise TypeError('Type not handled', dct)

        # PLAN B: If the dct object IS a dictionary, check to see if it has a "type" key

        if('type' in dct):
            if dct['type'] == "numpy.array":

                # at first glance, the following if statement might seem a bit confusing
                # since the dct object is identical for both. Basically, the pointer object
                # is created here (on the receiving end of a message) as opposed to on the sending
                # side. We decide whether to use the dictionary to construct a pointer or the
                # actual tensor based on wehther self.acquire is true. Note that this changes
                # how dct['id'] is used. If creating an actual tensor, the tensor id is set to dct['id]
                # otherwise, id_at_location is set to be dct['id']. Similarly with dct['owner'].

                # if we intend to receive the tensor itself, construct an array
                if(self.acquire):
                    return array(dct['data'], id=dct['id'], owner=self.worker)

                # if we intend to create a pointer, construct a pointer. Note that
                else:
                    return array_ptr(dct['data'],
                                     owner=self.worker,
                                     location=self.worker.get_worker(dct['owner']),
                                     id_at_location=dct['id'])
            elif dct['type'] == 'numpy.array_ptr':
                return self.worker.get_obj(dct['id_at_location'])

        # Plan C: As a last resort, use a Regex to try to find a type somewhere.
        # TODO: Plan C should never be called - but is used extensively in PySyft's PyTorch integratio

        pat = re.compile('__(.+)__')
        for key, obj in dct.items():
            if pat.search(key) is not None:
                obj_type = pat.search(key).group(1)
                # Case of a tensor
                if torch_utils.is_tensor(obj_type):
                    o = torch.guard['syft.' + obj_type].deser({key: obj}, self.worker, self.acquire)
                    return o
                # Case of a Variable
                elif torch_utils.is_variable(obj_type):
                    return sy.Variable.deser({key: obj}, self.worker, self.acquire, is_head=True)
                # Case of a Syft tensor
                elif torch_utils.is_syft_tensor(obj_type):
                    return sy._SyftTensor.deser_routing({key: obj}, self.worker, self.acquire)
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