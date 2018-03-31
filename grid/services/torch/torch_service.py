from ..base import BaseService
from .hook_service import HookService
from ... import channels
from ...lib import utils,torch_utils as tu

import torch
from bitcoin import base58

import inspect
import copy
import json


class TorchService(BaseService):
    # this service is responsible for certain things
    # common to both clients and workers
    def __init__(self, worker):
        super().__init__(worker)

        def print_messages(message):
            print(message.keys())
            fr = base58.encode(message['from'])
            print(message['data'])
            print("From:" + fr)
            # return message

        # Listen for people to send me tensors
        rec_callback = channels.torch_listen_for_obj_callback(
            self.worker.id)
        self.worker.listen_to_channel(rec_callback, self.receive_obj)

    def receive_obj(self, msg):
        self.receive_obj_break(msg)


    def receive_obj_break(self, msg):
        # TODO: generalize to Variable
        obj_msg = utils.unpack(msg)
        if (type(obj_msg) == str):
            obj_msg = json.loads(obj_msg)

        try: # The inverse of Tensor.ser (defined in torch_utils.py)
            _tensor_type = obj_msg['torch_type']
            try:
                # Ensure
                tensor_type = tu.types_guard(_tensor_type)
            except KeyError:
                raise TypeError(
                    "Tried to receive a non-Torch object of type {}.".format(
                        _tensor_type))
            # this could be a significant failure point, security-wise
            if ('data' in msg.keys()):
                data = obj_msg['data']
                data = tu.tensor_contents_guard(data)
                v = tensor_type(data)
            else:
                v = torch.old_zeros(0).type(tensor_type)

            try:
                # TorchClient case
                # delete registration from init; it's got the wrong id
                del self.worker.objects[v.id]
            except (AttributeError, KeyError):
                # Worker case: v was never formally registered
                pass

            v = self.register_object(v, id=obj_msg['id'], owners=obj_msg['owners'])
            return v
            
        except KeyError:
            return obj_msg['numeric']
