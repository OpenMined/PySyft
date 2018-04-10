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

        try:
            torch_type = tu.types_guard(obj_msg)

            if torch_type in self.var_types:
                v = self.build_var(obj_msg, torch_type)
            else:
                v = self.build_tensor(obj_msg, torch_type)

            return self.handle_register(v, obj_msg)
            
        except KeyError:
            # if obj_msg has no 'torch_type' key
            return obj_msg['numeric']


    @classmethod
    def build_tensor(cls, obj_msg, torch_type):
        # this could be a significant failure point, security-wise
        if 'data' in obj_msg.keys():
            data = obj_msg['data']
            data = tu.tensor_contents_guard(data)
            v = torch_type(data)
        else:
            v = torch.old_zeros(0).type(tensor_type)
        return v

    def build_var(self, obj_msg, torch_type):
        
        if 'data' in obj_msg.keys():
            data_msg = json.loads(obj_msg['data'])
            tensor_type = tu.types_guard(data_msg)
            data_obj = self.build_tensor(data_msg, tensor_type)
            data = self.handle_register(data_obj, data_msg)

        if 'grad' in obj_msg.keys():
            if obj_msg['grad'] is not None:
                grad_msg = json.loads(obj_msg['grad'])
                var_type = tu.types_guard(grad_msg)
                grad_obj = self.build_var(grad_msg, var_type)
                grad = self.handle_register(grad_obj, grad_msg)
            else:
                grad = None
        var = torch_type(data, volatile=obj_msg['volatile'],
            requires_grad=obj_msg['requires_grad'])
        var.grad = grad
        return var


    def handle_register(self, torch_object, obj_msg):
        try:
            # TorchClient case
            # delete registration from init; it's got the wrong id
            del self.worker.objects[torch_object.id]
        except (AttributeError, KeyError):
            # Worker case: v was never formally registered
            pass

        torch_object = self.register_object_(
            torch_object, id=obj_msg['id'], owners=obj_msg['owners'])
        return torch_object

