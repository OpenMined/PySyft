from ... import channels
from ..base import BaseService
from bitcoin import base58
import torch

import torch
from torch.autograd import Variable
import inspect
import random
import copy
from ...lib import utils
from ...lib import serde
import json


class TorchService(BaseService):

    # this service creates everything the client needs to be able to interact with torch on the Grid
    # (it's really awesome, but it's a WIP)

    def __init__(self, worker):
        super().__init__(worker)

        self.worker = worker

        self.objects = {}

        # self.hook_float_tensor_add()
        self.hook_float_tensor___init__()
        self.hook_float_tensor_serde()
        self.hook_float_tensor_send()
        self.hook_float_tensor_process_command()
        self.hook_float_tensor_get()
        self.hook_float_tensor___repr__()

        def print_messages(message):
            print(message.keys())
            fr = base58.encode(message['from'])
            print(message['data'])
            print("From:" + fr)
            # return message

        # I listen for people to send me tensors!!
        listen_for_obj_callback_channel = channels.torch_listen_for_obj_callback(
            self.worker.id)
        self.worker.listen_to_channel(listen_for_obj_callback_channel,
                                      self.receive_obj)

        # I listen for people to ask me for tensors!!
        listen_for_obj_callback_channel = channels.torch_listen_for_obj_req_callback(
            self.worker.id)
        self.worker.listen_to_channel(listen_for_obj_callback_channel,
                                      self.receive_obj_request)

    def receive_obj(self, msg):
        self.receive_obj_break(msg)

    def receive_obj_break(self, msg):

        dic = json.loads(msg['data'])

        if (dic['type'] == 'torch.FloatTensor'):
            obj = torch.FloatTensor.de(dic)
            obj.is_pointer_to_remote = False
            obj.owner = self.worker.id
            self.objects[obj.id] = obj
            return obj
        return "not a float tensor"

    def register_object(self, obj, is_pointer_to_remote):
        obj.id = random.randint(0, 1e10)
        obj.owner = self.worker.id
        obj.worker = self.worker
        obj.is_pointer_to_remote = False
        self.objects[obj.id] = obj
        return obj

    def send_obj(self, obj, to):
        self.worker.publish(
            channels.torch_listen_for_obj_callback(to), message=obj.ser())
        obj.is_pointer_to_remote = True
        obj.owner = to
        return obj

    def send_command(self, command, to):
        return to.receive_command(command)

    def request_obj(self, obj):

        return self.worker.request_response(
            channel=channels.torch_listen_for_obj_req_callback(obj.owner),
            message=obj.id,
            response_handler=self.receive_obj_break)

    def receive_obj_request(self, msg):

        obj_id, response_channel = json.loads(msg['data'])

        if (obj_id in self.objects.keys()):
            response_str = self.objects[obj_id].ser()
        else:
            response_str = 'n/a - tensor not found'

        self.worker.publish(channel=response_channel, message=response_str)

    def receive_command(self, command):
        if (command['base_type'] == 'torch.FloatTensor'):
            raw_response = torch.FloatTensor.process_command(self, command)

        return json.dumps(raw_response)

    def process_response(self, response):
        response = json.loads(response)
        tensor_ids = response
        out_tensors = list()
        for raw_msg in tensor_ids:
            msg = json.loads(raw_msg)
            if (msg["type"] == "torch.FloatTensor"):
                obj = torch.FloatTensor.de(msg)
            out_tensors.append(obj)

        if (len(out_tensors) > 1):
            return out_tensors
        elif (len(out_tensors) == 1):
            return out_tensors[0]
        else:
            return None

    def function2json(self, obj, name, frame, ix):

        args, varargs, keywords, values = inspect.getargvalues(frame)

        command = {}
        command[
            'id'] = ix  # This id is assigned as a placeholder for the data that the worker has
        command['command'] = name
        command['base_type'] = obj.type()
        command['args'] = args
        command['varargs'] = varargs
        command['keywords'] = keywords
        command['values'] = [values[arg].id for arg in args]
        command['types'] = [type(val) for val in command['values']]

        return command

    # GENERIC

    def assign_workers(self):
        def decorate(func):
            def send_to_workers(*args, **kwargs):
                if (args[0].is_pointer_to_remote):
                    command = func(*args, **kwargs)
                    response = self.send_command(command, args[0].owner)
                    return self.process_response(response)

                else:
                    return func(*args, **kwargs)

            return send_to_workers

        return decorate

    # FLOAT TENSOR FUNCTIONS
    def hook_float_tensor___init__(service_self):
        def new___init__(self, *args):
            super(torch.FloatTensor, self).__init__()
            self = service_self.register_object(self, False)

        torch.FloatTensor.__init__ = new___init__

    def hook_float_tensor_add(self2):
        @self2.assign_workers()
        def new_add(self, other):
            if (self.is_pointer_to_remote):
                frame = inspect.currentframe()
                command = self.owner.function2json(self, 'add', frame, self.id)
                return command
            else:
                result = self.old_add(other)
                return self2.register_object(result, True)

        try:
            torch.FloatTensor.old_add
        except:
            torch.FloatTensor.old_add = torch.FloatTensor.add

        torch.FloatTensor.add = new_add

    def hook_float_tensor_serde(self):
        def ser(self, include_data=True):

            msg = {}
            msg['type'] = 'torch.FloatTensor'
            if (include_data):
                msg['data'] = self.tolist()
            msg['id'] = self.id
            msg['owner'] = self.owner

            return json.dumps(msg)

        def de(msg):
            if (type(msg) == str):
                msg = json.loads(msg)

            if ('data' in msg.keys()):
                v = torch.FloatTensor(msg['data'])
            else:
                v = torch.zeros(0)

            del self.objects[v.id]

            if (msg['id'] in self.objects.keys()):
                v_orig = self.objects[msg['id']].set_(v)
                return v_orig
            else:
                self.objects[msg['id']] = v
                v.id = msg['id']
                v.owner = msg['owner']
                return v

        torch.FloatTensor.ser = ser
        torch.FloatTensor.de = de

    def hook_float_tensor___repr__(service_self):
        def __repr__(self):
            if (service_self.worker.id == self.owner):
                return self.old__repr__()
            else:
                return "[ torch.FloatTensor - Location:" + str(
                    self.owner) + " ]"

        # if haven't reserved the actual __repr__ function - reserve it now
        try:
            torch.FloatTensor.old__repr__
        except:
            torch.FloatTensor.old__repr__ = torch.FloatTensor.__repr__

        torch.FloatTensor.__repr__ = __repr__

    def hook_float_tensor_send(self):
        def send(self, new_owner):
            self.worker.services['torch_service'].send_obj(self, new_owner)
            self.set_(torch.zeros(0))
            return self

        torch.FloatTensor.send = send

    def hook_float_tensor_get(self):
        def get(self):

            if (self.worker.id != self.owner):
                self.worker.services['torch_service'].request_obj(self)

            return self

        torch.FloatTensor.get = get

    def hook_float_tensor_process_command(self):
        def process_command(worker, command):
            if (command['command'] == 'add'):
                a = worker.objects[int(command['values'][0])]
                b = worker.objects[int(command['values'][1])]
                c = a.add(b)
                return [c.ser(False)]
            else:
                return "command not found"
            ""

        torch.FloatTensor.process_command = process_command
