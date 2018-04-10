import json
import random
import re
import numbers

import torch
from ..base import BaseService
from ...lib import utils, torch_utils as tu
from ... import channels

class HookWorkerService(BaseService):
    def __init__(self, worker):
        super().__init__(worker)
        for tensor_type in self.tensor_types:
            tu.hook_tensor__ser(self, tensor_type)
        tu.hook_var__ser(self)

        # Listen for torch object requests
        req_callback = channels.torch_listen_for_obj_req_callback(
            self.worker.id)
        self.worker.listen_to_channel(req_callback, self.receive_obj_request)

        # Listen for incoming torch commands
        comm_callback = channels.torch_listen_for_command_callback(
            self.worker.id)
        self.worker.listen_to_channel(comm_callback, self.handle_command)


    def handle_command(self, message):
        """Main function that handles incoming torch commands."""
        message, response_channel = utils.unpack(message)
        # take in command message, return result of local execution
        result, owners = self.process_command(message)
        compiled = json.dumps(self.compile_result(result, owners))
        self.return_result(compiled, response_channel)


    def process_command(self, command_msg):
        """
        Process a command message from a client worker. Returns the
        result of the computation and a list of the result's owners.
        """
        # Args and kwargs contain special strings in place of tensors
        # Need to retrieve the tensors from self.worker.objects
        args = tu.map_tuple(self, command_msg['args'], tu.retrieve_tensor)
        kwargs = tu.map_dict(self, command_msg['kwargs'], tu.retrieve_tensor)
        has_self = command_msg['has_self']
        # TODO: Implement get_owners and refactor to make it prettier
        combined = list(args) + list(kwargs.values())

        if has_self:
            command = tu.command_guard(command_msg['command'],
                self.tensorvar_methods)
            obj_self = tu.retrieve_tensor(self, command_msg['self'])
            combined = combined + [obj_self]
            command = eval('obj_self.{}'.format(command))
        else:
            command = tu.command_guard(command_msg['command'], self.torch_funcs)
            command = eval('torch.{}'.format(command))

        # we need the original tensorvar owners so that we can register
        # the result properly
        _, owners = tu.get_owners(combined)

        return command(*args, **kwargs), owners


    def compile_result(self, result, owners):
        """
        Converts the result to a JSON serializable message for sending
        over PubSub.
        """
        try:
            # result is infrequently a numeric
            if isinstance(result, numbers.Number):
                return {'numeric':result}
            # result is usually a tensor/variable
            print(result)
            torch_type = re.search("<class '(torch.(.*))'>",
                str(result.__class__)).group(1)

            try:
                var_data = self.compile_result(result.data, owners)
            except (AttributeError, RuntimeError):
                var_data = None
            try:
                assert result.grad is not None
                var_grad = self.compile_result(result.grad, owners)
            except (AttributeError, AssertionError):
                var_grad = None

            result = self.register_object_(result, owners=owners)
            registration = dict(id=result.id,
                owners=result.owners, is_pointer=True)
            return dict(registration=registration, torch_type=torch_type,
                var_data=var_data, var_grad=var_grad)
        except AttributeError:
            # result is occasionally a sequence of tensors or variables
            return [self.compile_result(x, owners) for x in result]


    def return_result(self, compiled_result, response_channel):
        """Return compiled result of a torch command"""
        return self.worker.publish(
            channel=response_channel, message=compiled_result)


    def receive_obj_request(self, msg):
        """Handles requests for Torch objects."""
        obj_id, response_channel = utils.unpack(msg)

        if (obj_id in self.worker.objects.keys()):
            new_owner = re.search('(.+)_[0-9]{1,11}', response_channel).group(1)
            obj = self.register_object_(self.worker.objects[obj_id],
                id=obj_id, owners=[new_owner])
            response_str = obj._ser()
        else:
            # TODO: replace this with something that triggers a nicer
            #       error on the client
            response_str = 'n/a - tensor not found'

        self.worker.publish(channel=response_channel, message=response_str)