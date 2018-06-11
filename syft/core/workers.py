"""Interfaces for communicating about objects between Clients and Workers"""

import json
import numbers
import re


class BaseWorker(object):
    r"""
    Concatenates the given sequence of :attr:`seq` tensors in the given dimension.
    All tensors must either have the same shape (except in the concatenating
    dimension) or be empty.

    :func:`torch.cat` can be seen as an inverse operation for :func:`torch.split`
    and :func:`torch.chunk`.
    
    :func:`torch.cat` can be best understood via examples.

    :Parameters:
        
        * **seq (sequence of Tensors)** any python sequence of tensors of the same type. 
          Non-empty tensors provided must have the same shape, except in the cat dimension.
        
        * **dim (int, optional)** the dimension over which the tensors are concatenated
        
        * **out (Tensor, optional)** the output tensor
    
    :Example:
    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 0)
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 1)
    tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
             -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
             -0.5790,  0.1497]])
    """
    def __init__(self,  hook, id=0, is_client_worker=False):

        self.id = id
        self.is_client_worker = is_client_worker
        self._objects = {}
        self._known_workers = {}
        self.hook = hook

    def set_obj(self, remote_key, value, force=False):
        if(not self.is_client_worker or force):
            self._objects[remote_key] = value

    def get_obj(self, remote_key):
        # if(not self.is_client_worker):
        return self._objects[remote_key]

    def rm_obj(self, remote_key):
        # if(not self.is_client_worker):
        del self._objects[remote_key]

    def send_obj(self, message, recipient):
        raise NotImplementedError

    def receive_obj(self, message):
        raise NotImplementedError

    def request_obj(self, obj_id, sender):
        raise NotImplementedError


class VirtualWorker(BaseWorker):

    def __init__(self, hook, id=0, is_client_worker=False):
        super().__init__(id=id, hook=hook, is_client_worker=is_client_worker)

    def send_obj(self, obj, recipient):
        recipient.receive_obj(obj._ser())

    def receive_obj(self, message):

        message_obj = json.loads(message)
        obj_type = utils.types_guard(message_obj['torch_type'])
        obj = obj_type._deser(obj_type, message_obj)
        self.handle_register(obj, message_obj,force_attach_to_worker=True)

        # self.objects[message_obj['id']] = obj
        # obj.id = message_obj['id']

    def handle_register(self, torch_object, obj_msg, force_attach_to_worker=False):

        try:
            # TorchClient case
            # delete registration from init; it's got the wrong id
            self.rm_obj(torch_object.id)
        except (AttributeError, KeyError):
            # Worker case: v was never formally registered
            pass

        torch_object = self.hook.register_object_(self,
                                                  torch_object,
                                                  id=obj_msg['id'],
                                                  owners=[self.id],
                                                  force_attach_to_worker=force_attach_to_worker)

        return torch_object

    def request_obj(self, obj_id, sender):

        sender.send_obj(sender.get_obj(obj_id), self)

        return self.get_obj(obj_id)

    def request_response(self, recipient, message, response_handler, timeout=10):
        return response_handler(recipient.handle_command(message))

    def handle_command(self, message):
        """Main function that handles incoming torch commands."""

        message = message
        # take in command message, return result of local execution
        result, owners = self.process_command(message)

        compiled = self.compile_result(result, owners)

        compiled = json.dumps(compiled)
        if compiled is not None:
            return compiled
        else:
            return dict(registration=None, torch_type=None,
                        var_data=None, var_grad=None)

    def process_command(self, command_msg):
        """
        Process a command message from a client worker. Returns the
        result of the computation and a list of the result's owners.
        """
        # Args and kwargs contain special strings in place of tensors
        # Need to retrieve the tensors from self.worker.objects
        args = utils.map_tuple(self, command_msg['args'], utils.retrieve_tensor)
        kwargs = utils.map_dict(self, command_msg['kwargs'], utils.retrieve_tensor)
        has_self = command_msg['has_self']
        # TODO: Implement get_owners and refactor to make it prettier
        combined = list(args) + list(kwargs.values())

        if has_self:
            command = utils.command_guard(command_msg['command'],
                                          self.hook.tensorvar_methods)
            obj_self = utils.retrieve_tensor(self, command_msg['self'])
            combined = combined + [obj_self]
            command = eval('obj_self.{}'.format(command))
        else:
            command = utils.command_guard(command_msg['command'], self.torch_funcs)
            command = eval('torch.{}'.format(command))

        # we need the original tensorvar owners so that we can register
        # the result properly later on
        tensorvars = [x for x in combined if type(x).__name__ in self.hook.tensorvar_types_strs]
        _, owners = utils.get_owners(tensorvars)

        owner_ids = list()
        for owner in owners:
            owner_ids.append(owner.id)

        return command(*args, **kwargs), owner_ids

    def compile_result(self, result, owners):
        """
        Converts the result to a JSON serializable message for sending
        over PubSub.
        """
        if result is None:
            return dict(registration=None, torch_type=None,
                        var_data=None, var_grad=None)
        try:

            # result is infrequently a numeric
            if isinstance(result, numbers.Number):
                return {'numeric': result}

            # result is usually a tensor/variable
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
            try:
                result = self.hook.register_object_(self, result, id=result.id, owners=owners)
            except AttributeError:
                result = self.hook.register_object_(self, result, owners=owners)

            registration = dict(id=result.id,
                                owners=owners, is_pointer=True)

            return dict(registration=registration, torch_type=torch_type,
                        var_data=var_data, var_grad=var_grad)

        except AttributeError as e:
            # result is occasionally a sequence of tensors or variables

            return [self.compile_result(x, owners) for x in result]

    def return_result(self, compiled_result, response_channel):
        """Return compiled result of a torch command"""
        return self.worker.publish(
            channel=response_channel, message=compiled_result)
