from .torch_ import utils
import json

class BaseWorker(object):

    def __init__(self,  hook, id=0):

        self.id = id
        self.objects = {}
        self.hook = hook

    def send_obj(message,recipient):
        raise NotImplementedError

    def receive_obj(self, message):
        raise NotImplementedError

    def request_obj(self,obj_id,sender):
        raise NotImplementedError


class LocalWorker(BaseWorker):
    
    def __init__(self, hook, id=0):
        super().__init__(id=id,hook=hook)  
        
    def send_obj(self, obj, recipient):
        recipient.receive_obj(obj._ser())

    def receive_obj(self, message):

        message_obj = json.loads(message)
        obj_type = utils.types_guard(message_obj['torch_type'])
        obj = obj_type._deser(obj_type,message_obj['data'])

        self.objects[message_obj['id']] = obj
        obj.id = message_obj['id']

    def request_obj(self,obj_id,sender):
        
        sender.send_obj(sender.objects[obj_id],self)
        return self.objects[obj_id]

    def request_response(self, recipient, message, response_handler, timeout=10):
        
        return response_handler(recipient.process_command(message))

    def handle_command(self, message):
        """Main function that handles incoming torch commands."""
        print(message)
        message, response_channel = json.loads(message['data'])
        # take in command message, return result of local execution
        result, owners = self.process_command(message)
        compiled = json.dumps(self.compile_result(result, owners))
        if compiled is not None:
            self.return_result(compiled, response_channel)
        else:
            self.return_result(dict(registration=None, torch_type=None,
                var_data=None, var_grad=None), response_channel)

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

        return command(*args, **kwargs), owners

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
                return {'numeric':result}
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
                result = self.register_object_(result, id=result.id, owners=owners)
            except AttributeError:
                result = self.register_object_(result, owners=owners)
            registration = dict(id=result.id,
                owners=result.owners, is_pointer=True)
            return dict(registration=registration, torch_type=torch_type,
                var_data=var_data, var_grad=var_grad)
        except AttributeError as e:
            # result is occasionally a sequence of tensors or variables
            return [self.compile_result(x, owners) for x in result]


    def return_result(self, compiled_result, response_channel):
        """Return compiled result of a torch command"""
        return self.worker.publish(
            channel=response_channel, message=compiled_result)


