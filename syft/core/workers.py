"""Interfaces for communicating about objects between Clients and Workers"""

import json, numbers, re, random

from . import utils


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
        self._tmp_objects = {}
        self._known_workers = {}
        self.hook = hook

    def add_worker(self, worker):
        self._known_workers[worker.id] = worker

    def get_worker(self, id_or_worker):
        """If you pass in an ID, it will attempt to find the worker object (pointer)
        withins self._known_workers. If you instead pass in a pointer itself, it will 
        save that as a known_worker."""

        if(issubclass(type(id_or_worker), BaseWorker)):
            self._known_workers[id_or_worker.id] = id_or_worker
            return id_or_worker
        else:
            return self._known_workers[id_or_worker]

    def set_obj(self, remote_key, value, force=False, tmp=False):
        if(tmp and self.is_client_worker):
            self._tmp_objects[remote_key] = value

        if(not self.is_client_worker or force):
            self._objects[remote_key] = value

    def get_obj(self, remote_key):
        # if(not self.is_client_worker):
        
        return self._objects[remote_key]

    def rm_obj(self, remote_key):
        if(remote_key in self._objects):
            del self._objects[remote_key]

    def clear_tmp_objects(self):
        self._tmp_objects = {}

    def send_obj(self, message, recipient):
        raise NotImplementedError

    def receive_obj(self, message):
        raise NotImplementedError

    def request_obj(self, obj_id, sender):
        raise NotImplementedError

    # Helpers for HookService and TorchService
    @staticmethod
    def _check_workers(self, workers):
        if type(workers) is str:
            workers = [workers]
        if issubclass(type(workers), BaseWorker):
            workers = [workers]
        elif not hasattr(workers, '__iter__'):
            raise TypeError(
                """Can only send {} to a string worker ID or an iterable of
                string worker IDs, not {}""".format(self.__name__, workers)
                )
        return workers

    def process_response(self, response):
        """Processes a worker's response from a command."""
        # TODO: Extend to responses that are iterables.
        # TODO: Fix the case when response contains only a numeric
        # print(response)
        response = json.loads(response)
        try:
            return (response['registration'], response['torch_type'],
                    response['var_data'], response['var_grad'])
        except:
            return response

    # Worker needs to retrieve tensor by ID before computing with it
    def retrieve_tensor(self, self2, x):
        try:
            return self.get_obj(self.id_tensorvar(x))
        except TypeError:
            try:
                return [self.get_obj(i) for i in self.id_tensorvar(x)]
            except TypeError:
                return x
        except KeyError:
            return x

    def command_guard(self, command, allowed):
        if command not in allowed:
            raise RuntimeError(
                'Command "{}" is not a supported Torch operation.'.format(command))
        return command

    # # Client needs to identify a tensor before sending commands that use it
    def id_tensorvar(self, x):
        pat = re.compile('_fl.(.*)')
        try:
            if isinstance(x, str):
                return int(pat.search(x).group(1))
            else:
                return [self.id_tensorvar(i) for i in x]
        except AttributeError:
            return x

class VirtualWorker(BaseWorker):

    def __init__(self, hook, id=0, is_client_worker=False):
        super().__init__(id=id, hook=hook, is_client_worker=is_client_worker)

    def send_obj(self, obj, recipient, delete_local=True):
        obj = recipient.receive_obj(obj.ser())
        if(delete_local):
            self.rm_obj(obj.id)

        return obj

    def receive_obj(self, message):

        message_obj = json.loads(message)
        obj_type = self.hook.types_guard(message_obj['torch_type'])
        obj = obj_type.deser(obj_type, message_obj)

        self.handle_register(obj, message_obj, force_attach_to_worker=True)

        return obj

    def handle_register(self, torch_object, obj_msg, force_attach_to_worker=False, temporary=False):

        try:
            # TorchClient case
            # delete registration from init; it's got the wrong id
            self.rm_obj(torch_object.id)
        except (AttributeError, KeyError):
            # Worker case: v was never formally registered
            pass

        torch_object = self.register_object(self,
                                            torch_object,
                                            id=obj_msg['id'],
                                            owners=[self.id],
                                            force_attach_to_worker=force_attach_to_worker,
                                            temporary=temporary)


        return torch_object

    def register_object(self, worker, obj, force_attach_to_worker=False, temporary=False, **kwargs):
        """
        Registers an object with the current worker node.
        Selects an id for the object, assigns a list of owners,
        and establishes whether it's a pointer or not.

        Args:
            obj: a Torch instance, e.g. Tensor or Variable
        Default kwargs:
            id: random integer between 0 and 1e10
            owners: list containing local worker's IPFS id
            is_pointer: False
        """
        # TODO: Assign default id more intelligently (low priority)
        #       Consider popping id from long list of unique integers
        keys = kwargs.keys()

        obj.id = (kwargs['id']
                  if ('id' in keys and kwargs['id'] is not None)
                  else random.randint(0, 1e10))

        obj.owners = (kwargs['owners']
                      if 'owners' in keys
                      else [worker.id])

        # check to see if we can resolve owner id to pointer
        owner_pointers = list()
        for owner in obj.owners:
            if owner in self._known_workers.keys():
                owner_pointers.append(self._known_workers[owner])
            else:
                owner_pointers.append(owner)
        obj.owners = owner_pointers

        obj.is_pointer = (kwargs['is_pointer']
                          if 'is_pointer' in keys
                          else False)

        mal_points_away = obj.is_pointer and worker.id in obj.owners
        # print("Mal Points Away:" + str(mal_points_away))
        # print("self.local_worker.id in obj.owners == " + str(self.local_worker.id in obj.owners))
        # The following was meant to assure that we didn't try to
        # register objects we didn't have. We end up needing to register
        # objects with non-local owners on the worker side before sending
        # things off, so it's been relaxed.  Consider using a 'strict'
        # kwarg for strict checking of this stuff
        mal_points_here = False
        # mal_points_here = not obj.is_pointer and self.local_worker.id not in obj.owners
        if mal_points_away or mal_points_here:
            raise RuntimeError(
                'Invalid registry: is_pointer is {} but owners is {}'.format(
                    obj.is_pointer, obj.owners))
        # print("setting object:" + str(obj.id))
        self.set_obj(obj.id, obj, force=force_attach_to_worker, tmp=temporary)
        return obj

    def request_obj(self, obj_id, sender):
        sender = self.get_worker(sender)
        obj = sender.send_obj(sender.get_obj(obj_id), self)

        # for some reason, when returning obj from request_obj method, the gradient (obj.grad) gets 
        # re-initialized without being re-registered and as a consequence does not have an id, causing the 
        # x.grad.id to fail because it does not exist. As a result, we've needed to 
        # store objects temporarily in self._tmpobjects which seems to fix it. Super strange bug which took 
        # multiple days to figure out. The true cause is still unknown but this workaround seems to work 
        # well for now. Anyway, so we need to return a cleanup method which is called immediately before this
        # is returned to the client. Note that this is ONLY necessary for the client (which doesn't store objects
        # in self._objects)

        return obj, self.clear_tmp_objects

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
        args = utils.map_tuple(self, command_msg['args'], self.retrieve_tensor)
        kwargs = utils.map_dict(self, command_msg['kwargs'], self.retrieve_tensor)
        has_self = command_msg['has_self']
        # TODO: Implement get_owners and refactor to make it prettier
        combined = list(args) + list(kwargs.values())

        if has_self:
            command = self.command_guard(command_msg['command'],
                                          self.hook.tensorvar_methods)
            obj_self = self.retrieve_tensor(self, command_msg['self'])
            combined = combined + [obj_self]
            command = eval('obj_self.{}'.format(command))
        else:
            command = self.command_guard(command_msg['command'], self.torch_funcs)
            command = eval('torch.{}'.format(command))

        # we need the original tensorvar owners so that we can register
        # the result properly later on
        tensorvars = [x for x in combined if type(x).__name__ in self.hook.tensorvar_types_strs]
        _, owners = self.hook._get_owners(tensorvars)

        owner_ids = list()
        for owner in owners:
            if(type(owner) == int):
                owner_ids.append(owner)
            else:
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
                result = self.register_object(self, result, id=result.id, owners=owners)
            except AttributeError:
                result = self.register_object(self, result, owners=owners)

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
