"""Interfaces for communicating about objects between Clients and Workers"""

import json, numbers, re, random

from . import utils


class BaseWorker(object):
    r"""
    The BaseWorker class establishes a consistent interface for communicating between different machines
    about tensors, variables, models, and other network related information. It defines base functionality
    for storing objects within the worker (or in some cases specifically not storing them), and for keeping
    track of known workers across a network. 

    This class does NOT include any communication specific functionality such as peer-to-peer protocols, routing,
    node discovery, or even socket architecture. Only classes that extend the BaseWorker should carry such 
    functionality.

    :Parameters:
        
        * **hook (**:class:`.hooks.BaseHook` **)** This is a reference to the hook object which overloaded the 
          underlying deep learning framework.
        
        * **id (int or string, optional)** the integer or string identifier for this node
        
        * **is_client_worker (bool, optional)** a boolean which determines whether this worker is associeted 
          with an end user client. If so, it assumes that the client will maintain control over when 
          tensors/variables/models are instantiated or deleted as opposed to handling tensor/variable/model 
          lifecycle internally.

        * **objects (list of tensors, variables, or models, optional)** 
          When the worker is NOT a client worker, it stores all tensors it receives or creates in this dictionary.
          The key to each object is it's id.

        * **tmp_objects (list of tensors, variables, or models, optional)**
          When the worker IS a client worker, it stores some tensors temporarily in this _tmp_objects simply to ensure
          that they do not get deallocated by the Python garbage collector while in the process of being registered.
          This dictionary can be emptied using the clear_tmp_objects method.
    
        * **known_workers (list of **:class:`BaseWorker` ** objects, optional)** This dictionary can include all known workers on a network. Some extensions of BaseWorker will use this
          list to bootstrap the network.

        * **verbose (bool, optional)** A flag for whether or not to print events to stdout.

    """
    def __init__(self,  hook, id=0, is_client_worker=False, objects={}, tmp_objects={}, known_workers={}, verbose=True):

        # This is a reference to the hook object which overloaded the underlying deep learning framework
        # (at the time of writing this is exclusively TorchHook)
        self.hook = hook

        # the integer or string identifier for this node
        self.id = id

        # a boolean which determines whether this worker is associeted with an end user client. If so, it assumes
        # that the client will maintain control over when varialbes are instantiated or deleted as opposed to 
        # handling tensor/variable/model lifecycle internally.
        self.is_client_worker = is_client_worker

        # When the worker is NOT a client worker, it stores all tensors it receives or creates in this dictionary.
        # The key to each object is it's id.
        self._objects = {}
        for k,v in objects.items():
            self._objects[k] = v

        # When the worker IS a client worker, it stores some tensors temporarily in this _tmp_objects simply to ensure
        # that they do not get deallocated by the Python garbage collector while in the process of being registered.
        # This dictionary can be emptied using the clear_tmp_objects method.
        self._tmp_objects = {}
        for k,v in tmp_objects.items():
            self._tmp_objects[k] = v

        # This dictionary includes all known workers on a network. Extensions of BaseWorker will include advanced 
        # functionality for adding to this dictionary (node discovery). In some cases, one can initialize this with 
        # known workers to help bootstrap the network.
        self._known_workers = {}
        for k,v in known_workers.items():
            self._known_workers[k] = v

        # A flag for whether or not to print events to stdout.
        self.verbose = verbose

    def add_worker(self, worker):
        """add_worker(worker) -> None
        This method adds a worker to the list of _known_workers internal to the BaseWorker. It endows this class with
        the ability to communicate with the remote worker being added, such as sending and receiving objects, commands,
        or information about the network.

        :Parameters:
        
        * **worker (**:class:`BaseWorker` **)** This is an object pointer to a remote worker, which must have a unique id.
        
        :Example:

        >>> from syft.core.hooks import TorchHook
        >>> from syft.core.hooks import torch
        >>> from syft.core.workers import VirtualWorker
        >>> hook = TorchHook()
        Hooking into Torch...
        Overloading complete.
        >>> local = hook.local_worker
        >>> remote = VirtualWorker(id=1, hook=hook)
        >>> local.add_worker(remote)
        >>> x = torch.FloatTensor([1,2,3,4,5])
        >>> x
         1
         2
         3
         4
         5
        [torch.FloatTensor of size 5]
        >>> x.send(remote)
        >>> x
        [torch.FloatTensor - Locations:[<syft.core.workers.VirtualWorker object at 0x11848bda0>]]
        >>> x.get()
        >>> x
         1
         2
         3
         4
         5
        [torch.FloatTensor of size 5]        
        """
        if(worker.id in self._known_workers and self.verbose):
            print("WARNING: Worker ID " + str(worker.id) + " taken. Have I seen this worker before?")
            print("WARNING: Replacing it anyways... this could cause unexpected behavior...")

        self._known_workers[worker.id] = worker
        

    def get_worker(self, id_or_worker):
        """get_worker(self, id_or_worker) -> BaseWorker
        If you pass in an ID, it will attempt to find the worker object (pointer)
        withins self._known_workers. If you instead pass in a pointer itself, it will 
        save that as a known_worker if it does not exist as one. This method is primarily useful
        becuase often tensors have to store only the ID to a foreign worker which may or may not be known
        by the worker that is deserializing it at the time of deserialization. This method allows for 
        resolution of worker ids to workers to happen automatically while also making the current worker
        aware of new ones when discovered through other processes.

        
        :Parameters:
        
        * **id_or_worker (string or int or BaseWorker)** This is either the id of the object to be returned or the object itself.
        
        :Example:

        >>> from syft.core.hooks import TorchHook
        >>> from syft.core.hooks import torch
        >>> from syft.core.workers import VirtualWorker
        >>> # hook torch and create a local worker
        >>> hook = TorchHook()
        >>> local = hook.local_worker
        Hooking into Torch...
        Overloading complete.
        >>> # lets create a remote worker
        >>> remote = VirtualWorker(hook=hook,id=1)
        >>> local.add_worker(remote)
        >>> remote
        <syft.core.workers.VirtualWorker at 0x10e3a2a90>
        >>> # we can get the worker using it's id (1)
        >>> local.get_worker(1)
        <syft.core.workers.VirtualWorker at 0x10e3a2a90>
        >>> # or we can get the worker by passing in the worker
        >>> local.get_worker(remote)        
        <syft.core.workers.VirtualWorker at 0x10e3a2a90>
        """

        if(issubclass(type(id_or_worker), BaseWorker)):
            if(id_or_worker.id not in self._known_workers):
                self.add_worker(id_or_worker)
            return self._known_workers[id_or_worker.id]
        else:
            return self._known_workers[id_or_worker]

    def get_obj(self, remote_key):
        """get_obj(remote_key) -> a torch.Tensor or torch.autograd.Variable object
        This method fetches a tensor from the worker's internal registry of tensors using its id. Note that this
        will not work on client worker because they do not store objects interanlly. However, it can be used on
        remote workers, including remote :class:`VirtualWorker` objects, as pictured below in the example.

        :Parameters:
        
        * **remote_key (string or int)** This is the id of the object to be returned.
        
        :Example:

        >>> from syft.core.hooks import TorchHook
        >>> from syft.core.hooks import torch
        >>> from syft.core.workers import VirtualWorker
        >>> hook = TorchHook()
        >>> local = hook.local_worker
        >>> remote = VirtualWorker(hook=hook, id=1)
        >>> local.add_worker(remote)
        Hooking into Torch...
        Overloading complete.
        >>> x = torch.FloatTensor([1,2,3,4,5]).send(remote)
        >>> x
        [torch.FloatTensor - Locations:[<syft.core.workers.VirtualWorker object at 0x113f58c50>]]
        >>> x.id
        3214169934
        >>> remote.get_obj(x.id)
        [torch.FloatTensor - Locations:[1]]
        """
        
        return self._objects[remote_key]

    def set_obj(self, remote_key, value, force=False, tmp=False):
        """
        This method adds an object (such as a tensor) to one of the internal object registries. The main registry
        holds objects for a long period of time (until they are explicitly deleted) while the temporary registory
        will delete all object references contained therein when the cleanup method is called (self._clear_tmp_objects).

        
        The reason we have a temporary registry for clijen workers (instead of putting everything in the permanent one) is 
        because when the client creates a reference to an object (x = torch.zeros(10)), we want to ensure that the client 
        contains the only pointer to that object so that when the client deletes the object that __del__ gets called. If, 
        however, the object was also being stored in the permanent registry (self._objects), then when the variable went out
        of scope on the client, the python garbage collector wouldn't call __del__ because the internal registry would
        still have a reference to it. So, we use the temporary registry during the construction of the object but then
        delete all references to the object (other than the client's) once object construction and (recursive) registration
        is complete.

        When the worker is not a client worker (self._is_client_worker==False), this method just saves an object into the
        permanent registry, where it remains until it is explicitly deleted using self._rm_obj.
        
        :Parameters:
        
        * **remote_key (int)** This is an object id to an object to be stored in memory.

        * **value (torch.Tensor or torch.autograd.Variable) ** the object to be stored in memory.

        * **force (bool, optional)** if set to True, this will force the object to be stored in permenent memory, even if 
          the current worker is a client worker (Default: False).

        * **tmp (bobol, optional)** if set to True, this will allow an object to be stored in temporary memory if and only if
          the worker is also a client worker. If set to false, the object will not be stored in temporary memory, even if the 
          worker is a client worker.
        
        :Example:

        >>> from syft.core.hooks import TorchHook
        >>> from syft.core.hooks import torch
        >>> from syft.core.workers import VirtualWorker
        >>> import json
        >>> #
        >>> # hook pytorch and get worker
        >>> hook = TorchHook()
        >>> local = hook.local_worker
        Hooking into Torch...
        Overloading complete.
        >>> # showing that the current worker is a client worker (default)
        >>> local.is_client_worker
        True
        >>> message_obj = json.loads(' {"torch_type": "torch.FloatTensor", "data": [1.0, 2.0, 3.0, 4.0, 5.0], "id": 9756847736, "owners": [1], "is_pointer": false}')
        >>> obj_type = hook.types_guard(message_obj['torch_type'])
        >>> unregistered_tensor = torch.FloatTensor.deser(obj_type,message_obj)
        >>> unregistered_tensor
         1
         2
         3
         4
         5
        [torch.FloatTensor of size 5]
        >>> # calling set_obj on a client worker when force=False (default)
        >>> local.set_obj(unregistered_tensor.id, unregistered_tensor)
        >>> local._objects
        {}
        >>> # calling set_obj on a client worker when force=True
        >>> local.set_obj(unregistered_tensor.id, unregistered_tensor, force=True)
        >>> local._objects
        {3070459025: 
          1
          2
          3
          4
          5
         [torch.FloatTensor of size 5]}
        """


        if(tmp and self.is_client_worker):
            self._tmp_objects[remote_key] = value

        if(not self.is_client_worker or force):
            self._objects[remote_key] = value

    def rm_obj(self, remote_key):
        """
        This method removes an object from the permament object registory if it exists.

        :parameters:

        * **remote_key(int or string)** the id of the object to be removed
        """
        if(remote_key in self._objects):
            del self._objects[remote_key]

    def _clear_tmp_objects(self):
        """
        This method releases all objects from the temporary registry.
        """
        self._tmp_objects = {}

    def send_obj(self, message, recipient):
        """An interface that all extensions of BaseWorker must implement for functionality that sends an object 
        to another worker.
    
        :Parameters:

        * **message (object)** a python object to be sent

        * **recipient (** :class:`BaseWorker` **)** the worker object to send the message to.

        """
        raise NotImplementedError

    def receive_obj(self, message):
        """An interface that all extensions of BaseWorker must implement for functionality that receives an object 
        from another worker.

        :Parameters:

        * **message (object)** a string or binary object received from another worker
        """

        raise NotImplementedError

    def request_obj(self, obj_id, sender):
        """An interface that all extensions of BaseWorker must implement for functionality that requests an object be
        sent from another worker.
        
        :Parameters:

        * **obj_id (str or int)** the id of the object requested.

        * **sender (** :class:`BaseWorker` **)** the worker to request the object from.

        """
        raise NotImplementedError

    def process_response(self, response):
        """process_response(response) -> dict
        Processes a worker's response from a command, converting it from the raw form sent over the wire into
        python objects leading to the execution of a command such as storing a tensor or manipulating it in some way.


        :Parameters:

        * **response(string)** This is the raw message received from a foreign worker or client.

        * **out (dict)** the result of the parsing.

        """
        # TODO: Extend to responses that are iterables.
        # TODO: Fix the case when response contains only a numeric
        
        response = json.loads(response)

        try:
            return (response['registration'], response['torch_type'],
                    response['var_data'], response['var_grad'])
        except:
            return response


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

    # Worker needs to retrieve tensor by ID before computing with it
    def _retrieve_tensor(self, self2, x):
        try:
            return self.get_obj(self._id_tensorvar(x))
        except TypeError:
            try:
                return [self.get_obj(i) for i in self._id_tensorvar(x)]
            except TypeError:
                return x
        except KeyError:
            return x

    def _command_guard(self, command, allowed):
        if command not in allowed:
            raise RuntimeError(
                'Command "{}" is not a supported Torch operation.'.format(command))
        return command

    # # Client needs to identify a tensor before sending commands that use it
    def _id_tensorvar(self, x):
        pat = re.compile('_fl.(.*)')
        try:
            if isinstance(x, str):
                return int(pat.search(x).group(1))
            else:
                return [self._id_tensorvar(i) for i in x]
        except AttributeError:
            return x

class VirtualWorker(BaseWorker):
    r"""
    A virtualized worker simulating the existence of a remote machine. It is intended as a testing,  
    development, and performance evaluation tool that exists independent of a live or local network
    of workers. You don't even have to be connected to the internet to create a pool of Virtual Workers
    and start running computations on them.

    :Parameters:
        
        * **hook (**:class:`.hooks.BaseHook` **)** This is a reference to the hook object which overloaded the 
          underlying deep learning framework.
        
        * **id (int or string, optional)** the integer or string identifier for this node
        
        * **is_client_worker (bool, optional)** a boolean which determines whether this worker is associeted 
          with an end user client. If so, it assumes that the client will maintain control over when 
          tensors/variables/models are instantiated or deleted as opposed to handling tensor/variable/model 
          lifecycle internally.

        * **objects (list of tensors, variables, or models, optional)** 
          When the worker is NOT a client worker, it stores all tensors it receives or creates in this dictionary.
          The key to each object is it's id.

        * **tmp_objects (list of tensors, variables, or models, optional)**
          When the worker IS a client worker, it stores some tensors temporarily in this _tmp_objects simply to ensure
          that they do not get deallocated by the Python garbage collector while in the process of being registered.
          This dictionary can be emptied using the clear_tmp_objects method.
    
        * **known_workers (list of **:class:`BaseWorker` ** objects, optional)** This dictionary can include all known workers.

        * **verbose (bool, optional)** A flag for whether or not to print events to stdout.
    
        :Example:

        >>> from syft.core.hooks import TorchHook
        >>> from syft.core.hooks import torch
        >>> from syft.core.workers import VirtualWorker
        >>> hook = TorchHook()
        Hooking into Torch...
        Overloading complete.
        >>> local = hook.local_worker
        >>> remote = VirtualWorker(id=1, hook=hook)
        >>> local.add_worker(remote)
        >>> x = torch.FloatTensor([1,2,3,4,5])
        >>> x
         1
         2
         3
         4
         5
        [torch.FloatTensor of size 5]
        >>> x.send(remote)
        >>> x
        [torch.FloatTensor - Locations:[<syft.core.workers.VirtualWorker object at 0x11848bda0>]]
        >>> x.get()
        >>> x
         1
         2
         3
         4
         5
        [torch.FloatTensor of size 5]        
        """

    def __init__(self,  hook, id=0, is_client_worker=False, objects={}, tmp_objects={}, known_workers={}, verbose=False):
        super().__init__(hook=hook, id=id, is_client_worker=is_client_worker, objects=objects, tmp_objects=tmp_objects, known_workers=known_workers, verbose=verbose)

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

        return obj, self._clear_tmp_objects

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
        args = utils.map_tuple(self, command_msg['args'], self._retrieve_tensor)
        kwargs = utils.map_dict(self, command_msg['kwargs'], self._retrieve_tensor)
        has_self = command_msg['has_self']
        # TODO: Implement get_owners and refactor to make it prettier
        combined = list(args) + list(kwargs.values())

        if has_self:
            command = self._command_guard(command_msg['command'],
                                          self.hook.tensorvar_methods)
            obj_self = self._retrieve_tensor(self, command_msg['self'])
            combined = combined + [obj_self]
            command = eval('obj_self.{}'.format(command))
        else:
            command = self._command_guard(command_msg['command'], self.torch_funcs)
            command = eval('torch.{}'.format(command))

        # we need the original tensorvar owners so that we can register
        # the result properly later on
        tensorvars = [x for x in combined if type(x).__name__ in self.hook.tensorvar_types_strs]
        _, owners = self.hook.get_owners(tensorvars)

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
