import json
import numbers
import re
import random

from .. import utils


class BaseWorker(object):
    r"""
    The BaseWorker class establishes a consistent interface for
    communicating between different machines
    about tensors, variables, models, and other network related
    information. It defines base functionality
    for storing objects within the worker (or in some cases
    specifically not storing them), and for keeping
    track of known workers across a network.

    This class does NOT include any communication specific functionality
    such as peer-to-peer protocols, routing, node discovery, or even
    socket architecture. Only classes that extend the BaseWorker should
    carry such functionality.

    :Parameters:

        * **hook (**:class:`.hooks.BaseHook` **)** This is a reference
          to the hook object which overloaded the underlying deep
          learning framework.

        * **id (int or string, optional)** the integer or string identifier
          for this node

        * **is_client_worker (bool, optional)** a boolean which determines
          whether this worker is associeted with an end user client.
          If so, it assumes that the client will maintain control over when
          tensors/variables/models are instantiated or deleted as opposed
          to handling tensor/variable/model lifecycle internally.

        * **objects (list of tensors, variables, or models, optional)**
          When the worker is NOT a client worker, it stores all tensors
          it receives or creates in this dictionary.
          The key to each object is it's id.

        * **tmp_objects (list of tensors, variables, or models, optional)**
          When the worker IS a client worker, it stores some tensors
          temporarily in this _tmp_objects simply to ensure
          that they do not get deallocated by the Python garbage
          collector while in the process of being registered.
          This dictionary can be emptied using the clear_tmp_objects method.

        * **known_workers (list of **:class:`BaseWorker` ** objects, optional)**
          This dictionary can include all known workers on a network.
          Some extensions of BaseWorker will use this
          list to bootstrap the network.

        * **verbose (bool, optional)** A flag for whether or not to
          print events to stdout.

    """

    def __init__(self,  hook=None, id=0, is_client_worker=False, objects={},
                 tmp_objects={}, known_workers={}, verbose=True, queue_size=0):

        # This is a reference to the hook object which overloaded
        # the underlying deep learning framework
        # (at the time of writing this is exclusively TorchHook)
        self.hook = hook

        # the integer or string identifier for this node
        self.id = id

        # a boolean which determines whether this worker is
        # associeted with an end user client. If so, it assumes
        # that the client will maintain control over when varialbes
        # are instantiated or deleted as opposed to
        # handling tensor/variable/model lifecycle internally.
        self.is_client_worker = is_client_worker

        # The workers permanenet registry. When the worker is NOT
        # a client worker, it stores all tensors it receives
        #  or creates in this dictionary. The key to each object
        # is it's id.
        self._objects = {}
        for k, v in objects.items():
            self._objects[k] = v

        # The temporary registry. When the worker IS a client
        # worker, it stores some tensors temporarily in this
        # _tmp_objects simply to ensure that they do not get
        # deallocated by the Python garbage collector while
        # in the process of being registered. This dictionary
        # can be emptied using the clear_tmp_objects method.
        self._tmp_objects = {}
        for k, v in tmp_objects.items():
            self._tmp_objects[k] = v

        # This dictionary includes all known workers on a
        # network. Extensions of BaseWorker will include advanced
        # functionality for adding to this dictionary (node discovery).
        # In some cases, one can initialize this with
        # known workers to help bootstrap the network.
        self._known_workers = {}
        for k, v in known_workers.items():
            self._known_workers[k] = v

        # A flag for whether or not to print events to stdout.
        self.verbose = verbose

        # A list for storing messages to be sent as well as the max size of the list
        self.message_queue = []
        self.queue_size = queue_size

    def whoami(self):
        """Returns metadata information about the worker. This function returns the default
        which is the id and type of the current worker. Other worker types can extend this
        function with additional metadata such as network information.
        """

        return json.dumps({"id": self.id, "type":type(self)})

    def send_msg(self, message, message_type, recipient):
        """Sends a string message to another worker with message_type information
        indicating how the message should be processed.

        :Parameters:

        * **recipient (** :class:`VirtualWorker` **)** the worker being sent a message.

        * **message (string)** the message being sent

        * **message_type (string)** the type of message being sent. This affects how
          the message is processed by the recipient. The types of message are described
          in :func:`receive_msg`.

        * **out (object)** the response from the message being sent. This can be a variety
          of object types. However, the object is typically only used during testing or
          local development with :class:`VirtualWorker` workers.
        """
        message_wrapper = {}
        message_wrapper['message'] = message
        message_wrapper['type'] = message_type
        self.message_queue.append(message_wrapper)
        if self.queue_size:
            if len(self.message_queue) > self.queue_size:
                message_wrapper = self.compile_composite_message()
            else:
                return None

        message_wrapper_json = json.dumps(message_wrapper) + "\n"

        message_wrapper_json_binary = message_wrapper_json.encode()

        self.message_queue = []
        return self._send_msg(message_wrapper_json_binary, recipient)

    def compile_composite_message(self):
        """
        Returns a composite message in a dictionary from the message queue.
        Evenatually will take a recipient id.

        * **out (dict)** dictionary containing the message queue compiled
        as a composite message
        """

        message_wrapper = {}

        message_wrapper['message'] = {
            message_number: message for message_number, message in enumerate(self.message_queue)}
        message_wrapper['type'] = 'composite'

        return message_wrapper

    def _send_msg(self, message_wrapper_json_binary, recipient):
        """Sends a string message to another worker with message_type information
        indicating how the message should be processed.

        :Parameters:

        * **recipient (** :class:`VirtualWorker` **)** the worker being sent a message.

        * **message_wrapper_json_binary (binary)** the message being sent encoded in binary

        * **out (object)** the response from the message being sent. This can be a variety
          of object types. However, the object is typically only used during testing or
          local development with :class:`VirtualWorker` workers.
        """
        raise NotImplementedError

    def receive_msg(self, message_wrapper_json, is_binary=True):
        """Receives an message from a worker and then executes its contents appropriately.
        The message is encoded as a binary blob.

        * **message (binary)** the message being sent

        * **out (object)** the response. This can be a variety
          of object types. However, the object is typically only used during testing or
          local development with :class:`VirtualWorker` workers.
        """

        if(is_binary):
            message_wrapper_json = message_wrapper_json.decode('utf-8')
        message_wrapper = json.loads(message_wrapper_json)

        return self.process_message_type(message_wrapper)

    def process_message_type(self, message_wrapper):
        """
        This method takes a message wrapper and attempts to process
        it agaist known processing methods. If the method is a composite
        message, it unroles applies recursively

        * **message_wrapper (dict)** Dictionary containing the message
          and meta information

        * **out (object)** the response. This can be a variety
          of object types. However, the object is typically only
          used during testing or local development with
          :class:`VirtualWorker` workers.
        """
        message = message_wrapper['message']

        # Receiving an object from another worker
        if(message_wrapper['type'] == 'obj'):
            response = self.receive_obj(message)  # DONE!
            return response.ser()

        #  Receiving a request for an object from another worker
        elif(message_wrapper['type'] == 'req_obj'):
            return self.prepare_send_object(self.get_obj(message))

        #  A torch command from another workerinvolving one or more tensors
        #  hosted locally
        elif(message_wrapper['type'] == 'torch_cmd'):
            return json.dumps(self.handle_command(message)) + "\n"
        # A composite command. Must be unrolled
        elif(message_wrapper['type'] == 'composite'):
            return [self.process_message_type(message[message_number])
                    for message_number in message]

        return "Unrecognized message type:" + message_wrapper['type']

    def __str__(self):
        out = "<"
        out += str(type(self)).split("'")[1]
        out += " id:" + str(self.id)
        out += ">"
        return out

    def __repr__(self):
        return self.__str__()

    def add_worker(self, worker):
        """add_worker(worker) -> None
        This method adds a worker to the list of _known_workers
        internal to the BaseWorker. It endows this class with
        the ability to communicate with the remote worker being
        added, such as sending and receiving objects, commands,
        or information about the network.

        :Parameters:

        * **worker (**:class:`BaseWorker` **)** This is an object
          pointer to a remote worker, which must have a unique id.

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
        [torch.FloatTensor - Locations:[<VirtualWorker at 0x11848bda0>]]
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
            print("WARNING: Worker ID " + str(worker.id) +
                  " taken. Have I seen this worker before?")
            print(
                "WARNING: Replacing it anyways... this could cause unexpected behavior...")

        self._known_workers[worker.id] = worker

    def get_worker(self, id_or_worker):
        """get_worker(self, id_or_worker) -> BaseWorker
        If you pass in an ID, it will attempt to find the worker
        object (pointer) withins self._known_workers. If you
        instead pass in a pointer itself, it will save that as a
        known_worker if it does not exist as one. This method is primarily useful
        becuase often tensors have to store only the ID to a
        foreign worker which may or may not be known
        by the worker that is deserializing it at the time
        of deserialization. This method allows for resolution of
        worker ids to workers to happen automatically while also
        making the current worker aware of new ones when discovered
        through other processes.


        :Parameters:

        * **id_or_worker (string or int or** :class:`BaseWorker` **)**
          This is either the id of the object to be returned or the object itself.

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
        """get_obj(remote_key) -> a torch object
        This method fetches a tensor from the worker's internal
        registry of tensors using its id. Note that this
        will not work on client worker because they do not
        store objects interanlly. However, it can be used on
        remote workers, including remote :class:`VirtualWorker`
        objects, as pictured below in the example.

        :Parameters:

        * **remote_key (string or int)** This is the id of
          the object to be returned.

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
        [torch.FloatTensor - Locations:[<VirtualWorker at 0x113f58c50>]]
        >>> x.id
        3214169934
        >>> remote.get_obj(x.id)
        [torch.FloatTensor - Locations:[1]]
        """

        return self._objects[int(remote_key)]

    def set_obj(self, remote_key, value, force=False, tmp=False):
        """
        This method adds an object (such as a tensor) to one of
        the internal object registries. The main registry
        holds objects for a long period of time (until they are
        explicitly deleted) while the temporary registory
        will delete all object references contained therein when
        the cleanup method is called (self._clear_tmp_objects).


        The reason we have a temporary registry for clijen workers
        (instead of putting everything in the permanent one) is
        because when the client creates a reference to an object
        (x = torch.zeros(10)), we want to ensure that the client
        contains the only pointer to that object so that when the
        client deletes the object that __del__ gets called. If,
        however, the object was also being stored in the permanent
        registry (self._objects), then when the variable went out
        of scope on the client, the python garbage collector wouldn't
        call __del__ because the internal registry would
        still have a reference to it. So, we use the temporary registry
        during the construction of the object but then
        delete all references to the object (other than the client's)
        once object construction and (recursive) registration
        is complete.

        When the worker is not a client worker (self._is_client_worker==False),
        this method just saves an object into the permanent registry,
        where it remains until it is explicitly deleted using self._rm_obj.

        :Parameters:

        * **remote_key (int)** This is an object id to an object to be
          stored in memory.

        * **value (torch.Tensor or torch.autograd.Variable) ** the
          object to be stored in memory.

        * **force (bool, optional)** if set to True, this will force the
          object to be stored in permenent memory, even if
          the current worker is a client worker (Default: False).

        * **tmp (bobol, optional)** if set to True, this will allow an object
          to be stored in temporary memory if and only if
          the worker is also a client worker. If set to false, the object
          will not be stored in temporary memory, even if the
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
        >>> message_obj = json.loads(' {"torch_type": "torch.FloatTensor", \
"data": [1.0, 2.0, 3.0, 4.0, 5.0], "id": 9756847736, "owners": \
[1], "is_pointer": false}')
        >>> obj_type = hook.guard.types_guard(message_obj['torch_type'])
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
        This method removes an object from the permament object registory
        if it exists.

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

    def process_response(self, response):
        """process_response(response) -> dict
        Processes a worker's response from a command, converting it
        from the raw form sent over the wire into python objects
        leading to the execution of a command such as storing a
        tensor or manipulating it in some way.


        :Parameters:

        * **response(string)** This is the raw message received from
          a foreign worker or client.

        * **out (dict)** the result of the parsing.

        """
        # TODO: Extend to responses that are iterables.
        # TODO: Fix the case when response contains only a numeric

        response = json.loads(response)
        if(isinstance(response, str)):
            response = json.loads(response)

        if('registration' in response and 'torch_type' in response):
            return (response['registration'], response['torch_type'],
                    response['var_data'], response['var_grad'])
        else:
            return response

    def register_object(self, obj, force_attach_to_worker=False,
                        temporary=False, **kwargs):
        """
        Registers an object with the current worker node. Selects an
        id for the object, assigns a list of owners,
        and establishes whether it's a pointer or not. This method
        is generally not used by the client and is
        instead used by interal processes (hooks and workers).

        :Parameters:

        * **obj (a torch.Tensor or torch.autograd.Variable)** a Torch
          instance, e.g. Tensor or Variable to be registered

        * **force_attach_to_worker (bool)** if set to True, it will
          force the object to be stored in the worker's permanent registry

        * **temporary (bool)** If set to True, it will store the object
          in the worker's temporary registry.

        :kwargs:

        * **id (int or string)** random integer between 0 and 1e10 or
          string uniquely identifying the object.

        * **owners (list of ** :class:`BaseWorker` objects ** or ids)**
          owner(s) of the object

        * **is_pointer (bool, optional)** Whether or not the tensor being
          registered contains the data locally or is instead a pointer to
          a tensor that lives on a different worker.
        """
        # TODO: Assign default id more intelligently (low priority)
        #       Consider popping id from long list of unique integers

        keys = kwargs.keys()

        # DO NOT DELETE THIS TRY/CATCH UNLESS YOU KNOW WHAT YOU'RE DOING
        # PyTorch tensors wrapped invariables (if my_var.data) are python
        # objects that get deleted and re-created randomly according to
        # the whims of the PyTorch wizards. Thus, our attributes were getting
        # deleted with them (because they are not present in the underlying
        # C++ code.) Thus, so that these python objects do NOT get garbage
        # collected, we're creating a secondary reference to them from the
        # parent Variable object (which we have been told is stable). This
        # is experimental functionality but seems to solve the symptoms we
        # were previously experiencing.
        try:
            obj.data_backup = obj.data
        except:
            ""

        obj.id = (kwargs['id']
                  if ('id' in keys and kwargs['id'] is not None)
                  else random.randint(0, 1e10))

        obj.owners = (kwargs['owners']
                      if 'owners' in keys
                      else [self.id])

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

        mal_points_away = obj.is_pointer and self.id in obj.owners
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
                'Invalid registry: is_pointer is {} but owners is {} on tensor {}'.format(
                    obj.is_pointer, obj.owners, obj.id))
        # print("setting object:" + str(obj.id))
        self.set_obj(obj.id, obj, force=force_attach_to_worker, tmp=temporary)

        # Perform recursive operations.
        # If there is a child tensor (self.data)
        if(hasattr(obj, 'grad')):
            if(obj.grad is not None):
                # import pdb; pdb.set_trace()
                self.register_object(obj=obj.grad,
                                     force_attach_to_worker=force_attach_to_worker,
                                     temporary=temporary,
                                     id=obj.grad.id,
                                     owners=obj.owners,
                                     is_pointer=obj.is_pointer)
        try:
            _ = obj.data
            _ = type(_)
            if(obj.data is not None):
                self.register_object(obj=obj.data,
                                     force_attach_to_worker=force_attach_to_worker,
                                     temporary=temporary,
                                     id=obj.data.id,
                                     owners=obj.owners,
                                     is_pointer=obj.is_pointer)

        except RuntimeError:
            ""

        return obj

    def process_command(self, command_msg):
        """process_command(self, command_msg) -> (command output, list of owners)
        Process a command message from a client worker. Returns the
        result of the computation and a list of the result's owners.

        :Parameters:

        * **command_msg (dict)** The dictionary containing a
          command from another worker.

        * **out (command output, list of** :class:`BaseWorker`
          ids/objects **)** This executes the command
          and returns its output along with a list of
          the owners of the tensors involved.
        """
        # Args and kwargs contain special strings in place of tensors
        # Need to retrieve the tensors from self.worker.objects
        args = utils.map_tuple(
            self, command_msg['args'], self._retrieve_tensor)
        kwargs = utils.map_dict(
            self, command_msg['kwargs'], self._retrieve_tensor)
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
            command = self._command_guard(
                command_msg['command'], self.torch_funcs)
            command = eval('torch.{}'.format(command))

        # we need the original tensorvar owners so that we can register
        # the result properly later on
        tensorvars = [x for x in combined if type(
            x).__name__ in self.hook.tensorvar_types_strs]
        owners = list(
            set([owner for tensorvar in tensorvars for owner in tensorvar.owners]))

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
                result = self.register_object(
                    result, id=result.id, owners=owners)
            except AttributeError:
                result = self.register_object(result, owners=owners)

            registration = dict(id=result.id,
                                owners=owners, is_pointer=True)

            return dict(registration=registration, torch_type=torch_type,
                        var_data=var_data, var_grad=var_grad)

        except AttributeError as e:
            # result is occasionally a sequence of tensors or variables

            return [self.compile_result(x, owners) for x in result]

    def handle_command(self, message):
        """
        Main function that handles incoming torch commands.
        """

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

    def handle_register(self, torch_object, obj_msg, force_attach_to_worker=False, temporary=False):
        """
        This function is responsible for re-registering an object when it has
        been previously registered with the wrong id.

        :Parameters:

        * **torch_object (torch.Tensor or torch.autograd.Variable)** the object
        to be re-registered.

        * **obj_msg** (dict)** the message containing the proper id.

        * **force_attach_to_worker (bool)** if set to True, it will force the object
        to be stored in the worker's permanent registry
          even if the worker is a client worker.

        * **temporary (bool)** If set to True, it will store the object in the
        worker's temporary registry.

        * **out (torch.Tensor or torch.autograd.Variable)** returns the object
        newly registered.
        """

        # TODO: pass in just the id instead of the entire obj_msg.

        try:
            # TorchClient case
            # delete registration from init; it's got the wrong id
            self.rm_obj(torch_object.id)
        except (AttributeError, KeyError):
            # Worker case: v was never formally registered
            pass
        if('is_pointer' in obj_msg and obj_msg['is_pointer']):
            torch_object = self.register_object(torch_object,
                                                id=obj_msg['id'],
                                                owners=obj_msg['owners'],
                                                force_attach_to_worker=force_attach_to_worker,
                                                temporary=temporary,
                                                is_pointer=True)
        else:
            torch_object = self.register_object(torch_object,
                                        id=obj_msg['id'],
                                        owners=[self.id],
                                        force_attach_to_worker=force_attach_to_worker,
                                        temporary=temporary)

        return torch_object

    def prepare_send_object(self, obj, delete_local=True, send_pointer=False):

        obj_json = obj.ser(include_data=not send_pointer)

        if(delete_local):
            self.rm_obj(obj.id)

        return obj_json

    def send_obj(self, obj, recipient, delete_local=True, send_pointer=True):
        """send_obj(self, obj, recipient, delete_local=True) -> obj
        Sends an object to another :class:`VirtualWorker` and, by default, removes it
        from the local worker. It also returns the object as a special case when
        the caller is a client. In most cases, send_obj would be handled
        on the other side by storing it in the permament registry. However, for
        VirtualWorkers attached to clients, we don't want this to occur. Thus,
        this method returns the object as a workaround. See :func:`VirtualWorker.request_obj`
        for more deatils.

        :Parameters:

        * **obj (object)** a python object to be sent

        * **recipient (** :class:`VirtualWorker` **)** the worker object to send the message to.

        * **delete_local (bool, optional)** when set to true, it deletes the version of the
        object in the local registry.

        """

        # obj = recipient.receive_obj(obj.ser())
        _obj = self.send_msg(message=self.prepare_send_object(obj, delete_local, send_pointer=send_pointer),
                             message_type='obj',
                             recipient=recipient)

        if(delete_local):
            self.rm_obj(obj.id)

        return _obj

    def receive_obj(self, message):
        """receive_obj(self, message) -> (a torch.autograd.Variable or torch.Tensor object)
        Functionality that receives a Tensor or Variable from another VirtualWorker
        (as a string), deserializes it, and registers it within the local permanent registry.

        :Parameters:

        * **message(JSON string)** the message encoding the object being received.


        """

        message_obj = json.loads(message)
        obj_type = self.hook.guard.types_guard(message_obj['torch_type'])
        obj = obj_type.deser(obj_type, message_obj)

        self.handle_register(obj, message_obj, force_attach_to_worker=True)

        return obj

    def send_torch_command(self, recipient, message):
        """send_torch_command(self, recipient, message, response_handler, timeout=10) -> object

        This method sends a message to another worker in a way that hangs... waiting until the
        worker responds with a message. It then processes the response using a response handler

        :Parameters:

        * **recipient (** :class:`VirtualWorker` **)** the worker being sent a message.

        * **message (string)** the message being sent
        """
        response = self.send_msg(
            message=message, message_type='torch_cmd', recipient=recipient)
        response = self.process_response(response)
        return response

    def request_obj(self, obj_id, recipient):
        """request_obj(self, obj_id, sender)
        This method requests that another VirtualWorker send an object to the local one.
        In the case that the local one is a client,
        it simply returns the object. In the case that the local worker is not a client,
        it stores the object in the permanent registry.

        :Parameters:

        * **obj_id (str or int)** the id of the object being requested

        * **sender (** :class:`VirtualWorker` **)** the worker who currently has the
          object who is being requested to send it.
        """

        # resolves IDs to worker objects
        recipient = self.get_worker(recipient)

        obj_json = self.send_msg(
            message=obj_id, message_type='req_obj', recipient=recipient)
        obj = self.receive_obj(obj_json)

        # for some reason, when returning obj from request_obj method, the gradient
        # (obj.grad) gets re-initialized without being re-registered and as a
        # consequence does not have an id, causing the x.grad.id to fail because
        # it does not exist. As a result, we've needed to store objects temporarily
        # in self._tmpobjects which seems to fix it. Super strange bug which took
        # multiple days to figure out. The true cause is still unknown but this
        # workaround seems to work well for now. Anyway, so we need to return a cleanup
        # method which is called immediately before this is returned to the client.
        # Note that this is ONLY necessary for the client (which doesn't store objects
        # in self._objects)

        return obj, self._clear_tmp_objects

    # Helpers for HookService and TorchService
    @staticmethod
    def _check_workers(torch_obj, workers):
        if type(workers) is str:
            workers = [workers]
        if issubclass(type(workers), BaseWorker):
            workers = [workers]
        elif not hasattr(workers, '__iter__'):
            raise TypeError(
                """Can only send {} to a string worker ID or an iterable of
                string worker IDs, not {}""".format(torch_obj.__name__, workers)
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
