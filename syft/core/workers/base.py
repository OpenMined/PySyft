import torch
import json
import logging
import syft as sy
from abc import ABC, abstractmethod

from .. import utils


# import numbers
# import re
# import random
# import traceback


class BaseWorker(ABC):
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

    def __init__(self, hook=None, id=0, is_client_worker=False, objects={},
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
        self.add_worker(self)

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

        return json.dumps({"id": self.id, "type": type(self)})

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

        message_wrapper_json = (json.dumps(message_wrapper) + "\n").encode()
        self.message_queue = []
        return self._send_msg(message_wrapper_json, recipient)

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

    @abstractmethod
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
        return sy._LocalTensor()

    def receive_msg(self, message_wrapper_json):
        """Receives an message from a worker and then executes its contents appropriately.
        The message is encoded as a binary blob.

        * **message (binary)** the message being sent

        * **out (object)** the response. This can be a variety
          of object types. However, the object is typically only used during testing or
          local development with :class:`VirtualWorker` workers.
        """
        message_wrapper = utils.decode(message_wrapper_json, worker=self)

        response, private = self.process_message_type(message_wrapper)

        response = utils.encode(response, retrieve_pointers=False, private_local=private)

        response = json.dumps(response).encode()

        return response

    def process_message_type(self, message_wrapper):
        """
        This method takes a message wrapper and attempts to process
        it agaist known processing methods. If the method is a composite
        message, it unroles applies recursively

        * **message_wrapper (dict)** Dictionary containing the message
          and meta information

        * **out (object, bool)** the response. This can be a variety
          of object types. However, the object is typically only
          used during testing or local development with
          :class:`VirtualWorker` workers. The bool specifies if the
          response is private or not (private: we don't encode the data
          but juts info on the tensor; not private: we transmit data to be
          acquired by the receiver)
        """
        message = message_wrapper['message']

        # Receiving an object from another worker
        if message_wrapper['type'] == 'obj':
            response = message  # response is a tensorvar
            utils.fix_chain_ends(response)
            utils.assert_is_chain_well_formed(response)
            self.register(response)
            return {}, False

        #  Receiving a request for an object from another worker
        elif message_wrapper['type'] == 'req_obj':
            # Because it was pointed at, it's the first syft_object of the chain,
            # so its parent is the tensorvar
            syft_object = self.get_obj(message)
            tensorvar = syft_object.parent
            if utils.is_variable(syft_object.torch_type):
                syft_data_object = tensorvar.data.child
                self.de_register(syft_data_object)
                if tensorvar.grad is not None:
                    syft_grad_object = tensorvar.grad.child
                    self.de_register(syft_grad_object)
            self.de_register(syft_object)
            return tensorvar, False

        #  A torch command from another worker involving one or more tensors
        #  hosted locally
        elif message_wrapper['type'] == 'torch_cmd':
            result = self.process_command(message)
            self.register(result)
            return result, True  # Result is private

        # A composite command. Must be unrolled
        elif message_wrapper['type'] == 'composite':
            raise NotImplementedError('Composite command not handled at the moment')

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
        if (worker.id in self._known_workers and self.verbose):
            logging.warning(
                "Worker ID " + str(worker.id) + " taken. Have I seen this worker before?")
            logging.warning("Replacing it anyways... this could cause unexpected behavior...")
        if(worker.id in self._known_workers):
            logging.warn("Worker " + str(worker.id) + " already exists. Replacing old worker which could cause"+
                         "unexpected behavior")
        self._known_workers[worker.id] = worker

    def add_workers(self, workers):
        for worker in workers:
            self.add_worker(worker)

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

        if issubclass(type(id_or_worker), BaseWorker):
            if id_or_worker.id not in self._known_workers:
                self.add_worker(id_or_worker)
            result = self._known_workers[id_or_worker.id]
        else:
            if id_or_worker in self._known_workers:
                result = self._known_workers[id_or_worker]
            else:
                logging.warning('Worker', self.id, 'couldnt recognize worker', id_or_worker)
                result = id_or_worker

        if result is not None:
            return result
        else:
            return id_or_worker

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


        """

        if tmp and self.is_client_worker:
            self._tmp_objects[remote_key] = value

        if not self.is_client_worker or force:
            self._objects[remote_key] = value

    def rm_obj(self, remote_key):
        """
        This method removes an object from the permament object registory
        if it exists.

        :parameters:

        * **remote_key(int or string)** the id of the object to be removed
        """
        if remote_key in self._objects:
            del self._objects[remote_key]

    def _clear_tmp_objects(self):
        """
        This method releases all objects from the temporary registry.
        """
        self._tmp_objects = {}

    def de_register(self, obj):

        """
        Unregisters an object and its attribute
        """
        if utils.is_syft_tensor(obj):
            self.rm_obj(obj.id)
        elif utils.is_tensor(obj):
            self.de_register(obj.child)
        elif utils.is_variable(obj):
            self.de_register(obj.child)
            self.de_register(obj.data.child)
        # Case of a iter type non json serializable
        elif isinstance(obj, (list, tuple, set, bytearray, range)):
            for o in obj:
                self.de_register(o)
        elif obj is None:
            "do nothing"
        else:
            raise TypeError('The type', type(obj), 'is not supported at the moment')
        return

    def de_register_object(self, obj, _recurse_torch_objs=True):
        """
        Unregisters an object and removes attributes which are indicative
        of registration. Note that the way in which attributes are deleted
        has been informed by this StackOverflow post: https://goo.gl/CBEKLK
        """
        is_torch_tensor = isinstance(obj, torch.Tensor)

        if not is_torch_tensor:
            if hasattr(obj, 'id'):
                self.rm_obj(obj.id)
                del obj.id
            if hasattr(obj, 'owner'):
                del obj.owner

        if hasattr(obj, 'child'):
            if obj.child is not None:
                if is_torch_tensor:
                    if _recurse_torch_objs:
                        self.de_register_object(obj.child,
                                                _recurse_torch_objs=False)
                else:
                    self.de_register_object(obj.child,
                                            _recurse_torch_objs=_recurse_torch_objs)
            if not is_torch_tensor:
                delattr(obj, 'child')

    def register(self, result):
        """
            Register an object with SyftTensors
        """
        if utils.is_syft_tensor(result):
            syft_obj = result
            self.register_object(syft_obj)
        elif utils.is_tensor(result):
            tensor = result
            self.register_object(tensor.child)
        elif utils.is_variable(result):
            variable = result
            self.register(variable.child)
            self.register(variable.data.child)
        # Case of a iter type non json serializable
        elif isinstance(result, (list, tuple, set, bytearray, range)):
            for res in result:
                self.register(res)
        elif result is None:
            "do nothing"
        else:
            raise TypeError('The type', type(result), 'is not supported at the moment')
        return

    def register_object(self, obj, id=None):
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
        if not utils.is_syft_tensor(obj):
            raise TypeError("Can't register a non-SyftTensor")

        if id is None:
            id = obj.id

        if obj.owner.id == self.id:
            self.set_obj(id, obj)
        else:
            logging.warning("Registering a pointer on non-owned syftTensor.")
            pointer = obj.create_pointer()
            pointer.owner = self
            self.set_obj(pointer.id, pointer)
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
        # try:
        #    obj.data_backup = obj.data
        # except:
        #    ""

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

        utils.assert_has_only_torch_tensorvars(command_msg)

        attr = command_msg['command']
        has_self = command_msg['has_self']
        args = command_msg['args']
        kwargs = command_msg['kwargs']
        self_ = command_msg['self'] if has_self else None

        return self._execute_call(attr, self_, *args, **kwargs)

    def _execute_call(self, attr, self_, *args, **kwargs):
        """
        Transmit the call to the appropriate TensorType for handling
        """

        # Distinguish between a command with torch tensors (like when called by the client,
        # or received from another worker), and a command with syft tensor, which can occur
        # when a function is overloaded by a SyftTensor (for instance _PlusIsMinusTensor
        # overloads add and replace it by sub)
        try:
            utils.assert_has_only_torch_tensorvars((args, kwargs))
            is_torch_command = True
        except AssertionError:
            is_torch_command = False

        has_self = self_ is not None

        if has_self:
            command = self._command_guard(attr, torch.tensorvar_methods)
        else:
            command = self._command_guard(attr, torch.torch_modules)

        raw_command = {
            'command': command,
            'has_self': has_self,
            'args': args,
            'kwargs': kwargs
        }
        if has_self:
            raw_command['self'] = self_
        if is_torch_command:
            # Unwrap the torch wrapper
            syft_command, child_type = utils.prepare_child_command(
                raw_command, replace_tensorvar_with_child=True)
        else:
            # Get the next syft class
            # The actual syft class is the one which redirected (see the  _PlusIsMinus ex.)
            syft_command, child_type = utils.prepare_child_command(
                raw_command, replace_tensorvar_with_child=True)

        utils.assert_has_only_syft_tensors(syft_command)

        # Note: because we have pb of registration of tensors with the right worker,
        # and because having Virtual workers creates even more ambiguity, we specify the worker
        # performing the operation
        result = child_type.handle_call(syft_command, owner=self)

        utils.enforce_owner((raw_command, result), self)

        if is_torch_command:
            # Wrap the result
            if has_self and utils.is_in_place_method(attr):
                wrapper = utils.wrap_command_with(result, raw_command['self'])
            else:
                wrapper = utils.wrap_command(result)
            return wrapper
        else:
            # We don't need to wrap
            return result

    def send_obj(self, object, new_id, recipient, new_data_id=None, new_grad_id=None):
        """send_obj(self, obj, new_id, recipient, new_data_id=None) -> obj
        Sends an object to another :class:`VirtualWorker` and removes it
        from the local worker.

        :Parameters:
        * **object (object)** a python object to be sent
        * **new_id (int)** the id where the object should be stored
        * **recipient (** :class:`VirtualWorker` **)** the worker object to send the message to.

        """

        object.child.id = new_id
        if self.get_pointer_to(recipient, new_id) is not None:
            raise MemoryError('You already point at ', recipient, ':', new_id)
        if utils.is_variable(object.child.torch_type):
            if new_data_id is None or new_grad_id is None:
                raise AttributeError(
                    'Please provide the new_data_id and new_grad_id args, to be able to point to Var.data, .grad')

            if self.get_pointer_to(recipient, new_data_id) is not None:
                raise MemoryError('You already point at ', recipient, ':', new_id)
            assert new_id != new_data_id, \
                "You can't have the same id vor the variable and its data."
            assert new_id != new_grad_id, \
                "You can't have the same id vor the variable and its grad."
            assert new_data_id != new_grad_id

            object.data.child.id = new_data_id
            if object.grad is None:
                object.grad = sy.Variable(sy.zeros(object.size()))
            object.grad.child.id = new_grad_id
        object = utils.encode(object, retrieve_pointers=False, private_local=False)

        # We don't need any response to proceed to registration
        self.send_msg(message=object,
                      message_type='obj',
                      recipient=recipient)

    def send_torch_command(self, recipient, message):
        """send_torch_command(self, recipient, message) -> object

        This method sends a message to another worker in a way that hangs... waiting until the
        worker responds with a message. It then processes the response using a response handler

        :Parameters:

        * **recipient (** :class:`VirtualWorker` **)** the worker being sent a message.

        * **message (string)** the message being sent
        """
        if isinstance(recipient, (str, int)):
            raise TypeError('Recipient should be a worker object not his id.')

        response = self.send_msg(
            message=message,
            message_type='torch_cmd',
            recipient=recipient
        )

        response = utils.decode(response, worker=self)

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

        object = self.send_msg(message=obj_id,
                               message_type='req_obj',
                               recipient=recipient)

        object = utils.decode(object, worker=self)

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

        return object

    def get_pointer_to(self, location, id_at_location):
        # TODO: instead of looping on the objects,
        # with could keep a dict with keys = owners and subkeys id@loc
        # Will be crucial when having lots of variables, but means it has to be updated
        if not isinstance(location, (int, str)):
            location = location.id

        for key, syft_tensor in self._objects.items():
            if isinstance(syft_tensor, sy._PointerTensor):
                if syft_tensor.location.id == location \
                        and syft_tensor.id_at_location == id_at_location:
                    return syft_tensor

    @classmethod
    def _command_guard(cls, command, allowed):
        if isinstance(allowed, dict):
            allowed_names = []
            for module_name, func_names in allowed.items():
                for func_name in func_names:
                    allowed_names.append(module_name + '.' + func_name)
            allowed = allowed_names
        if command not in allowed:
            raise RuntimeError(
                'Command "{}" is not a supported Torch operation.'.format(command))
        return command

    @classmethod
    def _is_command_valid_guard(cls, command, allowed):
        try:
            cls._command_guard(command, allowed)
        except RuntimeError:
            return False
        return True
