import logging
import random
import sys

from abc import abstractmethod
import syft as sy
from syft import serde
from syft.frameworks.torch.tensors.interpreters import PointerTensor
from syft.exceptions import WorkerNotFoundException
from syft.workers import AbstractWorker
from syft.codes import MSGTYPE


class BaseWorker(AbstractWorker):
    """Contains functionality to all workers.

    Other workers will extend this class to inherit all functionality necessary
    for PySyft's protocol. Extensions of this class overrides two key methods
    _send_msg() and _recv_msg() which are responsible for defining the
    procedure for sending a binary message to another worker.

    At it's core, BaseWorker (and all workers) is a collection of objects owned
    by a certain machine. Each worker defines how it interacts with objects on
    other workers as well as how other workers interact with objects owned by
    itself. Objects are either tensors or of any type supported by the PySyft
    protocol.

    Args:
        hook: A reference to the TorchHook object which is used
            to modify PyTorch with PySyft's functionality.
        id: An optional string or integer unique id of the worker.
        known_workers: An optional dictionary of all known workers on a
            network which this worker may need to communicate with in the
            future. The key of each should be each worker's unique ID and
            the value should be a worker class which extends BaseWorker.
            Extensions of BaseWorker will include advanced functionality
            for adding to this dictionary(node discovery). In some cases,
            one can initialize this with known workers to help bootstrap
            the network.
        data: Initialize workers with data on creating worker object
        is_client_worker: An optional boolean parameter to indicate
            whether this worker is associated with an end user client. If
            so, it assumes that the client will maintain control over when
            variables are instantiated or deleted as opposed to handling
            tensor/variable/model lifecycle internally. Set to True if this
            object is not where the objects will be stored, but is instead
            a pointer to a worker that eists elsewhere.
        log_msgs: An optional boolean parameter to indicate whether all
            messages should be saved into a log for later review. This is
            primarily a development/testing feature.
    """

    def __init__(
        self,
        hook,
        id=0,
        known_workers={},
        data={},
        is_client_worker=False,
        log_msgs=False,
        verbose=False,
    ):
        """Initializes a BaseWorker."""

        self.hook = hook
        self.torch = None if hook is None else hook.torch
        self.id = id
        self.is_client_worker = is_client_worker
        self.log_msgs = log_msgs
        self.verbose = verbose
        self.msg_history = list()
        # A core object in every BaseWorker instantiation. A Collection of
        # objects where all objects are stored using their IDs as keys.
        self._objects = {}
        self._known_workers = {}
        for k, v in known_workers.items():
            self._known_workers[k] = v
        self.add_worker(self)
        # For performance, we cache each
        self._message_router = {
            MSGTYPE.CMD: self.execute_command,
            MSGTYPE.OBJ: self.set_obj,
            MSGTYPE.OBJ_REQ: self.respond_to_obj_req,
            MSGTYPE.OBJ_DEL: self.rm_obj,
            MSGTYPE.IS_NONE: self.is_tensor_none,
        }
        self.load_data(data)

    # SECTION: Methods which MUST be overridden by subclasses
    @abstractmethod
    def _send_msg(self, message, location):
        """Sends message from one worker to another.

        As BaseWorker implies, you should never instantiate this class by
        itself. Instead, you should extend BaseWorker in a new class which
        instantiates _send_msg and _recv_msg, each of which should specify the
        exact way in which two workers communicate with each other. The easiest
        example to study is VirtualWorker.

        Args:
            message: A string representing the message being sent from one
                worker to another.
            location: A BaseWorker instance that lets you provide the
                destination to send the message.

        Raises:
            NotImplementedError: Method not implemented error.
        """

        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def _recv_msg(self, message):
        """Receives the message.

        As BaseWorker implies, you should never instantiate this class by
        itself. Instead, you should extend BaseWorker in a new class which
        instantiates _send_msg and _recv_msg, each of which should specify the
        exact way in which two workers communicate with each other. The easiest
        example to study is VirtualWorker.

        Args:
            message: A string representing the message being received.

        Raises:
            NotImplementedError: Method not implemented error.

        """
        raise NotImplementedError  # pragma: no cover

    def load_data(self, data):
        """Allows workers to be initialized with data when created 
        
           The method registers the tensor individual tensor objects.
        
        Args:
            
            data: A list of tensors

            
        """

        for tensor in data:

            self.register_obj(tensor)
            tensor.owner = self

    def send_msg(self, msg_type, message, location):
        """Implements the logic to send messages.

        The message is serialized and sent to the specified location. The
        response from the location (remote worker) is deserialized and
        returned back.

        Every message uses this method.

        Args:
            msg_type: A integer representing the message type.
            message: A string representing the message being received.
            location: A BaseWorker instance that lets you provide the
                destination to send the message.

        Returns:
            The deserialized form of message from the worker at specified
            location.
        """
        if self.verbose:
            print(f"worker {self} sending {msg_type} {message} to {location}")
        # Step 0: combine type and message
        message = (msg_type, message)

        # Step 1: serialize the message to simple python objects
        bin_message = serde.serialize(message)

        # Step 2: send the message and wait for a response
        bin_response = self._send_msg(bin_message, location)

        # Step 3: deserialize the response
        response = serde.deserialize(bin_response, worker=self)

        return response

    def recv_msg(self, bin_message):
        """Implements the logic to receive messages.

        The binary message is deserialized and routed to the appropriate
        function. And, the response serialized the returned back.

        Every message uses this method.

        Args:
            bin_message: A binary serialized message.

        Returns:
            A binary message response.
        """

        # Step -1: save message if log_msgs ==  True
        if self.log_msgs:
            self.msg_history.append(bin_message)

        # Step 0: deserialize message
        (msg_type, contents) = serde.deserialize(bin_message, worker=self)
        if self.verbose:
            print(f"worker {self} received {msg_type} {contents}")
        # Step 1: route message to appropriate function
        response = self._message_router[msg_type](contents)

        # # Step 2: If response in none, set default
        # TODO: not sure if someone needed this - if this comment
        # is still here after Feb 15, 2018, please delete these
        # two lines of (commented out) code.
        # if response is None:
        #     response = None

        # Step 3: Serialize the message to simple python objects
        bin_response = serde.serialize(response)
        return bin_response

        # SECTION:recv_msg() uses self._message_router to route to these methods
        # Each method corresponds to a MsgType enum.

    def send(self, tensor, workers, ptr_id=None):
        """Sends tensor to the worker(s).

        Send a syft or torch tensor and his child, sub-child, etc (all the
        syft chain of children) to a worker, or a list of workers, with a given
        remote storage address.

        Args:
            tensor: A  Tensor object representing torch or syft tensor to send.
            workers: A BaseWorker object representing the worker(s) that will
                receive the object.
            ptr_id: An optional string or integer indicating the remote id of
                the object on the remote worker(s).

        Example:
            >>> import torch
            >>> import syft as sy
            >>> hook = sy.TorchHook(torch)
            >>> bob = sy.VirtualWorker(hook)
            >>> x = torch.Tensor([1, 2, 3, 4])
            >>> x.send(bob, 1000)
            Will result in bob having the tensor x with id 1000

        Returns:
            A PointerTensor object representing the pointer to the remote worker(s).
        """

        if not isinstance(workers, list):
            workers = [workers]

        assert len(workers) > 0, "Please provide workers to receive the data"

        if len(workers) == 1:
            worker = workers[0]
        else:
            # If multiple workers are provided , you want to send the same tensor
            # to all the workers. You'll get multiple pointers, or a pointer
            # with different locations
            raise NotImplementedError(
                "Sending to multiple workers is not \
                                        supported at the moment"
            )

        worker = self.get_worker(worker)

        if ptr_id is None:  # Define a remote id if not specified
            ptr_id = int(10e10 * random.random())

        pointer = tensor.create_pointer(
            owner=self, location=worker, id_at_location=tensor.id, register=True, ptr_id=ptr_id
        )

        # Send the object
        self.send_obj(tensor, worker)

        return pointer

    def execute_command(self, message):
        """
        Execute commands received from other workers
        :param message: the message specifying the command and the args
        :return: a pointer to the result
        """

        command, _self, args, kwargs = message

        # TODO add kwargs
        kwargs = {}
        command = command.decode("utf-8")
        # Handle methods
        if _self is not None:
            if sy.torch.is_inplace_method(command):
                getattr(_self, command)(*args, **kwargs)
                return
            else:
                tensor = getattr(_self, command)(*args, **kwargs)
        # Handle functions
        else:
            # At this point, the command is ALWAYS a path to a
            # function (i.e., torch.nn.functional.relu). Thus,
            # we need to fetch this function and run it.

            sy.torch.command_guard(command, "torch_modules")

            paths = command.split(".")
            command = self
            for path in paths:
                command = getattr(command, path)

            tensor = command(*args, **kwargs)

        # some functions don't return anything (such as .backward())
        # so we need to check for that here.
        if tensor is not None:

            # FIXME: should be added automatically
            tensor.owner = self

            # TODO: Handle when the response is not simply a tensor
            # don't re-register tensors if the operation was inline
            # not only would this be inefficient, but it can cause
            # serious issues later on
            # if(_self is not None):
            #     if(tensor.id != _self.id):
            self.register_obj(tensor)

            pointer = tensor.create_pointer(
                location=self,
                id_at_location=tensor.id,
                register=True,
                owner=self,
                ptr_id=tensor.id,
                garbage_collect_data=False,
            )
            return pointer

    def send_command(self, recipient, message):
        """
        Send a command through a message to a recipient worker
        :param recipient:
        :param message:
        :return:
        """

        response = self.send_msg(MSGTYPE.CMD, message, location=recipient)

        return response

    def set_obj(self, obj):
        """Adds an object to the registry of objects.

        Args:
            obj: A torch or syft tensor with an id
        """
        self._objects[obj.id] = obj

    def get_obj(self, obj_id):
        """Returns the object from registry.

        Look up an object from the registry using its ID.

        Args:
            obj_id: A string or integer id of an object to look up.
        """

        try:
            obj = self._objects[obj_id]

        except KeyError as e:

            if obj_id not in self._objects:
                msg = 'Tensor "' + str(obj_id) + '" not found on worker "' + str(self.id) + '"!!! '
                msg += (
                    "You just tried to interact with an object ID:"
                    + str(obj_id)
                    + " on worker "
                    + str(self.id)
                    + " which does not exist!!! "
                )
                msg += (
                    "Use .send() and .get() on all your tensors to make sure they're"
                    "on the same machines. "
                    "If you think this tensor does exist, check the ._objects dictionary"
                    "on the worker and see for yourself!!! "
                    "The most common reason this error happens is because someone calls"
                    ".get() on the object's pointer without realizing it (which deletes "
                    "the remote object and sends it to the pointer). Check your code to "
                    "make sure you haven't already called .get() on this pointer!!!"
                )
                raise KeyError(msg)
            else:
                raise e

        # An object called with get_obj will be "with high probability" serialized
        # and sent back, so it will be GCed but remote data is any shouldn't be
        # deleted
        if hasattr(obj, "child") and isinstance(obj.child, PointerTensor):
            obj.child.garbage_collect_data = False

        return obj

    def respond_to_obj_req(self, obj_id):
        """Returns the deregistered object from registry.

        Args:
            obj_id: A string or integer id of an object to look up.
        """

        obj = self.get_obj(obj_id)
        self.de_register_obj(obj)
        return obj

    def register_obj(self, obj, obj_id=None):
        """Registers the specified object with the current worker node.

        Selects an id for the object, assigns a list of owners, and establishes
        whether it's a pointer or not. This method is generally not used by the
        client and is instead used by internal processes (hooks and workers).

        Args:
            obj: A torch Tensor or Variable object to be registered.
            obj_id (int or string): random integer between 0 and 1e10 or
            string uniquely identifying the object.
        """
        if not self.is_client_worker:
            if obj_id is not None:
                obj.id = obj_id
            self.set_obj(obj)

    def de_register_obj(self, obj, _recurse_torch_objs=True):
        """Deregisters the specified object.

        Deregister and remove attributes which are indicative of registration.

        Args:
            obj: A torch Tensor or Variable object to be deregistered.
            _recurse_torch_objs: A boolean indicating whether the object is
                more complex and needs to be explored. Is not supported at the
                moment.
        """

        if hasattr(obj, "id"):
            self.rm_obj(obj.id)
        if hasattr(obj, "_owner"):
            del obj._owner

    def rm_obj(self, remote_key):
        """Removes an object.

        Remove the object from the permanent object registry if it exists.

        Args:
            remote_key: A string or integer representing id of the object to be
                removed.
        """
        if remote_key in self._objects:
            del self._objects[remote_key]

    # SECTION: convenience methods for constructing frequently used messages

    def send_obj(self, obj, location):
        """Send a torch object to a worker.

        Args:
            obj: A torch Tensor or Variable object to be sent.
            location: A BaseWorker instance indicating the worker which should
                receive the object.
        """

        return self.send_msg(MSGTYPE.OBJ, obj, location)

    def request_obj(self, obj_id, location):
        """Returns the requested object from specified location.

        Args:
            obj_id:  A string or integer id of an object to look up.
            location: A BaseWorker instance that lets you provide the lookup
                location.

        Returns:
            A torch Tensor or Variable object.
        """
        obj = self.send_msg(MSGTYPE.OBJ_REQ, obj_id, location)
        return obj

    # SECTION: Manage the workers network

    def get_worker(self, id_or_worker, fail_hard=False):
        """Returns the worker id or instance.

        Allows for resolution of worker ids to workers to happen automatically
        while also making the current worker aware of new ones when discovered
        through other processes.

        If you pass in an ID, it will try to find the worker object reference
        within self._known_workers. If you instead pass in a reference, it will
        save that as a known_worker if it does not exist as one.

        This method is useful because often tensors have to store only the ID
        to a foreign worker which may or may not be known by the worker that is
        de-serializing it at the time of deserialization.

        Args:
            id_or_worker: A string or integer id of the object to be returned
                or the BaseWorker object itself.
            fail_hard (bool): A boolean parameter indicating whether we want to
                throw an exception when a worker is not registered at this
                worker or we just want to log it.

        Returns:
            A string or integer id of the worker or the BaseWorker instance
            representing the worker.

        Example:
            >>> import syft as sy
            >>> hook = sy.TorchHook(verbose=False)
            >>> me = hook.local_worker
            >>> bob = sy.VirtualWorker(id="bob",hook=hook, is_client_worker=False)
            >>> me.add_worker([bob])
            >>> bob
            <syft.core.workers.virtual.VirtualWorker id:bob>
            >>> # we can get the worker using it's id (1)
            >>> me.get_worker('bob')
            <syft.core.workers.virtual.VirtualWorker id:bob>
            >>> # or we can get the worker by passing in the worker
            >>> me.get_worker(bob)
            <syft.core.workers.virtual.VirtualWorker id:bob>
        """

        if isinstance(id_or_worker, bytes):
            id_or_worker = str(id_or_worker, "utf-8")

        if isinstance(id_or_worker, (str, int)):
            if id_or_worker in self._known_workers:
                return self._known_workers[id_or_worker]
            else:
                if fail_hard:
                    raise WorkerNotFoundException
                logging.warning("Worker", self.id, "couldn't recognize worker", id_or_worker)
                return id_or_worker
        else:
            if id_or_worker.id not in self._known_workers:
                self.add_worker(id_or_worker)

        return id_or_worker

    def add_worker(self, worker):
        """Adds a single worker.

        Adds a worker to the list of _known_workers internal to the BaseWorker.
        Endows this class with the ability to communicate with the remote
        worker  being added, such as sending and receiving objects, commands,
        or  information about the network.

        Args:
            worker (:class:`BaseWorker`): A BaseWorker object representing the
                pointer to a remote worker, which must have a unique id.

        Example:
            >>> import torch
            >>> import syft as sy
            >>> hook = sy.TorchHook(verbose=False)
            >>> me = hook.local_worker
            >>> bob = sy.VirtualWorker(id="bob",hook=hook, is_client_worker=False)
            >>> me.add_worker([bob])
            >>> x = torch.Tensor([1,2,3,4,5])
            >>> x
            1
            2
            3
            4
            5
            [syft.core.frameworks.torch.tensor.FloatTensor of size 5]
            >>> x.send(bob)
            FloatTensor[_PointerTensor - id:9121428371 owner:0 loc:bob
                        id@loc:47416674672]
            >>> x.get()
            1
            2
            3
            4
            5
            [syft.core.frameworks.torch.tensor.FloatTensor of size 5]
        """
        if worker.id in self._known_workers:
            logging.warning(
                "Worker "
                + str(worker.id)
                + " already exists. Replacing old worker which could cause \
                    unexpected behavior"
            )
        self._known_workers[worker.id] = worker

    def add_workers(self, workers):
        """Adds several workers in a single call.

        Args:
            workers: A list of BaseWorker representing the workers to add.
        """
        for worker in workers:
            self.add_worker(worker)

    def __str__(self):
        """Returns the string representation of BaseWorker.

        A to-string method for all classes that extend BaseWorker.

        Returns:
            The Type and ID of the worker

        Example:
            A VirtualWorker instance with id 'bob' would return a string value of.
            >>> import syft as sy
            >>> bob = sy.VirtualWorker(id="bob")
            >>> bob
            <syft.workers.virtual.VirtualWorker id:bob>

        Note:
            __repr__ calls this method by default.
        """

        out = "<"
        out += str(type(self)).split("'")[1].split(".")[-1]
        out += " id:" + str(self.id)
        out += " #tensors:" + str(len(self._objects))
        out += ">"
        return out

    def __repr__(self):
        """Returns the official string representation of BaseWorker."""
        return self.__str__()

    def __getitem__(self, idx):
        return self._objects[idx]

    def clear_objects(self):
        """Removes all objects from the worker."""

        self._objects = {}
        return self

    @staticmethod
    def is_tensor_none(obj):
        return obj is None

    def request_is_remote_tensor_none(self, pointer):
        return self.send_msg(MSGTYPE.IS_NONE, pointer, location=pointer.location)

    def search(self, *query):
        """Search for a match between the query terms and the tensor's Id, Tag, or Description.
        Note that the query is an AND query meaning that every item in the list of strings (query*)
        must be found somewhere on the tensor in order for it to be included in the results.

        Args:
            query: a list of strings to match against.
            me: a reference to the worker calling the search.
            """
        results = list()
        for key, tensor in self._objects.items():
            found_something = True
            for query_item in query:
                match = False
                if query_item == str(key):
                    match = True

                if tensor.tags is not None:
                    if query_item in tensor.tags:
                        match = True

                if tensor.description is not None:
                    if query_item in tensor.description:
                        match = True

                if not match:
                    found_something = False

            if found_something:
                # set garbage_collect_data to False because if we're searching
                # for a tensor we don't own, then it's probably someone else's
                # decision to decide when to delete the tensor.
                ptr = tensor.create_pointer(
                    garbage_collect_data=False, owner=sy.local_worker
                ).wrap()
                results.append(ptr)

        return results
