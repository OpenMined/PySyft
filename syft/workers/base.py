import logging
import random
import sys

from abc import abstractmethod
import syft as sy

from syft.frameworks.torch.tensors.interpreters import AbstractTensor
from syft.frameworks.torch.tensors.interpreters import PointerTensor
from syft.exceptions import WorkerNotFoundException
from syft.exceptions import ResponseSignatureError
from syft.workers import AbstractWorker
from syft.workers import IdProvider
from syft.codes import MSGTYPE
from typing import Callable
from typing import List
from typing import Tuple
from typing import Union
import torch


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
        self, hook, id=0, data=None, is_client_worker=False, log_msgs=False, verbose=False
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

        # For performance, we cache each
        self._message_router = {
            MSGTYPE.CMD: self.execute_command,
            MSGTYPE.OBJ: self.set_obj,
            MSGTYPE.OBJ_REQ: self.respond_to_obj_req,
            MSGTYPE.OBJ_DEL: self.rm_obj,
            MSGTYPE.IS_NONE: self.is_tensor_none,
            MSGTYPE.GET_SHAPE: self.get_tensor_shape,
            MSGTYPE.SEARCH: self.deserialized_search,
        }

        self.load_data(data)

        # Declare workers as appropriate
        self._known_workers = {}
        if hook.local_worker is not None:
            known_workers = self.hook.local_worker._known_workers
            if self.id in known_workers:
                if isinstance(known_workers[self.id], type(self)):
                    # If a worker with this id already exists and it has the
                    # same type as the one being created, we copy all the attributes
                    # of the existing worker to this one.
                    self.__dict__.update(known_workers[self.id].__dict__)
                else:
                    raise RuntimeError("Worker initialized with the same id and different types.")
            else:
                hook.local_worker.add_worker(self)
                for worker_id, worker in hook.local_worker._known_workers.items():
                    if worker_id not in self._known_workers:
                        self.add_worker(worker)
                    if self.id not in worker._known_workers:
                        worker.add_worker(self)

    # SECTION: Methods which MUST be overridden by subclasses
    @abstractmethod
    def _send_msg(self, message: bin, location: "BaseWorker"):
        """Sends message from one worker to another.

        As BaseWorker implies, you should never instantiate this class by
        itself. Instead, you should extend BaseWorker in a new class which
        instantiates _send_msg and _recv_msg, each of which should specify the
        exact way in which two workers communicate with each other. The easiest
        example to study is VirtualWorker.

        Args:
            message: A binary message to be sent from one worker
                to another.
            location: A BaseWorker instance that lets you provide the
                destination to send the message.

        Raises:
            NotImplementedError: Method not implemented error.
        """

        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def _recv_msg(self, message: bin):
        """Receives the message.

        As BaseWorker implies, you should never instantiate this class by
        itself. Instead, you should extend BaseWorker in a new class which
        instantiates _send_msg and _recv_msg, each of which should specify the
        exact way in which two workers communicate with each other. The easiest
        example to study is VirtualWorker.

        Args:
            message: The binary message being received.

        Raises:
            NotImplementedError: Method not implemented error.

        """
        raise NotImplementedError  # pragma: no cover

    def load_data(self, data: List[Union[torch.Tensor, AbstractTensor]]) -> None:
        """Allows workers to be initialized with data when created

           The method registers the tensor individual tensor objects.

        Args:

            data: A list of tensors


        """

        if data:
            for tensor in data:
                self.register_obj(tensor)
                tensor.owner = self

    def send_msg(self, msg_type: int, message: str, location: "BaseWorker") -> object:

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
        bin_message = sy.serde.serialize(message)

        # Step 2: send the message and wait for a response
        bin_response = self._send_msg(bin_message, location)

        # Step 3: deserialize the response
        response = sy.serde.deserialize(bin_response, worker=self)

        return response

    def recv_msg(self, bin_message: bin) -> bin:
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
        (msg_type, contents) = sy.serde.deserialize(bin_message, worker=self)
        if self.verbose:
            print(f"worker {self} received {sy.codes.code2MSGTYPE[msg_type]} {contents}")
        # Step 1: route message to appropriate function
        response = self._message_router[msg_type](contents)

        # Step 2: Serialize the message to simple python objects
        bin_response = sy.serde.serialize(response)

        return bin_response

        # SECTION:recv_msg() uses self._message_router to route to these methods
        # Each method corresponds to a MsgType enum.

    def send(
        self,
        obj: Union[torch.Tensor, AbstractTensor],
        workers: "BaseWorker",
        ptr_id: Union[str, int] = None,
    ) -> PointerTensor:
        """Sends tensor to the worker(s).

        Send a syft or torch tensor/object and its child, sub-child, etc (all the
        syft chain of children) to a worker, or a list of workers, with a given
        remote storage address.

        Args:
            tensor: A syft/torch tensor/object object to send.
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

        if isinstance(obj, torch.Tensor):
            pointer = obj.create_pointer(
                owner=self, location=worker, id_at_location=obj.id, register=True, ptr_id=ptr_id
            )
        else:
            pointer = obj
        # Send the object
        self.send_obj(obj, worker)
        return pointer

    def execute_command(self, message):
        """
        Execute commands received from other workers
        :param message: the message specifying the command and the args
        :return: a pointer to the result
        """

        (command_name, _self, args, kwargs), return_ids = message

        # TODO add kwargs
        command_name = command_name.decode("utf-8")
        # Handle methods
        if _self is not None:
            if sy.torch.is_inplace_method(command_name):
                getattr(_self, command_name)(*args, **kwargs)
                return
            else:
                response = getattr(_self, command_name)(*args, **kwargs)
        # Handle functions
        else:
            # At this point, the command is ALWAYS a path to a
            # function (i.e., torch.nn.functional.relu). Thus,
            # we need to fetch this function and run it.

            sy.torch.command_guard(command_name, "torch_modules")

            paths = command_name.split(".")
            command = self
            for path in paths:
                command = getattr(command, path)

            response = command(*args, **kwargs)

        # some functions don't return anything (such as .backward())
        # so we need to check for that here.
        if response is not None:
            # Register response et create pointers for tensor elements
            try:
                response = sy.frameworks.torch.hook_args.register_response(
                    command_name, response, list(return_ids), self
                )
                return response
            except ResponseSignatureError:
                return_ids = IdProvider(return_ids)
                response = sy.frameworks.torch.hook_args.register_response(
                    command_name, response, return_ids, self
                )
                raise ResponseSignatureError(return_ids.generated)

    def send_command(self, recipient, message, return_ids=None):
        """
        Send a command through a message to a recipient worker
        :param recipient:
        :param message:
        :return:
        """
        if return_ids is None:
            return_ids = [int(10e10 * random.random())]

        message = (message, return_ids)

        try:
            _ = self.send_msg(MSGTYPE.CMD, message, location=recipient)
        except ResponseSignatureError as e:
            return_ids = e.ids_generated

        responses = []
        for return_id in return_ids:
            response = sy.PointerTensor(
                location=recipient,
                id_at_location=return_id,
                owner=self,
                id=int(10e10 * random.random()),
            )
            responses.append(response)

        if len(return_ids) == 1:
            return responses[0]

        return responses

    def set_obj(self, obj: Union[torch.Tensor, AbstractTensor]) -> None:

        """Adds an object to the registry of objects.

        Args:
            obj: A torch or syft tensor with an id
        """
        self._objects[obj.id] = obj

    def get_obj(self, obj_id: Union[str, int]) -> object:
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
        if hasattr(obj, "child"):
            if isinstance(obj.child, PointerTensor):
                obj.child.garbage_collect_data = False
            if isinstance(obj.child, (sy.AdditiveSharingTensor, sy.MultiPointerTensor)):
                shares = obj.child.child
                for worker, share in shares.items():
                    share.child.garbage_collect_data = False

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
        """
        Send a request to the remote worker that holds the target a pointer if
        the value of the remote tensor is None or not.
        Note that the pointer must be valid: if there is no target (which is
        different from having a target equal to None), it will return an error.

        Args:
            :param pointer: the pointer on which we can to get information

        :return: a boolean stating if the remote value is None
        """
        return self.send_msg(MSGTYPE.IS_NONE, pointer, location=pointer.location)

    @staticmethod
    def get_tensor_shape(obj):
        """
        Return the shape of a tensor casted into a list, to bypass the serialization of
        a torch.Size object.
        :param obj: torch.Tensor
        :return: a list containing the tensor shape
        """
        return list(obj.shape)

    def request_remote_tensor_shape(self, pointer):
        """
        Send a request to the remote worker that holds the target a pointer to
        have its shape.

        Args:
            :param pointer: the pointer on which we can to get the shape

        :return: a torch.Size object for the shape
        """
        shape = self.send_msg(MSGTYPE.GET_SHAPE, pointer, location=pointer.location)
        return sy.hook.torch.Size(shape)

    def search(self, *query):
        """Search for a match between the query terms and the tensor's Id, Tag, or Description.
        Note that the query is an AND query meaning that every item in the list of strings (query*)
        must be found somewhere on the tensor in order for it to be included in the results.

        Args:
            query: a list of strings to match against.
            me: a reference to the worker calling the search.
            """
        results = list()
        for key, obj in self._objects.items():
            found_something = True
            for query_item in query:
                # If deserialization produced a bytes object instead of a string,
                # make sure it's turned back to a string or a fair comparison.
                if isinstance(query_item, bytes):
                    query_item = query_item.decode("ascii")

                match = False
                if query_item == str(key):
                    match = True

                if obj.tags is not None:
                    if query_item in obj.tags:
                        match = True

                if obj.description is not None:
                    if query_item in obj.description:
                        match = True

                if not match:
                    found_something = False

            if found_something:
                if isinstance(obj, torch.Tensor):
                    # set garbage_collect_data to False because if we're searching
                    # for a tensor we don't own, then it's probably someone else's
                    # decision to decide when to delete the tensor.
                    ptr = obj.create_pointer(
                        garbage_collect_data=False, owner=sy.local_worker
                    ).wrap()
                    results.append(ptr)
                else:
                    obj.owner = sy.local_worker
                    results.append(obj)

        return results

    def deserialized_search(self, query_items: Tuple[str]) -> List[PointerTensor]:
        """Called when a message requesting a call to `search` is received.
        The serialized arguments will arrive as a `tuple` and it needs to be
        transformed to an arguments list.

        Args:
            query_items(tuple(str)): Tuple of items to search for. Should originate from the
            deserialization of a message requesting a search operation.

        Returns:
            list(PointerTensor): List of matched tensors.
        """
        return self.search(*query_items)

    def generate_triple(
        self, cmd: Callable, field: int, a_size: tuple, b_size: tuple, locations: list
    ):
        """Generates a multiplication triple and sends it to all locations

        Args:
            cmd: equation in einsum notation
            field: integer representing the field size
            a_size: tuple which is the size that a should be
            b_size: tuple which is the size that b should be
            locations: a list of workers where the triple should be shared between

        return:
            a triple of AdditiveSharedTensors such that c_shared = cmd(a_shared, b_shared)
        """
        a = self.torch.randint(field, a_size)
        b = self.torch.randint(field, b_size)
        c = cmd(a, b)
        a_shared = a.share(*locations, field=field, crypto_provider=self).child
        b_shared = b.share(*locations, field=field, crypto_provider=self).child
        c_shared = c.share(*locations, field=field, crypto_provider=self).child
        return a_shared, b_shared, c_shared
