import logging

from abc import abstractmethod
import syft as sy

from syft.frameworks.torch.tensors.interpreters import AbstractTensor
from syft.generic import ObjectStorage
from syft.exceptions import GetNotPermittedError
from syft.exceptions import WorkerNotFoundException
from syft.exceptions import ResponseSignatureError
from syft.workers import AbstractWorker
from syft import codes
from typing import Callable
from typing import List
from typing import Tuple
from typing import Union
from typing import TYPE_CHECKING
import torch

# this if statement avoids circular imports between base.py and pointer.py
if TYPE_CHECKING:
    from syft.frameworks.torch import pointers

logger = logging.getLogger(__name__)


class BaseWorker(AbstractWorker, ObjectStorage):
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
        auto_add: Determines whether to automatically add this worker to the
            list of known workers.
    """

    def __init__(
        self,
        hook: "sy.TorchHook",
        id: Union[int, str] = 0,
        data: Union[List, tuple] = None,
        is_client_worker: bool = False,
        log_msgs: bool = False,
        verbose: bool = False,
        auto_add: bool = True,
    ):
        """Initializes a BaseWorker."""
        super().__init__()
        self.hook = hook
        self.torch = None if hook is None else hook.torch
        self.id = id
        self.is_client_worker = is_client_worker
        self.log_msgs = log_msgs
        self.verbose = verbose
        self.auto_add = auto_add
        self.msg_history = list()

        # For performance, we cache each
        self._message_router = {
            codes.MSGTYPE.CMD: self.execute_command,
            codes.MSGTYPE.OBJ: self.set_obj,
            codes.MSGTYPE.OBJ_REQ: self.respond_to_obj_req,
            codes.MSGTYPE.OBJ_DEL: self.rm_obj,
            codes.MSGTYPE.IS_NONE: self.is_tensor_none,
            codes.MSGTYPE.GET_SHAPE: self.get_tensor_shape,
            codes.MSGTYPE.SEARCH: self.deserialized_search,
            codes.MSGTYPE.FORCE_OBJ_DEL: self.force_rm_obj,
        }

        self.load_data(data)

        # Declare workers as appropriate
        self._known_workers = {}
        if auto_add:
            if hook.local_worker is not None:
                known_workers = self.hook.local_worker._known_workers
                if self.id in known_workers:
                    if isinstance(known_workers[self.id], type(self)):
                        # If a worker with this id already exists and it has the
                        # same type as the one being created, we copy all the attributes
                        # of the existing worker to this one.
                        self.__dict__.update(known_workers[self.id].__dict__)
                    else:
                        raise RuntimeError(
                            "Worker initialized with the same id and different types."
                        )
                else:
                    hook.local_worker.add_worker(self)
                    for worker_id, worker in hook.local_worker._known_workers.items():
                        if worker_id not in self._known_workers:
                            self.add_worker(worker)
                        if self.id not in worker._known_workers:
                            worker.add_worker(self)
            else:
                # Make the local worker aware of itself
                # self is the to-be-created local worker
                self.add_worker(self)

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

    def remove_worker_from_registry(self, worker_id):
        """Removes a worker from the dictionary of known workers.
        Args:
            worker_id: id to be removed
        """
        del self._known_workers[worker_id]

    def remove_worker_from_local_worker_registry(self):
        """Removes itself from the registry of hook.local_worker.
        """
        self.hook.local_worker.remove_worker_from_registry(worker_id=self.id)

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
        local_autograd=False,
        preinitialize_grad=False,
    ) -> "pointers.ObjectPointer":
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
            local_autograd: Use autograd system on the local machine instead of PyTorch's
                autograd on the workers.
            preinitialize_grad: Initialize gradient for AutogradTensors to a tensor

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

        if hasattr(obj, "create_pointer"):
            if ptr_id is None:  # Define a remote id if not specified
                ptr_id = sy.ID_PROVIDER.pop()

            pointer = type(obj).create_pointer(
                obj,
                owner=self,
                location=worker,
                id_at_location=obj.id,
                register=True,
                ptr_id=ptr_id,
                local_autograd=local_autograd,
                preinitialize_grad=preinitialize_grad,
            )
        else:
            pointer = obj
        # Send the object
        self.send_obj(obj, worker)

        return pointer

    def execute_command(self, message: tuple) -> "pointers.PointerTensor":
        """
        Executes commands received from other workers.

        Args:
            message: A tuple specifying the command and the args.

        Returns:
            A pointer to the result.
        """

        (command_name, _self, args, kwargs), return_ids = message

        # TODO add kwargs
        command_name = command_name
        # Handle methods
        if _self is not None:
            if type(_self) == int:
                _self = BaseWorker.get_obj(self, _self)
                if _self is None:
                    return
            if type(_self) == str and _self == "self":
                _self = self
            if sy.torch.is_inplace_method(command_name):
                getattr(_self, command_name)(*args, **kwargs)
                return
            else:
                try:
                    response = getattr(_self, command_name)(*args, **kwargs)
                except TypeError:
                    # TODO Andrew thinks this is gross, please fix. Instead need to properly deserialize strings
                    new_args = [
                        arg.decode("utf-8") if isinstance(arg, bytes) else arg for arg in args
                    ]
                    response = getattr(_self, command_name)(*new_args, **kwargs)
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
            # Register response and create pointers for tensor elements
            try:
                response = sy.frameworks.torch.hook_args.register_response(
                    command_name, response, list(return_ids), self
                )
                return response
            except ResponseSignatureError:
                return_id_provider = sy.ID_PROVIDER
                return_id_provider.set_next_ids(return_ids, check_ids=False)
                return_id_provider.start_recording_ids()
                response = sy.frameworks.torch.hook_args.register_response(
                    command_name, response, return_id_provider, self
                )
                new_ids = return_id_provider.get_recorded_ids()
                raise ResponseSignatureError(new_ids)

    def send_command(
        self, recipient: "BaseWorker", message: str, return_ids: str = None
    ) -> Union[List["pointers.PointerTensor"], "pointers.PointerTensor"]:
        """
        Sends a command through a message to a recipient worker.

        Args:
            recipient: A recipient worker.
            message: A string representing the message being sent.
            return_ids: A list of strings indicating the ids of the
                tensors that should be returned as response to the command execution.

        Returns:
            A list of PointerTensors or a single PointerTensor if just one response is expected.
        """
        if return_ids is None:
            return_ids = tuple([sy.ID_PROVIDER.pop()])

        message = (message, return_ids)

        try:
            ret_val = self.send_msg(codes.MSGTYPE.CMD, message, location=recipient)
        except ResponseSignatureError as e:
            ret_val = None
            return_ids = e.ids_generated

        if ret_val is None or type(ret_val) == bytes:
            responses = []
            for return_id in return_ids:
                response = sy.PointerTensor(
                    location=recipient,
                    id_at_location=return_id,
                    owner=self,
                    id=sy.ID_PROVIDER.pop(),
                )
                responses.append(response)

            if len(return_ids) == 1:
                responses = responses[0]
        else:
            responses = ret_val
        return responses

    def get_obj(self, obj_id: Union[str, int]) -> object:
        """Returns the object from registry.

        Look up an object from the registry using its ID.

        Args:
            obj_id: A string or integer id of an object to look up.
        """
        obj = super().get_obj(obj_id)
        # An object called with get_obj will be "with high probability" serialized
        # and sent back, so it will be GCed but remote data is any shouldn't be
        # deleted
        if hasattr(obj, "child") and hasattr(obj.child, "set_garbage_collect_data"):
            obj.child.set_garbage_collect_data(value=False)

        if hasattr(obj, "private") and obj.private:
            return None

        return obj

    def respond_to_obj_req(self, obj_id: Union[str, int]):
        """Returns the deregistered object from registry.

        Args:
            obj_id: A string or integer id of an object to look up.
        """

        obj = self.get_obj(obj_id)
        if hasattr(obj, "allowed_to_get") and not obj.allowed_to_get():
            raise GetNotPermittedError()
        else:
            self.de_register_obj(obj)
            return obj

    def register_obj(self, obj: object, obj_id: Union[str, int] = None):
        """Registers the specified object with the current worker node.

        Selects an id for the object, assigns a list of owners, and establishes
        whether it's a pointer or not. This method is generally not used by the
        whether it's a pointer or not. This method is generally not used by the
        client and is instead used by internal processes (hooks and workers).

        Args:
            obj: A torch Tensor or Variable object to be registered.
            obj_id (int or string): random integer between 0 and 1e10 or
            string uniquely identifying the object.
        """
        if not self.is_client_worker:
            super().register_obj(obj, obj_id=obj_id)

    # SECTION: convenience methods for constructing frequently used messages

    def send_obj(self, obj: object, location: "BaseWorker"):
        """Send a torch object to a worker.

        Args:
            obj: A torch Tensor or Variable object to be sent.
            location: A BaseWorker instance indicating the worker which should
                receive the object.
        """
        return self.send_msg(codes.MSGTYPE.OBJ, obj, location)

    def request_obj(self, obj_id: Union[str, int], location: "BaseWorker") -> object:
        """Returns the requested object from specified location.

        Args:
            obj_id:  A string or integer id of an object to look up.
            location: A BaseWorker instance that lets you provide the lookup
                location.

        Returns:
            A torch Tensor or Variable object.
        """
        obj = self.send_msg(codes.MSGTYPE.OBJ_REQ, obj_id, location)
        return obj

    # SECTION: Manage the workers network

    def get_worker(
        self, id_or_worker: Union[str, int, "BaseWorker"], fail_hard: bool = False
    ) -> Union[str, int]:
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
                logger.warning("Worker %s couldn't recognize worker %s", self.id, id_or_worker)
                return id_or_worker
        else:
            if id_or_worker.id not in self._known_workers:
                self.add_worker(id_or_worker)

        return id_or_worker

    def add_worker(self, worker: "BaseWorker"):
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
            logger.warning(
                "Worker "
                + str(worker.id)
                + " already exists. Replacing old worker which could cause \
                    unexpected behavior"
            )
        self._known_workers[worker.id] = worker

        return self

    def add_workers(self, workers: List["BaseWorker"]):
        """Adds several workers in a single call.

        Args:
            workers: A list of BaseWorker representing the workers to add.
        """
        for worker in workers:
            self.add_worker(worker)

        return self

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
        out += " #objects:" + str(len(self._objects))
        out += ">"
        return out

    def __repr__(self):
        """Returns the official string representation of BaseWorker."""
        return self.__str__()

    def __getitem__(self, idx):
        return self._objects[idx]

    def clear_objects(self):
        """Removes all objects from the worker."""

        self._objects.clear()
        return self

    @staticmethod
    def is_tensor_none(obj):
        return obj is None

    def request_is_remote_tensor_none(self, pointer: "pointers.PointerTensor"):
        """
        Sends a request to the remote worker that holds the target a pointer if
        the value of the remote tensor is None or not.
        Note that the pointer must be valid: if there is no target (which is
        different from having a target equal to None), it will return an error.

        Args:
            pointer: The pointer on which we can to get information.

        Returns:
            A boolean stating if the remote value is None.
        """
        return self.send_msg(codes.MSGTYPE.IS_NONE, pointer, location=pointer.location)

    @staticmethod
    def get_tensor_shape(tensor: torch.Tensor) -> List:
        """
        Returns the shape of a tensor casted into a list, to bypass the serialization of
        a torch.Size object.

        Args:
            tensor: A torch.Tensor.

        Returns:
            A list containing the tensor shape.
        """
        return list(tensor.shape)

    def request_remote_tensor_shape(
        self, pointer: "pointers.PointerTensor"
    ) -> "sy.hook.torch.Size":
        """
        Sends a request to the remote worker that holds the target a pointer to
        have its shape.

        Args:
            pointer: A pointer on which we want to get the shape.

        Returns:
            A torch.Size object for the shape.
        """
        shape = self.send_msg(codes.MSGTYPE.GET_SHAPE, pointer, location=pointer.location)
        return sy.hook.torch.Size(shape)

    def fetch_plan(self, plan_id: Union[str, int]) -> "Plan":  # noqa: F821
        """Fetchs a copy of a the plan with the given `plan_id` from the worker registry.

        Args:
            plan_id: A string indicating the plan id.

        Returns:
            A plan if a plan with the given `plan_id` exists. Returns None otherwise.
        """
        if plan_id in self._objects:
            candidate = self._objects[plan_id]
            if isinstance(candidate, sy.Plan):
                plan = candidate.copy()
                plan.owner = sy.local_worker
                return plan

        return None

    def search(self, *query: List[str]) -> List["pointers.PointerTensor"]:
        """Search for a match between the query terms and a tensor's Id, Tag, or Description.

        Note that the query is an AND query meaning that every item in the list of strings (query*)
        must be found somewhere on the tensor in order for it to be included in the results.

        Args:
            query: A list of strings to match against.
            me: A reference to the worker calling the search.

        Returns:
            A list of PointerTensors.
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

                if isinstance(obj, torch.Tensor):
                    if obj.tags is not None:
                        if query_item in obj.tags:
                            match = True

                    if obj.description is not None:
                        if query_item in obj.description:
                            match = True

                if not match:
                    found_something = False

            if found_something:
                # set garbage_collect_data to False because if we're searching
                # for a tensor we don't own, then it's probably someone else's
                # decision to decide when to delete the tensor.
                ptr = obj.create_pointer(garbage_collect_data=False, owner=sy.local_worker).wrap()
                results.append(ptr)

        return results

    def deserialized_search(self, query_items: Tuple[str]) -> List["pointers.PointerTensor"]:
        """
        Called when a message requesting a call to `search` is received.
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
        """Generates a multiplication triple and sends it to all locations.

        Args:
            cmd: An equation in einsum notation.
            field: An integer representing the field size.
            a_size: A tuple which is the size that a should be.
            b_size: A tuple which is the size that b should be.
            locations: A list of workers where the triple should be shared between.

        Returns:
            A triple of AdditiveSharedTensors such that c_shared = cmd(a_shared, b_shared).
        """
        a = self.torch.randint(field, a_size)
        b = self.torch.randint(field, b_size)
        c = cmd(a, b)
        a_shared = a.share(*locations, field=field, crypto_provider=self).child
        b_shared = b.share(*locations, field=field, crypto_provider=self).child
        c_shared = c.share(*locations, field=field, crypto_provider=self).child
        return a_shared, b_shared, c_shared

    @staticmethod
    def create_message_execute_command(
        command_name: codes.MSGTYPE, command_owner=None, return_ids=None, *args, **kwargs
    ):
        """helper function creating a message tuple for the execute_command call

        Args:
            command_name: name of the command that shall be called
            command_owner: owner of the function (None for torch functions, "self" for classes derived from
                           workers.base or ptr_id for remote objects
            return_ids: optionally set the ids of the return values (for remote objects)
            *args:  will be passed to the call of command_name
            **kwargs:  will be passed to the call of command_name

        Returns:
            tuple: (command_name, command_owner, args, kwargs), return_ids

        """
        if return_ids is None:
            return_ids = []
        return tuple([codes.MSGTYPE.CMD, [[command_name, command_owner, args, kwargs], return_ids]])
