from abc import abstractmethod
from contextlib import contextmanager

import logging
from typing import Callable
from typing import List
from typing import Tuple
from typing import Union
from typing import TYPE_CHECKING

import syft as sy
from syft import codes
from syft.generic.frameworks.hook import hook_args
from syft.generic.frameworks.remote import Remote
from syft.generic.frameworks.types import FrameworkTensorType
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.frameworks.types import FrameworkShape
from syft.generic.object_storage import ObjectStorage
from syft.generic.object import AbstractObject
from syft.generic.tensor import AbstractTensor
from syft.generic.pointers.object_pointer import ObjectPointer
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.messaging.message import Message
from syft.messaging.message import ForceObjectDeleteMessage
from syft.messaging.message import OperationMessage
from syft.messaging.message import ObjectMessage
from syft.messaging.message import ObjectRequestMessage
from syft.messaging.message import IsNoneMessage
from syft.messaging.message import GetShapeMessage
from syft.messaging.message import PlanCommandMessage
from syft.messaging.message import SearchMessage
from syft.messaging.message import ExecuteWorkerFunctionMessage
from syft.execution.plan import Plan
from syft.workers.abstract import AbstractWorker

from syft.exceptions import GetNotPermittedError
from syft.exceptions import WorkerNotFoundException
from syft.exceptions import ResponseSignatureError
from syft.exceptions import PlanCommandUnknownError


# this if statement avoids circular imports between base.py and pointer.py
if TYPE_CHECKING:
    from syft.generic.frameworks.hook.hook import FrameworkHook

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
            a pointer to a worker that exists elsewhere.
        log_msgs: An optional boolean parameter to indicate whether all
            messages should be saved into a log for later review. This is
            primarily a development/testing feature.
        auto_add: Determines whether to automatically add this worker to the
            list of known workers.
        message_pending_time (optional): A number of seconds to delay the messages to be sent.
            The argument may be a floating point number for subsecond 
            precision.
    """

    def __init__(
        self,
        hook: "FrameworkHook",
        id: Union[int, str] = 0,
        data: Union[List, tuple] = None,
        is_client_worker: bool = False,
        log_msgs: bool = False,
        verbose: bool = False,
        auto_add: bool = True,
        message_pending_time: Union[int, float] = 0,
    ):
        """Initializes a BaseWorker."""
        super().__init__()
        self.hook = hook

        self.id = id
        self.is_client_worker = is_client_worker
        self.log_msgs = log_msgs
        self.verbose = verbose
        self.auto_add = auto_add
        self._message_pending_time = message_pending_time
        self.msg_history = list()

        # For performance, we cache all possible message types
        self._message_router = {
            OperationMessage: self.execute_command,
            PlanCommandMessage: self.execute_plan_command,
            ObjectMessage: self.set_obj,
            ObjectRequestMessage: self.respond_to_obj_req,
            ForceObjectDeleteMessage: self.rm_obj,  # FIXME: there is no ObjectDeleteMessage
            IsNoneMessage: self.is_tensor_none,
            GetShapeMessage: self.get_tensor_shape,
            SearchMessage: self.search,
            ForceObjectDeleteMessage: self.force_rm_obj,
            ExecuteWorkerFunctionMessage: self.execute_worker_function,
        }

        self._plan_command_router = {
            codes.PLAN_CMDS.FETCH_PLAN: self._fetch_plan_remote,
            codes.PLAN_CMDS.FETCH_PROTOCOL: self._fetch_protocol_remote,
        }

        self.load_data(data)

        # Declare workers as appropriate
        self._known_workers = {}
        if auto_add:
            if hook is not None and hook.local_worker is not None:
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

        # Used to keep track of a building plan
        self.init_plan = None

        if hook is None:
            self.framework = None
        else:
            # TODO[jvmancuso]: avoid branching here if possible, maybe by changing code in
            #     execute_command or command_guard to not expect an attribute named "torch"
            #     (#2530)
            self.framework = hook.framework
            if hasattr(hook, "torch"):
                self.torch = self.framework
                self.remote = Remote(self, "torch")
            elif hasattr(hook, "tensorflow"):
                self.tensorflow = self.framework
                self.remote = Remote(self, "tensorflow")

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

    @contextmanager
    def registration_enabled(self):
        self.is_client_worker = False
        try:
            yield self
        finally:
            self.is_client_worker = True

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

    def load_data(self, data: List[Union[FrameworkTensorType, AbstractTensor]]) -> None:
        """Allows workers to be initialized with data when created

           The method registers the tensor individual tensor objects.

        Args:

            data: A list of tensors
        """

        if data:
            for tensor in data:
                self.register_obj(tensor)
                tensor.owner = self

    def send_msg(self, message: Message, location: "BaseWorker") -> object:
        """Implements the logic to send messages.

        The message is serialized and sent to the specified location. The
        response from the location (remote worker) is deserialized and
        returned back.

        Every message uses this method.

        Args:
            msg_type: A integer representing the message type.
            message: A Message object
            location: A BaseWorker instance that lets you provide the
                destination to send the message.

        Returns:
            The deserialized form of message from the worker at specified
            location.
        """
        if self.verbose:
            print(f"worker {self} sending {message} to {location}")

        # Step 1: serialize the message to a binary
        bin_message = sy.serde.serialize(message, worker=self)

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
        msg = sy.serde.deserialize(bin_message, worker=self)

        if self.verbose:
            print(f"worker {self} received {type(msg).__name__} {msg.contents}")

        # Step 1: route message to appropriate function
        response = self._message_router[type(msg)](msg.contents)

        # Step 2: Serialize the message to simple python objects
        bin_response = sy.serde.serialize(response, worker=self)

        return bin_response

        # SECTION:recv_msg() uses self._message_router to route to these methods

    def send(
        self,
        obj: Union[FrameworkTensorType, AbstractTensor],
        workers: "BaseWorker",
        ptr_id: Union[str, int] = None,
        garbage_collect_data=None,
        **kwargs,
    ) -> ObjectPointer:
        """Sends tensor to the worker(s).

        Send a syft or torch tensor/object and its child, sub-child, etc (all the
        syft chain of children) to a worker, or a list of workers, with a given
        remote storage address.

        Args:
            tensor: A syft/framework tensor/object to send.
            workers: A BaseWorker object representing the worker(s) that will
                receive the object.
            ptr_id: An optional string or integer indicating the remote id of
                the object on the remote worker(s).
            local_autograd: Use autograd system on the local machine instead of PyTorch's
                autograd on the workers.
            preinitialize_grad: Initialize gradient for AutogradTensors to a tensor
            garbage_collect_data: argument passed down to create_pointer()

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

        if not isinstance(workers, (list, tuple)):
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

        if hasattr(obj, "create_pointer") and not isinstance(
            obj, sy.Protocol
        ):  # TODO: this seems like hack to check a type
            if ptr_id is None:  # Define a remote id if not specified
                ptr_id = sy.ID_PROVIDER.pop()

            pointer = type(obj).create_pointer(
                obj,
                owner=self,
                location=worker,
                id_at_location=obj.id,
                register=True,
                ptr_id=ptr_id,
                garbage_collect_data=garbage_collect_data,
                **kwargs,
            )
        else:
            pointer = obj

        # Send the object
        self.send_obj(obj, worker)

        return pointer

    def execute_command(self, message: tuple) -> PointerTensor:
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
            if sy.framework.is_inplace_method(command_name):
                # TODO[jvmancuso]: figure out a good way to generalize the
                # above check (#2530)
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

            sy.framework.command_guard(command_name)

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
                response = hook_args.register_response(
                    command_name, response, list(return_ids), self
                )
                return response
            except ResponseSignatureError:
                return_id_provider = sy.ID_PROVIDER
                return_id_provider.set_next_ids(return_ids, check_ids=False)
                return_id_provider.start_recording_ids()
                response = hook_args.register_response(
                    command_name, response, return_id_provider, self
                )
                new_ids = return_id_provider.get_recorded_ids()
                raise ResponseSignatureError(new_ids)

    def execute_worker_function(self, message: tuple):
        """Executes commands received from other workers.

                Args:
                    message: A tuple specifying the command and the args.

                Returns:
                    A pointer to the result.
                """
        command_name, (args, kwargs, return_ids) = message

        command_name = command_name
        response = getattr(self, command_name)(*args, **kwargs)
        #  TODO [midokura-silvia]: send the tensor directly
        #  TODO this code is currently necessary for the async_fit method in websocket_client.py
        import torch

        if isinstance(response, torch.Tensor):
            self.register_obj(obj=response, obj_id=return_ids[0])
            return None
        return response

    def execute_plan_command(self, message: tuple):
        """Executes commands related to plans.

        This method is intended to execute all commands related to plans and
        avoiding having several new message types specific to plans.

        Args:
            message: A tuple specifying the command and args.
        """
        command_name, args = message
        try:
            command = self._plan_command_router[command_name]
        except KeyError:
            raise PlanCommandUnknownError(command_name)

        return command(*args)

    def send_command(
        self, recipient: "BaseWorker", message: tuple, return_ids: str = None
    ) -> Union[List[PointerTensor], PointerTensor]:
        """
        Sends a command through a message to a recipient worker.

        Args:
            recipient: A recipient worker.
            message: A tuple representing the message being sent.
            return_ids: A list of strings indicating the ids of the
                tensors that should be returned as response to the command execution.

        Returns:
            A list of PointerTensors or a single PointerTensor if just one response is expected.
        """
        if return_ids is None:
            return_ids = tuple([sy.ID_PROVIDER.pop()])

        cmd_name = message[0]
        cmd_owner = message[1]
        cmd_args = message[2]
        cmd_kwargs = message[3]

        try:
            ret_val = self.send_msg(
                OperationMessage(cmd_name, cmd_owner, cmd_args, cmd_kwargs, return_ids),
                location=recipient,
            )
        except ResponseSignatureError as e:
            ret_val = None
            return_ids = e.ids_generated

        if ret_val is None or type(ret_val) == bytes:
            responses = []
            for return_id in return_ids:
                response = PointerTensor(
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

    def respond_to_obj_req(self, request_msg: tuple):
        """Returns the deregistered object from registry.

        Args:
            request_msg (tuple): Tuple containing object id, user credentials and reason.
        """
        obj_id, user, reason = request_msg
        obj = self.get_obj(obj_id)
        if hasattr(obj, "allow") and not obj.allow(user):
            raise GetNotPermittedError()
        else:
            self.de_register_obj(obj)
            return obj

    def register_obj(self, obj: object, obj_id: Union[str, int] = None):
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
            super().register_obj(obj, obj_id=obj_id)

    def de_register_obj(self, obj: object, _recurse_torch_objs: bool = True):
        """
        De-registers the specified object with the current worker node.

        Args:
            obj: the object to deregister
            _recurse_torch_objs: A boolean indicating whether the object is
                more complex and needs to be explored.
        """
        if not self.is_client_worker:
            super().de_register_obj(obj, _recurse_torch_objs)

    # SECTION: convenience methods for constructing frequently used messages

    def send_obj(self, obj: object, location: "BaseWorker"):
        """Send a torch object to a worker.

        Args:
            obj: A torch Tensor or Variable object to be sent.
            location: A BaseWorker instance indicating the worker which should
                receive the object.
        """
        return self.send_msg(ObjectMessage(obj), location)

    def request_obj(
        self, obj_id: Union[str, int], location: "BaseWorker", user=None, reason: str = ""
    ) -> object:
        """Returns the requested object from specified location.

        Args:
            obj_id (int or string):  A string or integer id of an object to look up.
            location (BaseWorker): A BaseWorker instance that lets you provide the lookup
                location.
            user (object, optional): user credentials to perform user authentication.
            reason (string, optional): a description of why the data scientist wants to see it.
        Returns:
            A torch Tensor or Variable object.
        """
        obj = self.send_msg(ObjectRequestMessage((obj_id, user, reason)), location)
        return obj

    # SECTION: Manage the workers network

    def get_worker(
        self, id_or_worker: Union[str, int, "BaseWorker"], fail_hard: bool = False
    ) -> Union[str, int, AbstractWorker]:
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

        if isinstance(id_or_worker, str) or isinstance(id_or_worker, int):
            return self._get_worker_based_on_id(id_or_worker, fail_hard=fail_hard)
        else:
            return self._get_worker(id_or_worker)

    def _get_worker(self, worker: AbstractWorker):
        if worker.id not in self._known_workers:
            self.add_worker(worker)
        return worker

    def _get_worker_based_on_id(self, worker_id: Union[str, int], fail_hard: bool = False):
        # A worker should always know itself
        if worker_id == self.id:
            return self

        worker = self._known_workers.get(worker_id, worker_id)

        if worker == worker_id:
            if fail_hard:
                raise WorkerNotFoundException
            logger.warning("Worker %s couldn't recognize worker %s", self.id, worker_id)
        return worker

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
        return self._objects.get(idx, None)

    @staticmethod
    def is_tensor_none(obj):
        return obj is None

    def request_is_remote_tensor_none(self, pointer: PointerTensor):
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
        return self.send_msg(IsNoneMessage(pointer), location=pointer.location)

    @staticmethod
    def get_tensor_shape(tensor: FrameworkTensorType) -> List:
        """
        Returns the shape of a tensor casted into a list, to bypass the serialization of
        a torch.Size object.

        Args:
            tensor: A torch.Tensor.

        Returns:
            A list containing the tensor shape.
        """
        return list(tensor.shape)

    def request_remote_tensor_shape(self, pointer: PointerTensor) -> FrameworkShape:
        """
        Sends a request to the remote worker that holds the target a pointer to
        have its shape.

        Args:
            pointer: A pointer on which we want to get the shape.

        Returns:
            A torch.Size object for the shape.
        """
        shape = self.send_msg(GetShapeMessage(pointer), location=pointer.location)
        return sy.hook.create_shape(shape)

    def fetch_plan(
        self, plan_id: Union[str, int], location: "BaseWorker", copy: bool = False
    ) -> "Plan":  # noqa: F821
        """Fetchs a copy of a the plan with the given `plan_id` from the worker registry.

        This method is executed for local execution.

        Args:
            plan_id: A string indicating the plan id.

        Returns:
            A plan if a plan with the given `plan_id` exists. Returns None otherwise.
        """
        message = PlanCommandMessage("fetch_plan", (plan_id, copy))
        plan = self.send_msg(message, location=location)

        return plan

    def _fetch_plan_remote(self, plan_id: Union[str, int], copy: bool) -> "Plan":  # noqa: F821
        """Fetches a copy of a the plan with the given `plan_id` from the worker registry.

        This method is executed for remote execution.

        Args:
            plan_id: A string indicating the plan id.

        Returns:
            A plan if a plan with the given `plan_id` exists. Returns None otherwise.
        """
        if plan_id in self._objects:
            candidate = self._objects[plan_id]
            if isinstance(candidate, sy.Plan):
                if copy:
                    return candidate.copy()
                else:
                    return candidate

        return None

    def fetch_protocol(
        self, protocol_id: Union[str, int], location: "BaseWorker", copy: bool = False
    ) -> "Plan":  # noqa: F821
        """Fetch a copy of a the protocol with the given `protocol_id` from the worker registry.

        This method is executed for local execution.

        Args:
            protocol_id: A string indicating the protocol id.

        Returns:
            A protocol if a protocol with the given `protocol_id` exists. Returns None otherwise.
        """
        message = PlanCommandMessage("fetch_protocol", (protocol_id, copy))
        protocol = self.send_msg(message, location=location)

        return protocol

    def _fetch_protocol_remote(
        self, protocol_id: Union[str, int], copy: bool
    ) -> "Protocol":  # noqa: F821
        """
        Target function of fetch_protocol, find and return a protocol
        """
        if protocol_id in self._objects:

            candidate = self._objects[protocol_id]
            if isinstance(candidate, sy.Protocol):
                return candidate

        return None

    def search(self, query: Union[List[Union[str, int]], str, int]) -> List[PointerTensor]:
        """Search for a match between the query terms and a tensor's Id, Tag, or Description.

        Note that the query is an AND query meaning that every item in the list of strings (query*)
        must be found somewhere on the tensor in order for it to be included in the results.

        Args:
            query: A list of strings to match against.
            me: A reference to the worker calling the search.

        Returns:
            A list of PointerTensors.
        """
        if isinstance(query, (str, int)):
            query = [query]

        results = list()
        for key, obj in self._objects.items():
            found_something = True
            for query_item in query:
                # If deserialization produced a bytes object instead of a string,
                # make sure it's turned back to a string or a fair comparison.
                if isinstance(query_item, bytes):
                    query_item = query_item.decode("ascii")
                query_item = str(query_item)

                match = False
                if query_item == str(key):
                    match = True

                if isinstance(obj, (AbstractObject, FrameworkTensor)):
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

    def request_search(self, query: List[str], location: "BaseWorker"):

        result = self.send_msg(SearchMessage(query), location=location)

        return result

    def _get_msg(self, index):
        """Returns a decrypted message from msg_history. Mostly useful for testing.

        Args:
            index: the index of the message you'd like to receive.

        Returns:
            A decrypted messaging.Message object.

        """

        return sy.serde.deserialize(self.msg_history[index], worker=self)

    @property
    def message_pending_time(self):
        """
        Returns:
            The pending time in seconds for messaging between virtual workers.
        """
        return self._message_pending_time

    @message_pending_time.setter
    def message_pending_time(self, seconds: Union[int, float]) -> None:
        """Sets the pending time to send messaging between workers.

        Args:
            seconds: A number of seconds to delay the messages to be sent.
            The argument may be a floating point number for subsecond 
            precision.

        """
        if self.verbose:
            print(f"Set message pending time to {seconds} seconds.")

        self._message_pending_time = seconds

    @staticmethod
    def create_message_execute_command(command_name: str, return_ids=None, *args, **kwargs):
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
        return ExecuteWorkerFunctionMessage(command_name, (args, kwargs, return_ids))

    @property
    def serializer(self, workers=None) -> codes.TENSOR_SERIALIZATION:
        """
        Define the serialization strategy to adopt depending on the workers it's connected to.
        This is relevant in particular for Tensors which can be serialized in an efficient way
        between workers which share the same Deep Learning framework, but must be converted to
        lists or json-like objects in other cases.

        Args:
            workers: (Optional) the list of workers involved in the serialization. If not
                provided, self._known_workers is used.

        Returns:
            A str code:
                'all': serialization must be compatible with all kinds of workers
                'torch': serialization will only work between workers that support PyTorch
                (more to come: 'tensorflow', 'numpy', etc)
        """
        if workers is not None:
            if not isinstance(workers, list):
                workers = [workers]
            workers.append(self)
        else:
            # self is referenced in self.__known_workers
            # NOTE: for some reason self._known_workers may contain a Plan
            workers = [w for _, w in self._known_workers.items() if isinstance(w, AbstractWorker)]

        frameworks = set()
        for worker in workers:
            if worker.framework is not None:
                framework = worker.framework.__name__
            else:
                framework = "None"

            frameworks.add(framework)

        if len(frameworks) == 1 and frameworks == {"torch"}:
            return codes.TENSOR_SERIALIZATION.TORCH
        else:
            return codes.TENSOR_SERIALIZATION.ALL

    @staticmethod
    def simplify(_worker: AbstractWorker, worker: AbstractWorker) -> tuple:
        return (sy.serde.msgpack.serde._simplify(_worker, worker.id),)

    @staticmethod
    def detail(worker: AbstractWorker, worker_tuple: tuple) -> Union[AbstractWorker, int, str]:
        """
        This function reconstructs a PlanPointer given it's attributes in form of a tuple.

        Args:
            worker: the worker doing the deserialization
            plan_pointer_tuple: a tuple holding the attributes of the PlanPointer
        Returns:
            A worker id or worker instance.
        """
        worker_id = sy.serde.msgpack.serde._detail(worker, worker_tuple[0])

        referenced_worker = worker.get_worker(worker_id)

        return referenced_worker

    @staticmethod
    def force_simplify(_worker: AbstractWorker, worker: AbstractWorker) -> tuple:
        return (
            sy.serde.msgpack.serde._simplify(_worker, worker.id),
            sy.serde.msgpack.serde._simplify(_worker, worker._objects),
            worker.auto_add,
        )

    @staticmethod
    def force_detail(worker: AbstractWorker, worker_tuple: tuple) -> tuple:
        worker_id, _objects, auto_add = worker_tuple
        worker_id = sy.serde.msgpack.serde._detail(worker, worker_id)

        result = sy.VirtualWorker(sy.hook, worker_id, auto_add=auto_add)
        _objects = sy.serde.msgpack.serde._detail(worker, _objects)
        result._objects = _objects

        # make sure they weren't accidentally double registered
        for _, obj in _objects.items():
            if obj.id in worker._objects:
                del worker._objects[obj.id]

        return result
