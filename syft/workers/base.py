from abc import ABC, abstractmethod
import logging
import syft

from .. import serde


MSGTYPE_CMD = 1
MSGTYPE_OBJ = 2
MSGTYPE_OBJ_REQ = 3
MSGTYPE_EXCEPTION = 4


class BaseWorker(ABC):
    """
    This is the class which contains functionality generic to all workers. Other workers will
    extend this class to inherit all functionality necessary for PySyft's protocol. Extensions
    of this class will override two key methods _send_msg() and _recv_msg() which are responsible
    for defining the procedure for sending a binary message to another worker.

    At it's core, you can think of BaseWorker (and thus all workers) as simply a collection of
    objects owned by a certain machine. Each worker defines how it interacts with objects on other
    workers as well as how other workers interact with objects owned by itself. Objects are most
    frequently tensors but they can be of any type supported by the PySyft protocol.
    """

    def __init__(self, hook=None, id=0, known_workers={}, is_client_worker=False):

        # if hook is None and hasattr(syft, "local_worker"):
        #    hook = syft.local_worker.hook

        # This is a reference to the hook object which overloaded
        # the underlying deep learning framework
        self.hook = hook

        # the integer or string identifier for this node
        self.id = id

        # is_client_worker determines whether this worker is
        # associated with an end user client. If so, it assumes
        # that the client will maintain control over when variables
        # are instantiated or deleted as opposed to
        # handling tensor/variable/model lifecycle internally.
        self.is_client_worker = is_client_worker

        # This is the core object in every BaseWorker instantiation, a collection of
        # objects. All objects are stored using their IDs as keys.
        self._objects = {}

        # This dictionary includes all known workers on a network. Extensions of
        # BaseWorker will include advanced functionality for adding to this dictionary
        # (node discovery). In some cases, one can initialize this with known workers to
        # help bootstrap the network.
        self._known_workers = {}
        for k, v in known_workers.items():
            self._known_workers[k] = v
        self.add_worker(self)

        # if hasattr(sy, "local_worker"):
        #     sy.local_worker.add_worker(self)
        #
        # self.add_worker(sy.local_worker)

        # For performance, we cache each
        self._message_router = {MSGTYPE_OBJ: self.set_obj, MSGTYPE_OBJ_REQ: self.get_obj}

    # SECTION: Methods which MUST be overridden by subclasses

    @abstractmethod
    def _send_msg(self, message, location):
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def _recv_msg(self, message):
        raise NotImplementedError  # pragma: no cover

    # SECTION: Generic Message Sending/Receiving Logic
    # Every message uses these methods.

    def send_msg(self, msg_type, message, location):
        # Step 0: combine type and message
        message = (msg_type, message)

        # Step 1: serialize the message to simple python objects
        bin_message = serde.serialize(message)

        # Step 2: send the message and wait for a response
        bin_response = self._send_msg(bin_message, location)

        # Step 3: deserialize the response
        response = serde.deserialize(bin_response)

        return response

    def recv_msg(self, bin_message):

        # Step 0: deserialize message
        (msg_type, contents) = serde.deserialize(bin_message)

        # Step 1: route message to appropriate function
        response = self._message_router[msg_type](contents)

        # Step 2: If response is none, set default
        if response is None:
            response = 0

        # Step 3: Serialize the message to simple python objects
        bin_response = serde.serialize(response)

        return bin_response

    # SECTION: recv_msg() uses self._message_router to route to these methods
    # Each method corresponds to a MsgType enum.

    def set_obj(self, obj):
        self._objects[obj["id"]] = obj

    def get_obj(self, obj_id):
        return self._objects[obj_id]

    # SECTION: convenience methods for constructing frequently used messages

    def send_obj(self, obj, location):
        return self.send_msg(MSGTYPE_OBJ, obj, location)

    def request_obj(self, obj_id, location):
        return self.send_msg(MSGTYPE_OBJ_REQ, obj_id, location)

    # SECTION: Execute operations

    def _execute_call(self, attr, self_, *args, **kwargs):
        """
        Receive, analyse and optionally forwar a call to perform a command
        """
        has_self = self_ is not None

        # if has_self:
        #     command = torch._command_guard(attr, "tensorvar_methods")
        # else:
        #     command = torch._command_guard(attr, "torch_modules")

        if has_self:
            native_attr = syft.torch.command_guard(attr, "tensorvar_methods", get_native=True)
            command = getattr(self_, native_attr)
        else:
            native_func = syft.torch.command_guard(attr, "torch_modules", get_native=True)
            command = native_func
        # print("command", command, type(command))
        response = command(*args, **kwargs)

        return response

    # SECTION: Manage the workers network

    def get_worker(self, id_or_worker):
        """get_worker(self, id_or_worker) -> BaseWorker
        If you pass in an ID, it will try to find the worker object reference
        within self._known_workers. If you instead pass in a reference, it will
        save that as a known_worker if it does not exist as one. This method is
        primarily useful because often tensors have to store only the ID to a
        foreign worker which may or may not be known by the worker that is
        deserializing it at the time of deserialization. This method allows for
        resolution of worker ids to workers to happen automatically while also
        making the current worker aware of new ones when discovered through
        other processes.
        :Parameters:
        * **id_or_worker (string or int or** :class:`BaseWorker` **)**
          This is either the id of the object to be returned or the object itself.
        :Example:
        >>> import syft as sy
        >>> hook = sy.TorchHook(verbose=False)
        >>> me = hook.local_worker
        >>> bob = sy.VirtualWorker(id="bob",hook=hook, is_client_worker=False)
        >>> me.add_workers([bob])
        >>> bob
        <syft.core.workers.virtual.VirtualWorker id:bob>
        >>> # we can get the worker using it's id (1)
        >>> me.get_worker('bob')
        <syft.core.workers.virtual.VirtualWorker id:bob>
        >>> # or we can get the worker by passing in the worker
        >>> me.get_worker(bob)
        <syft.core.workers.virtual.VirtualWorker id:bob>
        """
        if isinstance(id_or_worker, (str, int)):
            if id_or_worker in self._known_workers:
                return self._known_workers[id_or_worker]
            else:
                logging.warning("Worker", self.id, "couldnt recognize worker", id_or_worker)
                return id_or_worker
        else:
            if id_or_worker.id not in self._known_workers:
                self.add_worker(id_or_worker)

        return id_or_worker

    def add_worker(self, worker):
        """add_worker(worker) -> None This method adds a worker to the list of
        _known_workers internal to the BaseWorker. It endows this class with
        the ability to communicate with the remote worker being added, such as
        sending and receiving objects, commands, or information about the
        network.
        :Parameters:
        * **worker (**:class:`BaseWorker` **)** This is an object
          pointer to a remote worker, which must have a unique id.
        :Example:
        >>> import syft as sy
        >>> hook = sy.TorchHook(verbose=False)
        >>> me = hook.local_worker
        >>> bob = sy.VirtualWorker(id="bob",hook=hook, is_client_worker=False)
        >>> me.add_workers([bob])
        >>> x = sy.Tensor([1,2,3,4,5])
        >>> x
         1
         2
         3
         4
         5
        [syft.core.frameworks.torch.tensor.FloatTensor of size 5]
        >>> x.send(bob)
        FloatTensor[_PointerTensor - id:9121428371 owner:0 loc:bob id@loc:47416674672]
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
                + " already exists. Replacing old worker which could cause unexpected behavior"
            )

        # Add worker to the list of known workers
        # it's just a mapping from ID->object
        self._known_workers[worker.id] = worker

    def add_workers(self, workers):
        """
        Convenient function to add several workers in a single call
        :param workers: list of workers
        """
        for worker in workers:
            self.add_worker(worker)
