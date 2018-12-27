import logging
import random

from abc import abstractmethod
from syft.util import WorkerNotFoundException
from syft import serde
from syft.workers import AbstractWorker

MSGTYPE_CMD = 1
MSGTYPE_OBJ = 2
MSGTYPE_OBJ_REQ = 3
MSGTYPE_EXCEPTION = 4


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
    """

    def __init__(self, hook=None, id=0, known_workers={}, is_client_worker=False):
        """Constructor of generic worker class
        
        Extensions of BaseWorker will include advanced functionality for adding to known_workers dictionary (node discovery)

        Args:
            hook (TorchHook, optional): An instance of :class:`TorchHook` class. A reference to the hook object which
                was used to modify PyTorch with PySyft's functionality. It overload the underlying deep learning framework
            id ((int or str), optional): the unique id of the worker (node).
            known_workers (dict, optional): This dictionary includes all known workers on a network. 
                The key of each element should be each worker's unique ID and the value should be a worker class which extends BaseWorker (this class).
                It can be initialized with known workers to help bootstrap the network.
            is_client_worker (bool, optional): If true, this object is not actually
                where the objects will be stored, but it is instead a pointer to a worker that exists
                elsewhere. This client mantains control over variables created by itself.
        """

        # if hook is None and hasattr(syft, "local_worker"):
        #    hook = syft.local_worker.hook

        self.hook = hook
        self.id = id
        self.is_client_worker = is_client_worker
        # A core object in every BaseWorker instantiation. A Collection of
        # objects where all objects are stored using their IDs as keys.
        self._objects = {}

        # This dictionary includes all known workers on a network.
        self._known_workers = {}
        for k, v in known_workers.items():
            self._known_workers[k] = v
        self.add_worker(self)
        # For performance, we cache each
        self._message_router = {MSGTYPE_OBJ: self.set_obj, MSGTYPE_OBJ_REQ: self.respond_to_obj_req}

    # SECTION: Methods which MUST be overridden by subclasses
    @abstractmethod
    def _send_msg(self, message, location):
        """Abstract method to send message.

        It specifies the exact way in which two workers communicate with each
        other. The easiest example to study is probably VirtualWorker.

        Args:
            message (str): the message being sent from one worker to another.
            location (BaseWorker): An instance of :class:`.workers.BaseWorker` class. It is the destination to send the
                message.
        Raises:
            NotImplementedError: An error occurred when instantiating a child of BaseWorker without specify _send_msg method.
        """

        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def _recv_msg(self, message):
        """Abstract method to receive message.

        It specifies the exact way in which two workers communicate with each
        other. The easiest example to study is probably VirtualWorker.

        Args:
            message (str): the message being received.
        Raises:
            NotImplementedError: An error occurred when instantiating a child of BaseWorker without specify _recv_msg method.
        """

        raise NotImplementedError  # pragma: no cover

    def send_msg(self, msg_type, message, location):
        """Method called to send the message.

        Args:
            msg_type (str): The type of message
            message (Tensor): the syft or torch tensor to send
            location (BaseWorker): An instance of :class:`.workers.BaseWorker` class. It is the destination to send the message.
        Returns:
            The response deserialized
        """

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
        """Method called to receive the message.

        Args:
            bin_message (bin): The object received in binary
        Returns:
            A simpley python object 
        """

        # Step 0: deserialize message
        (msg_type, contents) = serde.deserialize(bin_message)

        # Step 1: route message to appropriate function
        response = self._message_router[msg_type](contents)

        # Step 2: If response in none, set default
        if response is None:
            response = 0

        # Step 3: Serialize the message to simple python objects
        bin_response = serde.serialize(response)

        return bin_response

        # SECTION:recv_msg() uses self._message_router to route to these methods
        # Each method corresponds to a MsgType enum.

    def send(self, tensor, workers, ptr_id=None):
        """Send a syft or torch tensor and his child, sub-child, etc (ie all the
        syft chain of children) to a worker, or a list of workers, with a given
        remote storage address.

        Example:
            >>> x.send(bob, 1000)
            >>> #will result in bob having the tensor x with id 1000

        Args:
            tensor (Tensor): the syft or torch tensor to send
            workers (BaseWorker): Instances of :class:`workers.BaseWorker` class. The worker or list of Workers
                which will receive the object
            ptr_id ((str or int), optional): The remote id of the object on the remote worker(s).
        Returns:
            The pointer object that points to the tensor sent
        """
        if not isinstance(workers, list):
            workers = [workers]

        assert len(workers) > 0, "Please provide workers to receive the data"

        if len(workers) == 1:
            worker = workers[0]
        else: # pragma: no cover
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

        # Send the object
        self.send_obj(tensor, worker)

        pointer = tensor.create_pointer(
            owner=self, location=worker, id_at_location=tensor.id, register=True, ptr_id=ptr_id
        )

        return pointer

    def set_obj(self, obj):
        """This adds an object to the registry of objects.

        Args:
            obj (tuple(object, object)): an id, object tuple.
        """

        self._objects[obj.id] = obj

    def get_obj(self, obj_id):
        """Look up an object from the registry using its ID.

        Args:
            obj_id (str or int): the id of an object to look up
        Returns:
            The object that match with the obj_id        
        """

        obj = self._objects[obj_id]

        return obj

    def respond_to_obj_req(self, obj_id):
        obj = self.get_obj(obj_id)
        self.de_register_obj(obj)
        return obj

    def register_obj(self, obj, obj_id=None):
        """Registers an object with the current worker node. 
        
        This method is used by internal processes (hooks and workers). 
        It selects an id for the object, assigns a list of owners, and establishes whether it's a
        pointer or not.

        Args:
            obj (Tensor or Variable): a Torch instance, e.g. Tensor or Variable to be registered
            obj_id ((str or int),optional): The id associated to the object to be registered
        """

        if not self.is_client_worker:
            obj.id = obj_id
            self.set_obj(obj)

    def de_register_obj(self, obj, _recurse_torch_objs=True):
        """Unregister an object and removes attributes which are indicative of
        registration.

        TODO: _recurse_torch_objs is not implemented

        Args:
            obj (a torch.Tensor or torch.autograd.Variable): A Torch instance, 
            e.g. Tensor or Variable to be de-registered
            _recurse_torch_objs (bool): is used if the object is more complex
            and needs to be explored. Is not supported at the moment
        """

        if hasattr(obj, "id"):
            print("removing object")
            self.rm_obj(obj.id)
        if hasattr(obj, "owner"):
            del obj.owner

    def rm_obj(self, remote_key):
        """Removes an object from the permanent object registry if it exists.

        Args:
            remote_key(int or string): id of the object to be removed
        """
        if remote_key in self._objects:
            del self._objects[remote_key]

    # SECTION: convenience methods for constructing frequently used messages

    def send_obj(self, obj, location):
        """Send a torch object to a worker

        Args:
            obj (a torch.Tensor or torch.autograd.Variable): A Torch instance, 
            e.g. Tensor or Variable to be sent
            location (worker): a worker which should receive the object
        """
        return self.send_msg(MSGTYPE_OBJ, obj, location)

    def request_obj(self, obj_id, location):
        """
        Returns:
            obj (a torch.Tensor or torch.autograd.Variable): A Torch instance, 
            e.g. Tensor or Variable requested
        """
        obj = self.send_msg(MSGTYPE_OBJ_REQ, obj_id, location)
        return obj

    # SECTION: Manage the workers network

    def get_worker(self, id_or_worker, fail_hard=False):
        """Allows for resolution of worker ids to workers to happen
        automatically while also making the current worker aware of new ones
        when discovered through other processes.
        
        If you pass in an ID, it will try to find the worker object reference
        within self._known_workers. 

        If you instead pass in a reference, it will save that as a known_worker 
        if it does not exist as one. 

        This method is useful because often tensors have to store only the ID to
        a foreign worker which may or may not be known by the worker that is 
        deserializing it at the time of deserialization.

        Example:
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

        Args:
            id_or_worker (string or int or BaseWorker): This is either the id of the object to be returned or an instance of :class:`BaseWorker` class.
            fail_hard (bool): Wether we want to throw an exception when a worker is not registered at this worker or
                we just want to log it
        Returns:
            The type and ID of the worker (string or int or :class:'BaseWorker')
        """

        if isinstance(id_or_worker, bytes):
            id_or_worker = str(id_or_worker, "utf-8")

        if isinstance(id_or_worker, (str, int)):
            if id_or_worker in self._known_workers:
                return self._known_workers[id_or_worker]
            else:
                if fail_hard: # pragma: no cover
                    raise WorkerNotFoundException ("The worker has not been found")
                logging.warning("Worker", self.id, "couldn't recognize worker", id_or_worker)
                return id_or_worker
        else:
            if id_or_worker.id not in self._known_workers:
                self.add_worker(id_or_worker)

        return id_or_worker

    def add_worker(self, worker):
        """Add a new worker.
        This method adds a worker to the list of
        _known_workers internal to the BaseWorker. It endows this class with
        the ability to communicate with the remote worker being added, such as
        sending and receiving objects, commands, or information about the
        network.

        Example:
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

        Args:
            worker (BaseWorker): An instance of :class:`BaseWorker` class. This is an object
                pointer to a remote worker, which must have a unique id.
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
        """Add workers.
        
        Convenient function to add several workers in a single call

        Args:
            workers (list): the workers to add.
        """

        for worker in workers:
            self.add_worker(worker)

    def __str__(self):
        """to-string method for all classes that extend BaseWorker. 

        Returns:
            Type and ID of the worker

        Example: 
            A VirtualWorker instance with id 'bob' would return a string value of.
            >>> import syft
            >>> bob = syft.VirtualWorker(id="bob")
            >>> bob
            <syft.workers.virtual.VirtualWorker id:bob>

        Note:
            __repr__ calls this method by default.
        """

        out = "<"
        out += str(type(self)).split("'")[1]
        out += " id:" + str(self.id)
        out += ">"
        return out

    def __repr__(self):
        return self.__str__()
