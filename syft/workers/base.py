import logging
import random

from abc import abstractmethod
from syft.util import WorkerNotFoundException
from syft import serde
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

    """

    def __init__(self, hook=None, id=0, known_workers={}, is_client_worker=False):
        """Initializes a BaseWorker

        Args:
            hook (.hook.TorchHook, optional): A reference to the hook object
                which is used to modify PyTorch with PySyft's functionality.

            id (str or int, optional): Unique id of the worker

            known_workers (dict, optional): A dictionary of all known workers on
                a network which this worker may need to communicate with in the
                future. The key of each should be each worker's unique ID and
                the value should be a worker class which extends BaseWorker.
                Extensions of BaseWorker will include advanced functionality
                for adding to this dictionary(node discovery). In some cases,
                one can initialize this with known workers to help bootstrap the
                network.


            is_client_worker (bool, optional): True or False based on whether
                this worker is associated with an end user client. If so, it
                assumes that the client will maintain control over when
                variables are instantiated or deleted as opposed to handling
                tensor/variable/model lifecycle internally. Set to True if this
                object is not where the objects will be stored, but is instead
                a pointer to a worker that eists elsewhere.
        """
        self.hook = hook
        self.id = id
        self.is_client_worker = is_client_worker
        # A core object in every BaseWorker instantiation. A Collection of
        # objects where all objects are stored using their IDs as keys.
        self._objects = {}
        self._known_workers = {}
        for k, v in known_workers.items():
            self._known_workers[k] = v
        self.add_worker(self)
        # For performance, we cache each
        self._message_router = {
            MSGTYPE.OBJ: self.set_obj,
            MSGTYPE.OBJ_REQ: self.respond_to_obj_req,
            MSGTYPE.OBJ_DEL: self.rm_obj,
        }

    # SECTION: Methods which MUST be overridden by subclasses
    @abstractmethod
    def _send_msg(self, message, location):
        """As BaseWorker implies, you should never instantiate this class by
        itself. Instead, you should extend BaseWorker in a new class which
        instantiates _send_msg and _recv_msg, each of which should specify the
        exact way in which two workers communicate with each other. The easiest
        example to study is VirtualWorker.

        Args:
            message (str): the message being sent from one worker to another.

            location (:class:`.workers.BaseWorker`) the destination to send the
                message.

        Raises:
            NotImplementedError:

        """

        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def _recv_msg(self, message):
        """As BaseWorker implies, you should never instantiate this class by
        itself. Instead, you should extend BaseWorker in a new class which
        instantiates _send_msg and _recv_msg, each of which should specify the
        exact way in which two workers communicate with each other. The easiest
        example to study is VirtualWorker.

        Args:
            message (str): the message being received.

        Raises:
            NotImplementedError:

        """
        raise NotImplementedError  # pragma: no cover

    def send_msg(self, msg_type, message, location):
        """Message Sending Logic. Every message uses this method.

        Returns:
            response:
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
        """Message receiving Logic. Every message uses this method.

        Returns:
            bin_response:
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
        """Send a syft or torch tensor and his child, sub-child, etc (all the
        syft chain of children) to a worker, or a list of workers, with a given
        remote storage address.

        Args:
            tensor (torch.Tensor): the syft or torch tensor to send

            workers (:class:`....workers.BaseWorker`): the workers which will
            receive the object

            id ((str or int), optional): the remote id of the object on the
            remote worker(s).

        Example:
            >>> x.send(bob, 1000)
            >>> #will result in bob having the tensor x with id 1000
        """
        if not isinstance(workers, list):
            workers = [workers]

        assert len(workers) > 0, "Please provide workers to receive the data"

        if len(workers) == 1:
            worker = workers[0]
        else:
            # If multiple workers, you want to send the same tensor to multiple
            # workers. Assumingly you'll get multiple pointers, or a pointer
            # with different locations
            raise NotImplementedError(
                "Sending to multiple workers is not \
                                        supported at the moment"
            )

        worker = self.get_worker(worker)

        if ptr_id is None:  # Define a remote id if not specified
            ptr_id = int(10e10 * random.random())

        # Send the object

        # obj_is_new_to_recipient is a boolean value which is TRUE if
        # and only if the object was not already on the remote worker.
        # the reason we need it is to prevent accidentally deleting
        # the object we just sent when a previous reference to that
        # object gets deleted... aka

        # x_ptr = x.send(bob) #this works fine
        # x_ptr = x.send(bob) #this would result in bob having NO obj

        # however, by using this boolean value we can tell whether or
        # not to create a new pointer or to re-use the old one
        # which keeps the old one from sending a rm_obj command
        # to the remote worker when it gets garbage collected
        obj_is_new_to_recipient = self.send_obj(tensor, worker)

        if obj_is_new_to_recipient:
            pointer = tensor.create_pointer(
                owner=self, location=worker, id_at_location=tensor.id, register=True, ptr_id=ptr_id
            )

            return pointer
        else:
            return tensor.ptr()

    def set_obj(self, obj):
        """This adds an object to the registry of objects.

        Args:
          obj_data (tuple(object, object)): an id, object tuple.

        """
        if obj.id not in self._objects:
            self._objects[obj.id] = obj
            return True
        else:
            return False

    def get_obj(self, obj_id):
        """Look up an object from the registry using its ID.

        Args:
          obj_id (str or int): the id of an object to look up

          out (object): the object being returned

        """

        obj = self._objects[obj_id]

        return obj

    def respond_to_obj_req(self, obj_id):

        obj = self.get_obj(obj_id)
        self.de_register_obj(obj)
        return obj

    def register_obj(self, obj, obj_id=None):
        """Registers an object with the current worker node. Selects an id for
        the object, assigns a list of owners, and establishes whether it's a
        pointer or not. This method is generally not used by the client and is
        instead used by internal processes (hooks and workers).

        Args:
          obj (a torch.Tensor or torch.autograd.Variable): A Torch instance,
          e.g. Tensor or Variable to be registered

          force_attach_to_worker (bool): if set to True, it will force the
          object to be stored in the worker's permanent registry

          temporary (bool): If set to True, it will store the object in the
          worker's temporary registry.

        Keyword Args:
          id (int or string): random integer between 0 and 1e10 or
          string uniquely identifying the object.

          owners (list of :class:`BaseWorker` objects ** or ids):
          owner(s) of the object

          is_pointer (bool, optional): Whether or not the tensor being
          registered contains the data locally or is instead a pointer to
          a tensor that lives on a different worker.
        """
        if not self.is_client_worker:
            self.set_obj(obj)

    def de_register_obj(self, obj, _recurse_torch_objs=True):
        """Unregister an object and removes attributes which are indicative of
        registration.

        Args:
          obj (a torch.Tensor or torch.autograd.Variable): A Torch instance,
          e.g. Tensor or Variable to be de-registered

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

        """
        Args:
          obj (a torch.Tensor or torch.autograd.Variable): A Torch instance,
          e.g. Tensor or Variable to be sent
        """
        return self.send_msg(MSGTYPE.OBJ, obj, location)

    def request_obj(self, obj_id, location):
        """
        Returns:
          obj (a torch.Tensor or torch.autograd.Variable): A Torch instance,
          e.g. Tensor or Variable requested
        """
        obj = self.send_msg(MSGTYPE.OBJ_REQ, obj_id, location)

        return obj

        # SECTION: Manage the workers network

    def get_worker(self, id_or_worker, fail_hard=False):
        """ Allows for resolution of worker ids to workers to happen
        automatically while also making the current worker aware of new ones
        when discovered through other processes.

        If you pass in an ID, it will try to find the worker object reference
        within self._known_workers.

        If you instead pass in a reference, it will save that as a known_worker
        if it does not exist as one.

        This method is useful because often tensors have to store only the ID to
        a foreign worker which may or may not be known by the worker that is
        deserializing it at the time of deserialization.

        Args:
            id_or_worker (string or int or :class:`BaseWorker`): id of the
            object to be returned or the object itself.

            fail_hard (bool): Whether we want to throw an exception when a
            worker is not registered at this worker or we just want to log it.

        Returns:
            id_or_worker (string or int or :class:'BaseWorker')

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
                logging.warning("Worker", self.id, "couldnt recognize worker", id_or_worker)
                return id_or_worker
        else:
            if id_or_worker.id not in self._known_workers:
                self.add_worker(id_or_worker)

        return id_or_worker

    def add_worker(self, worker):
        """Adds a worker to the list of _known_workers internal to the
        BaseWorker.

        Endows this class with the ability to communicate with the remote worker
        being added, such as sending and receiving objects, commands, or
        information about the network.

        Args:
            worker (:class:`BaseWorker`): An object pointer to a remote worker,
            which must have a unique id.

        Example:

        >>> import syft as sy
        >>> hook = sy.TorchHook(verbose=False)
        >>> me = hook.local_worker
        >>> bob = sy.VirtualWorker(id="bob",hook=hook, is_client_worker=False)
        >>> me.add_worker([bob])
        >>> x = sy.Tensor([1,2,3,4,5])
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
        """Function to add several workers in a single call

        Args:
            workers (list): Workers to add.
        """
        for worker in workers:
            self.add_worker(worker)

    def __str__(self):
        """to-string method for all classes that extend BaseWorker.

        Returns:
            Type and ID of the worker

        Example:
        A VirtualWorker instance with id 'bob' would return a string value of.
        >>> bob
        <syft.core.workers.virtual.VirtualWorker id:bob>

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
