import torch
import json
import msgpack
import time
import logging
import syft as sy
import numpy as np
from abc import ABC, abstractmethod

from .. import utils
from ..frameworks.torch import utils as torch_utils
from ..frameworks import encode
from ..frameworks import encode as syft_encoder_router


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

    def __init__(
        self,
        hook=None,
        id=0,
        is_client_worker=False,
        objects={},
        tmp_objects={},
        known_workers={},
        verbose=True,
        queue_size=0,
    ):

        if hook is None and hasattr(sy, "local_worker"):
            hook = sy.local_worker.hook

        # This is a reference to the hook object which overloaded
        # the underlying deep learning framework
        # (at the time of writing this is exclusively TorchHook)
        self.hook = hook

        # the integer or string identifier for this node
        self.id = id

        # a boolean which determines whether this worker is
        # associated with an end user client. If so, it assumes
        # that the client will maintain control over when variables
        # are instantiated or deleted as opposed to
        # handling tensor/variable/model lifecycle internally.
        self.is_client_worker = is_client_worker

        # The workers permanent registry. When the worker is NOT
        # a client worker, it stores all tensors it receives
        #  or creates in this dictionary. The key to each object
        # is it's id.
        self._objects = {}
        self._pointers = {known_worker.id: {} for known_worker in known_workers}
        for k, v in objects.items():
            self._objects[k] = v
            # Register the pointer by location/id@location
            if isinstance(v, sy._PointerTensor):
                v.register_pointer()

        # The temporary registry. When the worker IS a client
        # worker, it stores some tensors temporarily in this
        # _tmp_objects simply to ensure that they do not get
        # de-allocated by the Python garbage collector while
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

        if hasattr(sy, "local_worker"):
            sy.local_worker.add_worker(self)
            self.add_worker(sy.local_worker)

    def whoami(self):
        """Returns metadata information about the worker.

        This function returns the default which is the id and type of
        the current worker. Other worker types can extend this function
        with additional metadata such as network information.
        """

        return self.encode_msg({"id": self.id, "type": type(self)})

    def _search(self, query):
        """Queries all local tensors which have string ids, returning the
        tensors which match the query. The query is composed of one or more
        substrings which all be contained within a tensor's ID in order for it
        to be a match.

        :param query: either a string or a list of strings
        :return: a list of tensors
        """

        if isinstance(query, str):
            query = {query}
        else:
            query = set(query)

        results = set()
        for id in self._objects.keys():
            if isinstance(id, str):
                failed = False
                for constraint in query:
                    if constraint not in id:
                        failed = True
                if not failed:
                    results.add(id)
        return results

    def search(self, query="#boston"):
        """Queries all local tensors which have string ids, returning the
        tensors which match the query. The query is composed of one or more
        substrings which all be contained within a tensor's ID in order for it
        to be a match.

        :param query: either a string or a list of strings
        :return: a list of tensors
        """

        return self._search(query)

    def send_msg(self, message, message_type, recipient):
        """Sends a string message to another worker with message_type
        information indicating how the message should be processed.

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

        # create a an empty message wrapper
        message_wrapper = {}

        # this is the contents of the message sent from a framework
        message_wrapper["message"] = message

        # this information determins how the message is routed when it is received
        # you can find how this type information is handled in the function
        # process_message_type. At present it includes obj, req_obj, torch_cmd,
        # numpy_cmd, composite, and query as possible values.
        message_wrapper["type"] = message_type

        # if you're planning to send groups of messages at a time, this appends
        # the message to a queue. However, by default the queue size is set to 1
        # which means that this doesn't do anything.
        self.message_queue.append(message_wrapper)

        # manages grouping messages (for faster performance)
        if self.queue_size:
            if len(self.message_queue) > self.queue_size:

                # if the queue is full, reset the message_wrapper object
                # with all messages to be set.
                message_wrapper = self.compile_composite_message()
            else:
                return None

        # this packages the message dictionary into JSON and adds a final newline
        # i believe the extra newline was necessary - possibly to make decoding
        # batches of messages working. Note that .encode() converts this json to
        # binary
        message_wrapper_json = self.encode_msg(message_wrapper)

        # empty the message queue which previously held our messages
        self.message_queue = []

        # since all logic for this class is general to ALL worker types, we now
        # need to call the worker-specific message send function wihch sends
        # the message according to the correct protocol (such as HTTPS, Socket,
        # or other ways of sending messages).
        return self._send_msg(message_wrapper_json, recipient)

    def compile_composite_message(self):
        """Returns a composite message in a dictionary from the message queue.
        Evenatually will take a recipient id.

        * **out (dict)** dictionary containing the message queue compiled
        as a composite message
        """

        # create new message container
        message_wrapper = {}

        # iterate through messages and create a list
        message_wrapper["message"] = {
            message_number: message
            for message_number, message in enumerate(self.message_queue)
        }
        message_wrapper["type"] = "composite"

        return message_wrapper

    @abstractmethod
    def _send_msg(self, message_wrapper_json_binary, recipient):
        """Sends a string message to another worker with message_type
        information indicating how the message should be processed. Note that
        this function is to be overridden by actual worker classes (BaseWorker
        is the generic class). For example, if this was an HTTPWorker this
        function would initiate an HTTP request and send the contents of
        message_wrapper_json_binary.

        :Parameters:

        * **recipient (** :class:`VirtualWorker` **)** the worker being sent a message.

        * **message_wrapper_json_binary (binary)** the message being sent encoded in binary

        * **out (object)** the response from the message being sent. This can be a variety
          of object types. However, the object is typically only used during testing or
          local development with :class:`VirtualWorker` workers.
        """

        raise NotImplementedError

    def receive_msg(self, message_wrapper_json):
        """Receives an message from a worker and then executes its contents
        appropriately. The message is encoded as a binary blob.

        * **message (binary)** the message being sent

        * **out (object)** the response. This can be a variety
          of object types. However, the object is typically only used during testing or
          local development with :class:`VirtualWorker` workers.
        """

        # load json into a dictionary where all objects have been deserialized
        message_wrapper = encode.decode(
            self.decode_msg(message_wrapper_json), worker=self
        )

        # route message to appropriate logic and execute the command, returning
        # the "response" which should be sent back to the original worker. "private" (bool)
        # determines whether we are intentioanlly leaving out the data in the response
        # and instead sending pointers to the data which we will actually keep locally
        response, private = self.process_message_type(message_wrapper)

        # serialize any objects in the response into their string/dictionary form (recursive)
        response = encode.encode(
            response, retrieve_pointers=False, private_local=private
        )

        response = self.encode_msg(response)

        return response

    def encode_msg(self, msg):
        # convert the response dictionary to a string and then convert that string to binary
        return msgpack.packb(msg, use_bin_type=True)

    def decode_msg(self, msg):
        return msgpack.unpackb(msg, raw=False)

    def process_message_type(self, message_wrapper):
        """This method takes a message wrapper and attempts to process it
        agaist known processing methods. If the method is a composite message,
        it unroles applies recursively.

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

        # the contents of the message
        message = message_wrapper["message"]

        # this series of if/else statements uses the message_wrapper['type']
        # value to determine where to route the incoming message.

        # if the message contains an object being sent to us
        if message_wrapper["type"] == "obj":

            object = message

            # if object is a numpy array
            if isinstance(message, np.ndarray):
                """do nothing."""

            # if object is a Torch object - pre-process it for registration
            else:
                torch_utils.fix_chain_ends(object)
                # torch_utils.assert_is_chain_well_formed(object)

            # register the object, saving it in self._objects and ensuring that
            # object.owner is set correctly
            self.register(object)

            # we do not send a response back
            # TODO: send a "successful" or "not successful" response?
            return {}, False

        # if the message contains Receiving a request for an object
        # to be sent to another worker. For example "x.get()" would execute here.
        # if x is a pointer to an object hosted on this worker.
        elif message_wrapper["type"] == "req_obj":

            # Because it was pointed at, it's the first syft_object of the chain,
            # so its parent is the tensorvar
            object = self.get_obj(message)

            # if object being returned is a numpy array
            if isinstance(object, np.ndarray):
                """"""
                # delete the numpy array from our local registry
                self.de_register(object)

                # send the numpy array back to the worker that asked for it
                return object, False

            # object is a pytorch array
            else:

                # if the object is NOT a variable, then we simply
                # take the object's parent, and return the entire object
                # all children will be serialized recursively
                tensorvar = object.parent

                # if the object is a variable, we have to make special
                # considerations to ensure that the data and grad are all
                # properly deregistered
                if torch_utils.is_variable_name(object.torch_type):
                    syft_data_object = tensorvar.data.child
                    self.de_register(syft_data_object)
                    if tensorvar.grad is not None:
                        syft_grad_object = tensorvar.grad.child
                        self.de_register(syft_grad_object)

                # deregister the object
                self.de_register(object)

                # return the object
                # False means we're actually return the data (it's not private)
                return tensorvar, False

        #  A torch command from another worker involving one or more tensors
        #  hosted locally. For example: "z = x + y" would execute here.
        elif message_wrapper["type"] == "torch_cmd":

            # route the command to the torch command logic
            result = self.process_torch_command(message)

            # save the results locally in self._objects and ensure
            # that .owner is set correctly
            torch_utils.enforce_owner(result, self)
            if torch_utils.is_variable(result):
                if result.grad is not None:
                    torch_utils.link_var_chain_to_data_and_grad_chains(
                        result, result.data, result.grad
                    )
                else:
                    torch_utils.link_var_chain_to_data_chain(result, result.data)

            self.register(result)

            # return result of torch operation
            # Result is private - so only actually return a pointer to the result
            return result, True

        # a numpy command from another worker involving one or more local numpy arrays
        # hosted locally. For example "z = x + y" would execute here.
        elif message_wrapper["type"] == "numpy_cmd":

            # route the command to the numpy command logic
            result = self.process_numpy_command(message)

            # save the result locally in self._objects and ensure that
            # .owner is set correctly.
            self.register(result)

            # return teh result of the numpy operation
            # Result is private - so only actually return a pointer to result
            return result, True

        # A composite command. Must be unrolled
        elif message_wrapper["type"] == "composite":
            raise NotImplementedError("Composite command not handled at the moment")

        # a message asking for a list of tensors which fit a certain criteria.
        # at the time of writing this comment, this is a partial string match on the id
        # of the tensor. For example, if self._workers has a tensor with an id
        # "12345 #boston_housing #input" then a query from a pointer to this worker of
        # bob.search("#boston_housing") would return a list of pointers including
        # the one with the "12345 #boston_housing #input" id.
        elif message_wrapper["type"] == "query":

            # perform the search over all tensors on the worker
            tensors = self.search(message)

            # convert the resulting tensors to pointers
            pointers = []

            for tensor in tensors:

                # initialize a pointer to the tensor.
                ptr = tensor.parent.create_pointer()

                # serialize the pointer recursively
                encoding = encode.encode(
                    ptr, private_local=False, retrieve_pointers=True
                )

                pointers.append(encoding)

            # return the list of pointers.
            return pointers, True

        # Hopeflly we don't reach this point.
        return "Unrecognized message type:" + message_wrapper["type"]

    def __str__(self):
        """This is a simple to-string for all classes that extend BaseWorker
        which just returns the type and ID of the worker. For example, a
        VirtualWorker instance with id 'bob' would return a string value of.

        <syft.core.workers.virtual.VirtualWorker id:bob>

        Note that __repr__ calls this method by default.
        """

        out = "<"
        out += str(type(self)).split("'")[1]
        out += " id:" + str(self.id)
        out += ">"
        return out

    def __repr__(self):
        return self.__str__()

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
        >>> x = sy.FloatTensor([1,2,3,4,5])
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
        if worker.id in self._known_workers and self.verbose:
            logging.warning(
                "Worker ID "
                + str(worker.id)
                + " taken. Have I seen this worker before?"
            )
            logging.warning(
                "Replacing it anyways... this could cause unexpected behavior..."
            )
        if worker.id in self._known_workers:
            logging.warn(
                "Worker "
                + str(worker.id)
                + " already exists. Replacing old worker which could cause unexpected behavior"
            )

        # add worker to the list of known workers
        # it's just a mapping from ID->object
        self._known_workers[worker.id] = worker

        self._pointers[worker.id] = {}

    def add_workers(self, workers):

        for worker in workers:
            self.add_worker(worker)

    def get_worker(self, id_or_worker):
        """get_worker(self, id_or_worker) -> BaseWorker If you pass in an ID,
        it will attempt to find the worker object (pointer) withins
        self._known_workers. If you instead pass in a pointer itself, it will
        save that as a known_worker if it does not exist as one. This method is
        primarily useful becuase often tensors have to store only the ID to a
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

        if issubclass(type(id_or_worker), BaseWorker):
            if id_or_worker.id not in self._known_workers:
                self.add_worker(id_or_worker)
            result = self._known_workers[id_or_worker.id]
        else:
            if id_or_worker in self._known_workers:
                result = self._known_workers[id_or_worker]
            else:
                logging.warning(
                    "Worker", self.id, "couldnt recognize worker", id_or_worker
                )
                result = id_or_worker

        if result is not None:
            return result
        else:
            return id_or_worker

    def get_obj(self, remote_key):
        """get_obj(remote_key) -> a torch object This method fetches a tensor
        from the worker's internal registry of tensors using its id. Note that
        this will not work on client worker because they do not store objects
        internally. However, it can be used on remote workers, including remote
        :class:`VirtualWorker` objects, as pictured below in the example.

        :Parameters:

        * **remote_key (string or int)** This is the id of
          the object to be returned.
        """

        try:
            obj = self._objects[remote_key]
        except:
            msg = (
                'Tensor "'
                + str(remote_key)
                + '" not found on worker "'
                + str(self.id)
                + '"!!!\n\n'
            )
            msg += (
                "You just tried to interact with an object ID:"
                + str(remote_key)
                + " on worker "
                + str(self.id)
                + " which does not exist!!! "
            )
            msg += "Use .send() and .get() on all your tensors to make sure they're on the same machines.\n\n"
            msg += "If you think this tensor does exist, check the ._objects dictionary on the worker and see for"
            msg += " yourself!!! "
            msg += "The most common reason this error happens is because someone calls .get() on the object's"
            msg += " pointer without realizing it (which deletes the remote object and sends it to the pointer)."
            msg += " Check your code to make sure you haven't already called .get() on this pointer!!!"

            raise Exception(msg)
        # Fix ownership if the obj has been modified out of control (like with backward())
        torch_utils.enforce_owner(obj, self)
        return obj

    def set_obj(self, remote_key, value, force=False, tmp=False):
        """This method adds an object (such as a tensor) to one of the internal
        object registries. The main registry holds objects for a long period of
        time (until they are explicitly deleted) while the temporary registory
        will delete all object references contained therein when the cleanup
        method is called (self._clear_tmp_objects).

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

        * **tmp (bool, optional)** if set to True, this will allow an object
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
        """This method removes an object from the permament object registory if
        it exists.

        :parameters:

        * **remote_key(int or string)** the id of the object to be removed
        """

        if remote_key in self._objects:
            obj = self._objects[remote_key]
            if isinstance(obj, sy._PointerTensor):
                pointer = obj
                location = (
                    pointer.location
                    if isinstance(pointer.location, (int, str))
                    else pointer.location.id
                )
                id_at_location = pointer.id_at_location
                if location in self._pointers.keys():
                    if id_at_location in self._pointers[location].keys():
                        del self._pointers[location][id_at_location]
            del self._objects[remote_key]

    def _clear_tmp_objects(self):
        """This method releases all objects from the temporary registry."""
        self._tmp_objects = {}

    def de_register(self, obj):

        """Un-register an object and its attribute."""
        if torch_utils.is_syft_tensor(obj):
            self.rm_obj(obj.id)
        elif torch_utils.is_tensor(obj):
            self.de_register(obj.child)
        elif torch_utils.is_variable(obj):
            self.de_register(obj.child)
            self.de_register(obj.data.child)
        # Case of a iter type non json serializable
        elif isinstance(obj, (list, tuple, set, bytearray, range)):
            for o in obj:
                self.de_register(o)
        elif obj is None:
            """do nothing."""
        elif isinstance(obj, np.ndarray):
            self.rm_obj(obj.id)
        else:
            raise TypeError("The type", type(obj), "is not supported at the moment")
        return

    def de_register_object(self, obj, _recurse_torch_objs=True):
        """Unregisters an object and removes attributes which are indicative of
        registration.

        Note that the way in which attributes are deleted has been
        informed by this StackOverflow post: https://goo.gl/CBEKLK
        """

        is_torch_tensor = torch_utils.is_tensor(obj)

        if not is_torch_tensor:
            if hasattr(obj, "id"):
                self.rm_obj(obj.id)
                del obj.id
            if hasattr(obj, "owner"):
                del obj.owner

        if hasattr(obj, "child"):
            if obj.child is not None:
                if is_torch_tensor:
                    if _recurse_torch_objs:
                        self.de_register_object(obj.child, _recurse_torch_objs=False)
                else:
                    self.de_register_object(
                        obj.child, _recurse_torch_objs=_recurse_torch_objs
                    )
            if not is_torch_tensor:
                delattr(obj, "child")

    def register(self, result):
        """Register an object with SyftTensors."""
        if torch_utils.is_syft_tensor(result):
            syft_obj = result
            self.register_object(syft_obj)
        elif torch_utils.is_tensor(result):
            tensor = result
            self.register_object(tensor.child)
        elif torch_utils.is_variable(result):
            variable = result
            self.register(variable.child)
            self.register(variable.data.child)
            if not hasattr(variable, "grad") or variable.grad is None:
                variable.init_grad_()
            self.register(variable.grad.child)
            self.register(variable.grad.data.child)
        # Case of a iter type non json serializable
        elif isinstance(result, (list, tuple, set, bytearray, range)):
            for res in result:
                self.register(res)
        elif result is None:
            """do nothing."""
        elif isinstance(result, np.ndarray):
            self.register_object(result)
        else:
            raise TypeError("The type", type(result), "is not supported at the moment")
        return

    def register_object(self, obj, id=None):
        """Registers an object with the current worker node. Selects an id for
        the object, assigns a list of owners, and establishes whether it's a
        pointer or not. This method is generally not used by the client and is
        instead used by internal processes (hooks and workers).

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
        # if not torch_utils.is_syft_tensor(obj):
        # raise TypeError("Can't register a non-SyftTensor")

        if id is None:
            id = obj.id
        else:
            obj.id = id

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

    def process_numpy_command(self, command_msg):
        """process_command(self, command_msg) -> (command output, list of
        owners) Process a command message from a client worker. Returns the
        result of the computation and a list of the result's owners.

        :Parameters:

        * **command_msg (dict)** The dictionary containing a
          command from another worker.

        * **out (command output, list of** :class:`BaseWorker`
          ids/objects **)** This executes the command
          and returns its output along with a list of
          the owners of the tensors involved.
        """

        # torch_utils.assert_has_only_torch_tensorvars(command_msg)

        attr = command_msg["command"]
        has_self = command_msg["has_self"]
        args = command_msg["args"]
        kwargs = command_msg["kwargs"]
        self_ = command_msg["self"] if has_self else None

        return self._execute_numpy_call(attr, self_, *args, **kwargs)

    def _execute_numpy_call(self, attr, self_, *args, **kwargs):
        """Transmit the call to the appropriate TensorType for handling."""

        # Distinguish between a command with torch tensors (like when called by the client,
        # or received from another worker), and a command with syft tensor, which can occur
        # when a function is overloaded by a SyftTensor (for instance _PlusIsMinusTensor
        # overloads add and replace it by sub)
        # try:
        #     torch_utils.assert_has_only_torch_tensorvars((args, kwargs))
        #     is_torch_command = True
        # except AssertionError:
        is_torch_command = False

        has_self = self_ is not None

        # if has_self:
        #     command = torch._command_guard(attr, 'tensorvar_methods')
        # else:
        #     command = torch._command_guard(attr, 'torch_modules')
        command = attr

        raw_command = {
            "command": command,
            "has_self": has_self,
            "args": args,
            "kwargs": kwargs,
        }
        if has_self:
            raw_command["self"] = self_

        # if is_torch_command:
        #     # Unwrap the torch wrapper
        #     syft_command, child_type = torch_utils.prepare_child_command(
        #         raw_command, replace_tensorvar_with_child=True)
        # else:
        #     # Get the next syft class
        #     # The actual syft class is the one which redirected (see the  _PlusIsMinus ex.)
        #     syft_command, child_type = torch_utils.prepare_child_command(
        #         raw_command, replace_tensorvar_with_child=True)
        #
        #     torch_utils.assert_has_only_syft_tensors(syft_command)

        # Note: because we have pb of registration of tensors with the right worker,
        # and because having Virtual workers creates even more ambiguity, we specify the worker
        # performing the operation
        # torch_utils.enforce_owner(raw_command, self)

        result = sy.array.handle_call(raw_command, owner=self)

        torch_utils.enforce_owner(result, self)

        if is_torch_command:
            # Wrap the result
            if has_self and utils.is_in_place_method(attr):
                result = torch_utils.wrap_command_with(result, raw_command["self"])
            else:
                result = torch_utils.wrap_command(result)

        return result

    def process_torch_command(self, command_msg):
        """process_command(self, command_msg) -> (command output, list of
        owners) Process a command message from a client worker. Returns the
        result of the computation and a list of the result's owners.

        :Parameters:

        * **command_msg (dict)** The dictionary containing a
          command from another worker.

        * **out (command output, list of** :class:`BaseWorker`
          ids/objects **)** This executes the command
          and returns its output along with a list of
          the owners of the tensors involved.
        """

        # torch_utils.assert_has_only_torch_tensorvars(command_msg)

        attr = command_msg["command"]
        has_self = command_msg["has_self"]
        args = command_msg["args"]
        kwargs = command_msg["kwargs"]
        self_ = command_msg["self"] if has_self else None

        return self._execute_call(attr, self_, *args, **kwargs)

    def _execute_call(self, attr, self_, *args, **kwargs):
        """Transmit the call to the appropriate TensorType for handling."""

        # if this is none - then it means that self_ is not a torch wrapper
        # and we need to execute one level higher TODO: not ok for complex args
        if self_ is not None and self_.child is None:
            new_args = [arg.wrap(True) for arg in args]
            return self._execute_call(attr, self_.wrap(True), *new_args, **kwargs)

        # Distinguish between a command with torch tensors (like when called by the client,
        # or received from another worker), and a command with syft tensor, which can occur
        # when a function is overloaded by a SyftTensor (for instance _PlusIsMinusTensor
        # overloads add and replace it by sub)
        try:
            torch_utils.assert_has_only_torch_tensorvars((args, kwargs))
            is_torch_command = True
        except AssertionError:
            is_torch_command = False

        has_self = self_ is not None

        if has_self:
            command = torch._command_guard(attr, "tensorvar_methods")
        else:
            command = torch._command_guard(attr, "torch_modules")

        raw_command = {
            "command": command,
            "has_self": has_self,
            "args": args,
            "kwargs": kwargs,
        }
        if has_self:
            raw_command["self"] = self_
        if is_torch_command:
            # Unwrap the torch wrapper
            syft_command, child_type = torch_utils.prepare_child_command(
                raw_command, replace_tensorvar_with_child=True
            )
        else:
            # Get the next syft class
            # The actual syft class is the one which redirected (see the  _PlusIsMinus ex.)
            syft_command, child_type = torch_utils.prepare_child_command(
                raw_command, replace_tensorvar_with_child=True
            )

            # torch_utils.assert_has_only_syft_tensors(syft_command)

        # Note: because we have pb of registration of tensors with the right worker,
        # and because having Virtual workers creates even more ambiguity, we specify the worker
        # performing the operation

        result = child_type.handle_call(syft_command, owner=self)

        torch_utils.enforce_owner((raw_command, result), self)

        if is_torch_command:
            # Wrap the result
            if has_self and utils.is_in_place_method(attr):
                # TODO: fix this properly: don't wrap the same way if syft or Variable
                if torch_utils.is_variable(result) or torch_utils.is_tensor(result):
                    wrapper = torch_utils.wrap_command_with(
                        result.child, raw_command["self"]
                    )
                else:
                    wrapper = torch_utils.wrap_command_with(result, raw_command["self"])
            else:
                wrapper = torch_utils.wrap_command(result)
            return wrapper
        else:
            # We don't need to wrap
            return result

    def send_obj(
        self,
        object,
        new_id,
        recipient,
        new_data_id=None,
        new_grad_id=None,
        new_grad_data_id=None,
    ):
        """send_obj(self, obj, new_id, recipient, new_data_id=None) -> obj
        Sends an object to another :class:`VirtualWorker` and removes it from
        the local worker.

        :Parameters:
        * **object (object)** a python object to be sent
        * **new_id (int)** the id where the object should be stored
        * **recipient (** :class:`VirtualWorker` **)** the worker object to send the message to.
        """

        # if the object is a torch object, run some special checks, otherwise just set the ID
        if hasattr(object, "child"):
            object.child.id = new_id

            if torch_utils.is_variable_name(object.child.torch_type):
                if (
                    new_data_id is None
                    or new_grad_id is None
                    or new_grad_data_id is None
                ):
                    raise AttributeError(
                        "Please provide the new_data_id, new_grad_id, and new_grad_data_id args, to be able to point to"
                        + "Var.data, .grad"
                    )

                if self.get_pointer_to(recipient, new_data_id) is not None:
                    raise MemoryError("You already point at ", recipient, ":", new_id)
                assert (
                    new_id != new_data_id
                ), "You can't have the same id for the variable and its data."
                assert (
                    new_id != new_grad_id
                ), "You can't have the same id for the variable and its grad."
                assert new_id != new_grad_data_id

                assert new_data_id != new_grad_id

                assert new_data_id != new_grad_data_id

                assert new_grad_id != new_grad_data_id

                object.data.child.id = new_data_id

                if object.grad is None:
                    object.init_grad_()

                object.grad.child.id = new_grad_id

                object.grad.data.child.id = new_grad_data_id
        else:
            object.id = new_id

        if self.get_pointer_to(recipient, new_id) is not None:
            raise MemoryError("You already point at ", recipient, ":", new_id)
        
        if self is recipient:
            raise MemoryError("The recipient {} is the same as the owner {} of the object {} that you are trying to send".format(recipient, self, object.id))
            
        object = encode.encode(object, retrieve_pointers=False, private_local=False)

        # We don't need any response to proceed to registration
        self.send_msg(message=object, message_type="obj", recipient=recipient)

    def send_torch_command(self, recipient, message):
        """send_torch_command(self, recipient, message) -> object.

        This method sends a message to another worker in a way that hangs... waiting until the
        worker responds with a message. It then processes the response using a response handler

        :Parameters:

        * **recipient (** :class:`VirtualWorker` **)** the worker being sent a message.

        * **message (string)** the message being sent
        """
        return self.send_command(recipient, message, framework="torch")

    def send_command(self, recipient, message, framework="torch"):

        if isinstance(recipient, (str, int)):
            raise TypeError("Recipient should be a worker object not his id.")

        response = self.send_msg(
            message=message, message_type=framework + "_cmd", recipient=recipient
        )

        response = encode.decode(response, worker=self)

        return response

    def request_obj(self, obj_id, recipient):
        """request_obj(self, obj_id, sender) This method requests that another
        VirtualWorker send an object to the local one. In the case that the
        local one is a client, it simply returns the object. In the case that
        the local worker is not a client, it stores the object in the permanent
        registry.

        :Parameters:

        * **obj_id (str or int)** the id of the object being requested

        * **sender (** :class:`VirtualWorker` **)** the worker who currently has the
          object who is being requested to send it.
        """

        object = self.send_msg(
            message=obj_id, message_type="req_obj", recipient=recipient
        )

        object = encode.decode(object, worker=self)

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
        # We keep a dict with keys = owners and subkeys id@loc : self._pointers[location][id@loc] = obj_id
        # But it has to be updated every time you add, SEND or de_register a pointer
        if not isinstance(location, (int, str)):
            location = location.id

        if location in self._pointers.keys():
            if id_at_location in self._pointers[location].keys():
                object_id = self._pointers[location][id_at_location]
                # Note that the following condition can be false if you send multiple times a pointer,
                # Because then we don't de-register the old pointer in self._pointers
                if object_id in self._objects:
                    return self._objects[object_id]
