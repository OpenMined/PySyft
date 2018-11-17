from abc import ABC, abstractmethod

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

    def __init__(self):

        # This is the core object in every BaseWorker instantiation, a collection of
        # objects. All objects are stored using their IDs as keys.
        self._objects = {}

        # For performance, we cache each
        self._message_router = {}
        self._message_router[MSGTYPE_OBJ] = self.set_obj
        self._message_router[MSGTYPE_OBJ_REQ] = self.get_obj

    # SECTION: Methods which MUST be overridden by subclasses

    @abstractmethod
    def _send_msg(self, message, location):
        NotImplementedError  # pragma: no cover

    @abstractmethod
    def _recv_msg(self, message):
        NotImplementedError  # pragma: no cover

    # SECTION: Generic Message Sending/Receiving Logic
    # EVery message uses these methods.

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
        self._objects[obj.id] = obj

    def get_obj(self, obj_id):
        return self._objects[obj_id]

    # SECTION: convenience methods for constructing frequently used messages

    def send_obj(self, obj, location):
        return self.send_msg(MSGTYPE_OBJ, obj, location)

    def request_obj(self, obj_id, location):
        return self.send_msg(MSGTYPE_OBJ_REQ, obj_id, location)
