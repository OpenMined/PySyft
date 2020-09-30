"""A Pointer is the main handler when interacting with remote data.
A Pointer object represents an API for interacting with data (of any type)
at a specific location. Pointer should never be instantiated, only subclassed.

The relation between pointers and data is many to one,
there can be multiple pointers pointing to the same piece of data, meanwhile,
a pointer cannot point to multiple data sources.

A pointer is just an object id on a remote location and a set of methods that can be
executed on the remote machine directly on that object. One note that has to be made
is that all operations between pointers will return a pointer, the only way to have access
to the result is by calling .get() on the pointer.

There are two proper ways of receiving a pointer on some data:
    1. When sending that data on a remote machine the user receives a pointer.
    2. When the user searches for the data in an object store it receives a pointer to that data,
    if it has the correct permissions for that.

After receiving a pointer, one might want to get the data behind the pointer locally. For that the
user should:
    1. Request access by calling .request().
    Example:

    .. code-block::

        pointer_object.request(request_name = "Request name", reason = "Request reason")

    2.1 - The data owner has to approve the request (check the domain node docs).
    2.2 - The data user checks if the request has been approved (check the domain node docs).
    3. After the request has been approved, the data user can call .get() on the pointer to get the
    data locally.
    Example:

    .. code-block::

        pointer_object.get()

Pointers are being generated for most types of objects in the data science scene, but what you can
do on them is not the pointers job, see the lib module for more details. One can see the pointer
as a proxy to the actual data, the filtering and the security being applied where the data is being
held.

Example:

.. code-block::

    # creating the data holder domain
    domain_1 = Domain(name="Data holder domain")

    # creating dummy data
    tensor = th.tensor([1, 2, 3])

    # creating the data holder client
    domain_1_client = domain_1.get_root_client()

    # sending the data to the client and receiving a pointer of that data.
    data_ptr_domain_1 = tensor.send(domain_1_client)

    # creating the data user domain
    domain_2 = Domain(name="Data user domain")

    # creating a request to access the data
    data_ptr_domain_1.request(
        request_name="My Request", reason="I'd lke to see this pointer"
    )

    # getting the remote id of the object
    requested_object = data_ptr_domain_1.id_at_location

    # getting the request id
    message_request_id = domain_1_client.request_queue.get_request_id_from_object_id(
        object_id=requested_object
    )

    # the data holder accepts the request
    domain_1.requests[0].owner_client_if_available = domain_1_client
    domain_1.requests[0].accept()

    # the data user checks if the data holder approved his request
    response = data_ptr_domain_1.check_access(node=domain_2, request_id=message_request_id)

"""
# stdlib
from typing import Any
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
import syft as sy

# syft relative
from ...decorators.syft_decorator_impl import syft_decorator
from ...proto.core.pointer.pointer_pb2 import Pointer as Pointer_PB
from ..common.pointer import AbstractPointer
from ..common.serde.deserialize import _deserialize
from ..common.uid import UID
from ..io.address import Address
from ..node.abstract.node import AbstractNode
from ..node.common.action.garbage_collect_object_action import (
    GarbageCollectObjectAction,
)
from ..node.common.action.get_object_action import GetObjectAction
from ..store.storeable_object import StorableObject


# TODO: Fix the Client, Address, Location confusion
class Pointer(AbstractPointer):
    """
    Pointer is the handler when interacting with remote data.

    Automatically generated subclasses of Pointer need to be able to look up
    the path and name of the object type they point to as a part of serde. For more
    information on how subclasses are automatically generated, please check the ast
    module.

    :param location: The location where the data is being held.
    :type location: Address
    :param id_at_location: The UID of the object on the remote location.
    :type id_at_location: UID
    """

    path_and_name: str

    def __init__(
        self,
        client: Any,
        id_at_location: Optional[UID] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        if id_at_location is None:
            id_at_location = UID()

        if tags is None:
            tags = []

        self.client = client
        self.id_at_location = id_at_location
        self.tags = tags
        self.description = description
        self.gc_enabled = True

    def get(
        self,
    ) -> StorableObject:
        """Method to download a remote object from a pointer object if you have the right
        permissions.

        :return: returns the downloaded data
        :rtype: StorableObject
        """
        obj_msg = GetObjectAction(
            obj_id=self.id_at_location,
            address=self.client.address,
            reply_to=self.client.address,
        )

        response = self.client.send_immediate_msg_with_reply(msg=obj_msg)

        return response.obj

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Pointer_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: Pointer_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return Pointer_PB(
            points_to_object_with_path=self.path_and_name,
            pointer_name=type(self).__name__,
            id_at_location=self.id_at_location.serialize(),
            location=self.client.address.serialize(),
            tags=self.tags,
            description=self.description,
        )

    @staticmethod
    def _proto2object(proto: Pointer_PB) -> "Pointer":
        """Creates a Pointer from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of Pointer
        :rtype: Pointer

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """
        # TODO: we need _proto2object to include a reference to the node doing the
        # deserialization so that we can convert location into a client object. At present
        # it is an address object which will cause things to break later.

        points_to_type = sy.lib_ast(
            proto.points_to_object_with_path, return_callable=True
        )
        pointer_type = getattr(points_to_type, proto.pointer_name)
        # WARNING: This is sending a serialized Address back to the constructor
        # which currently depends on a Client for send_immediate_msg_with_reply
        return pointer_type(
            id_at_location=_deserialize(blob=proto.id_at_location),
            client=_deserialize(blob=proto.location),
            tags=proto.tags,
            description=proto.description,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """Return the type of protobuf object which stores a class of this type

        As a part of serialization and deserialization, we need the ability to
        lookup the protobuf object type directly from the object type. This
        static method allows us to do this.

        Importantly, this method is also used to create the reverse lookup ability within
        the metaclass of Serializable. In the metaclass, it calls this method and then
        it takes whatever type is returned from this method and adds an attribute to it
        with the type of this class attached to it. See the MetaSerializable class for details.

        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType

        """

        return Pointer_PB

    def request(
        self,
        request_name: str = "",
        name: str = "",
        reason: str = "",
    ) -> None:
        """Method that requests access to the data on which the pointer points to.

        Example:

        .. code-block::

            # data holder domain
            domain_1 = Domain(name="Data holder")

            # data
            tensor = th.tensor([1, 2, 3])

            # generating the client for the domain
            domain_1_client = domain_1.get_root_client()

            # sending the data and receiving a pointer
            data_ptr_domain_1 = tensor.send(domain_1_client)

            # requesting access to the pointer
            data_ptr_domain_1.request(request_name="My Request", reason="Research project.")

        :param request_name: The title of the request that the data owner is going to see.
        :type request_name: str
        :param reason: The description of the request. This is the reason why you want to have
        access to the data.
        :type reason: str

        .. note::
            This method should be usen when the remote data associated with the pointer wants to be
            downloaded locally (or use .get() on the pointer).
        """
        # syft relative
        from ..node.domain.service import RequestMessage

        # optional kwarg to set name
        request_name = request_name
        if len(name) > 0:
            request_name = name

        msg = RequestMessage(
            request_name=request_name,
            request_description=reason,
            address=self.client.address,
            owner_address=self.client.address,
            object_id=self.id_at_location,
            requester_verify_key=self.client.verify_key,
        )

        print("Request Message Id:" + str(msg.id))

        self.client.send_immediate_msg_without_reply(msg=msg)

    def check_access(self, node: AbstractNode, request_id: UID) -> any:  # type: ignore
        """Method that checks the status of an already made request. There are three possible
        outcomes when requesting access:
            1. RequestStatus.Accepted - your request has been approved, you can not .get() your
            data.
            2. RequestStatus.Pending - your request has not been reviewed yet.
            3. RequestStatus.Rejected - your request has been rejected.

        :param node: The node that queries the request status.
        :type node: AbstractNode
        :param request_id: The request on which you are querying the status.
        :type request_id: UID
        """
        # syft relative
        from ..node.domain.service import RequestAnswerMessage

        msg = RequestAnswerMessage(
            request_id=request_id, address=self.client.address, reply_to=node.address
        )
        response = self.client.send_immediate_msg_with_reply(msg=msg)

        return response.status

    def __del__(self) -> None:
        _client_type = type(self.client)
        if (_client_type == Address) or issubclass(_client_type, AbstractNode):
            # it is a serialized pointer that we receive from another client do nothing
            return

        if self.gc_enabled:
            # Create the delete message
            msg = GarbageCollectObjectAction(
                obj_id=self.id_at_location, address=self.client.address
            )

            # Send the message
            self.client.send_eventual_msg_without_reply(msg=msg)
