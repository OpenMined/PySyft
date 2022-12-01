"""A Pointer is the main handler when interacting with remote data.
A Pointer object represents an API for interacting with data (of any type)
at a specific location. The pointer should never be instantiated, only subclassed.

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

    pointer_object.request(name = "Request name", reason = "Request reason")

2. The data owner has to approve the request (check the domain node docs).
3. The data user checks if the request has been approved (check the domain node docs).
4. After the request has been approved, the data user can call .get() on the pointer to get the
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
        name="My Request", reason="I'd lke to see this pointer"
    )

    # getting the remote id of the object
    requested_object = data_ptr_domain_1.id_at_location

    # getting the request id
    message_request_id = domain_1_client.requests.get_request_id_from_object_id(
        object_id=requested_object
    )

    # the data holder accepts the request
    domain_1.requests[0].owner_client_if_available = domain_1_client
    domain_1.requests[0].accept()

    # the data user checks if the data holder approved his request
    response = data_ptr_domain_1.check_access(node=domain_2, request_id=message_request_id)

"""
# stdlib
import time
from typing import Any
from typing import List
from typing import Optional
import warnings

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey
import requests

# relative
from ...logger import debug
from ...logger import error
from ...logger import warning
from ...proto.core.pointer.pointer_pb2 import Pointer as Pointer_PB
from ..common.pointer import AbstractPointer
from ..common.serde.deserialize import _deserialize
from ..common.serde.serializable import serializable
from ..common.serde.serialize import _serialize as serialize
from ..common.uid import UID
from ..io.address import Address
from ..node.abstract.node import AbstractNode
from ..node.common.action.get_object_action import GetObjectAction
from ..node.common.exceptions import AuthorizationError
from ..node.common.exceptions import DatasetDownloadError
from ..node.common.node_service.get_repr.get_repr_service import GetReprMessage
from ..node.common.node_service.object_search_permission_update.obj_search_permission_messages import (
    ObjectSearchPermissionUpdateMessage,
)
from ..node.enums import PointerStatus
from ..store.storeable_object import StorableObject


# TODO: Fix the Client, Address, Location confusion
@serializable()
class Pointer(AbstractPointer):
    """
    The pointer is the handler when interacting with remote data.

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
    _pointable: bool = False
    __name__ = "DefaultPointerDunderNamePleaseChangeMe"

    def __init__(
        self,
        client: Any,
        id_at_location: Optional[UID] = None,
        object_type: str = "",
        tags: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        super().__init__(
            client=client,
            id_at_location=id_at_location,
            tags=tags,
            description=description,
        )
        self.object_type = object_type
        # _exhausted becomes True in get() call
        # when delete_obj is True and network call
        # has already been made
        self._exhausted = False

    @property
    def block(self) -> AbstractPointer:
        while not self.exists:
            time.sleep(0.1)
        return self

    def block_with_timeout(self, secs: int, secs_per_poll: int = 1) -> AbstractPointer:

        total_secs = secs

        while not self.exists and secs > 0:
            time.sleep(secs_per_poll)
            secs -= secs_per_poll

        if not self.exists:
            raise Exception(
                f"Object with id {self.id_at_location} still doesn't exist after {total_secs} second timeout."
            )

        return self

    @property
    def exists(self) -> bool:
        """Sometimes pointers can point to objects which either have not yet
        been created or were created but have now been destroyed. This method
        asks a remote node whether the object this pointer is pointing to can be
        found in the database."""
        return self.client.obj_exists(obj_id=self.id_at_location)

    def __repr__(self) -> str:
        if hasattr(self.client, "obj_exists"):
            _ptr_status = (
                PointerStatus.READY.value
                if self.exists
                else PointerStatus.PROCESSING.value
            )
            return f"<{self.__name__} -> {self.client.name}:{self.id_at_location.no_dash}, status={_ptr_status}>"
        else:
            return (
                f"<{self.__name__} -> {self.client.name}:{self.id_at_location.no_dash}>"
            )

    def _get(
        self,
        delete_obj: bool = True,
        verbose: bool = False,
        proxy_only: bool = False,
        timeout_secs: Optional[int] = None,
    ) -> StorableObject:
        """Method to download a remote object from a pointer object if you have the right
        permissions.

        :return: returns the downloaded data
        :rtype: StorableObject
        """

        # relative
        from ...core.node.common.client import GET_OBJECT_TIMEOUT
        from ..node.common.action.exception_action import UnknownPrivateException

        debug(
            f"> GetObjectAction for id_at_location={self.id_at_location} "
            + f"with delete_obj={delete_obj}"
        )
        obj_msg = GetObjectAction(
            id_at_location=self.id_at_location,
            address=self.client.address,
            reply_to=self.client.address,
            delete_obj=delete_obj,
        )

        obj: Any = None
        is_processing_pointer = self.client.processing_pointers.get(
            self.id_at_location, False
        )

        start_time = time.time()
        future_time = (
            float(timeout_secs if timeout_secs is not None else GET_OBJECT_TIMEOUT)
            + start_time
        )

        # If pointer is one of the processing pointers and didn't timeout keep trying
        while is_processing_pointer and future_time > time.time():
            try:
                obj = self.client.send_immediate_msg_with_reply(
                    msg=obj_msg, timeout=timeout_secs, verbose=True
                )

                # If we reached here it's because we didn't have any failure,
                # so we were able to retrieve the pointer successfully.
                # So it isn't a processing pointer anymore and we can exit the while loop.
                # without wait the timeout
                is_processing_pointer = False
            except UnknownPrivateException:
                time.sleep(0.5)
                pass

        # If pointer was there, then we remove it from the processing_pointer list
        self.client.processing_pointers.pop(self.id_at_location, None)

        # if we didn't get the object try one last time
        if not obj:
            obj = self.client.send_immediate_msg_with_reply(
                msg=obj_msg, timeout=timeout_secs
            )

        if not proxy_only and obj.obj.is_proxy:
            presigned_url_path = obj.obj._data.url
            presigned_url = self.client.url_from_path(presigned_url_path)
            response = requests.get(presigned_url)

            if not response.ok:
                error_msg = (
                    f"\nFailed to get object {self.id_at_location} from store\n."
                    + f"Status Code: {response.status_code} {response.reason}"
                )
                raise DatasetDownloadError(error_msg)
            obj = _deserialize(response.content, from_bytes=True)
        else:
            if proxy_only:
                print(
                    "**Warning**: Proxy data class does not exist for this object. Fetching the real data."
                )
            obj = obj.data

        if delete_obj:
            # relative
            from ..node.common.node_service.generic_payload.syft_message import (
                NewSyftMessage,
            )
            from ..node.common.node_service.object_delete.object_delete_message import (
                ObjectDeleteMessage,
            )

            # TODO: Fix circular import
            # This deletes the data from both database and blob store
            obj_del_msg: NewSyftMessage = ObjectDeleteMessage(
                address=self.client.address,
                reply_to=self.client.address,
                kwargs={
                    "id_at_location": self.id_at_location.to_string(),
                },
            ).sign(signing_key=self.client.signing_key)

            try:
                self.client.send_immediate_msg_with_reply(msg=obj_del_msg)
            except AuthorizationError:
                print("**Warning:** You don't have delete permissions to the object.")

        if self.is_enum:
            enum_class = self.client.lib_ast.query(self.path_and_name).object_ref
            return enum_class(obj)

        return obj

    def get_copy(
        self,
        request_block: bool = False,
        timeout_secs: int = 20,
        reason: str = "",
        verbose: bool = False,
    ) -> Optional[StorableObject]:
        """Method to download a remote object from a pointer object if you have the right
        permissions. Optionally can block while waiting for approval.

        :return: returns the downloaded data
        :rtype: Optional[StorableObject]
        """
        return self.get(
            request_block=request_block,
            timeout_secs=timeout_secs,
            reason=reason,
            delete_obj=False,
            verbose=verbose,
        )

    def print(self) -> "Pointer":
        obj = None
        try:
            obj_msg = GetReprMessage(
                id_at_location=self.id_at_location,
                address=self.client.address,
                reply_to=self.client.address,
            )

            obj = self.client.send_immediate_msg_with_reply(msg=obj_msg).repr
        except Exception as e:
            if "You do not have permission to .get()" in str(
                e
            ) or "UnknownPrivateException" in str(e):

                # relative
                from ..node.common.node_service.request_receiver.request_receiver_messages import (
                    RequestStatus,
                )

                response_status = self.request(
                    reason="Calling remote print",
                    block=True,
                    timeout_secs=3,
                )
                if (
                    response_status is not None
                    and response_status == RequestStatus.Accepted
                ):
                    return self.print()

        # TODO: Create a remote print interface for objects which displays them in a
        # nice way, we could also even buffer this between chained ops until we return
        # so that we can print once and display a nice list of data and ops
        # issue: https://github.com/OpenMined/PySyft/issues/5167
        if obj is not None:
            print(obj)
        else:
            print(f"No permission to print() {self}")

        return self

    def publish(self, sigma: float = 1.5, private: bool = True) -> Any:

        # relative
        from ..node.common.node_service.publish.publish_service import (
            PublishScalarsAction,
        )

        id_at_location = UID()

        obj_msg = PublishScalarsAction(
            id_at_location=id_at_location,
            address=self.client.address,
            publish_ids_at_location=[self.id_at_location],
            sigma=sigma,
            private=private,
        )

        self.client.send_immediate_msg_without_reply(msg=obj_msg)
        # create pointer which will point to float result

        if not hasattr(self, "PUBLISH_POINTER_TYPE"):
            raise TypeError(
                f"Publish operation cannot be performed on pointer type: {self.__name__}"
            )

        ptr = self.client.lib_ast.query(self.PUBLISH_POINTER_TYPE).pointer_type(  # type: ignore
            client=self.client
        )
        ptr.id_at_location = id_at_location
        ptr._pointable = True
        ptr.client.processing_pointers[ptr.id_at_location] = True
        # return pointer
        return ptr

    def get(
        self,
        request_block: bool = False,
        timeout_secs: Optional[int] = None,
        reason: str = "",
        delete_obj: bool = True,
        verbose: bool = False,
        proxy_only: bool = False,
    ) -> Optional[StorableObject]:
        """Method to download a remote object from a pointer object if you have the right
        permissions. Optionally can block while waiting for approval.

        :return: returns the downloaded data
        :rtype: Optional[StorableObject]
        """
        if proxy_only and delete_obj:
            delete_obj = False
            print("**Warning**: Fetching proxy_only will not delete the real object")

        # relative
        from ..node.common.node_service.request_receiver.request_receiver_messages import (
            RequestStatus,
        )

        if self._exhausted:
            raise ReferenceError(
                "Object has already been deleted. This pointer is exhausted"
            )

        if not request_block:
            result = self._get(
                delete_obj=delete_obj,
                verbose=verbose,
                proxy_only=proxy_only,
                timeout_secs=timeout_secs,
            )
        else:
            if timeout_secs is None:
                timeout_secs = 600  # old default
            response_status = self.request(
                reason=reason,
                block=True,
                timeout_secs=timeout_secs,
                verbose=verbose,
            )
            if (
                response_status is not None
                and response_status == RequestStatus.Accepted
            ):
                result = self._get(
                    delete_obj=delete_obj, verbose=verbose, timeout_secs=timeout_secs
                )
            else:
                return None

        if result is not None and delete_obj:
            self.gc_enabled = False
            self._exhausted = True

        return result

    def _object2proto(self) -> Pointer_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: Pointer_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        return Pointer_PB(
            points_to_object_with_path=self.path_and_name,
            pointer_name=type(self).__name__,
            id_at_location=serialize(self.id_at_location),
            location=serialize(self.client.address),
            tags=self.tags,
            description=self.description,
            object_type=self.object_type,
            attribute_name=getattr(self, "attribute_name", ""),
            public_shape=serialize(getattr(self, "public_shape", None), to_bytes=True),
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
        # relative
        from ...lib import lib_ast

        # TODO: we need _proto2object to include a reference to the node doing the
        # deserialization so that we can convert location into a client object. At present
        # it is an address object which will cause things to break later.
        points_to_type = lib_ast.query(proto.points_to_object_with_path)
        pointer_type = getattr(points_to_type, proto.pointer_name)

        # WARNING: This is sending a serialized Address back to the constructor
        # which currently depends on a Client for send_immediate_msg_with_reply

        out = pointer_type(
            id_at_location=_deserialize(blob=proto.id_at_location),
            client=_deserialize(blob=proto.location),
            tags=proto.tags,
            description=proto.description,
            object_type=proto.object_type,
        )

        out.public_shape = _deserialize(proto.public_shape, from_bytes=True)
        return out

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
        reason: str = "",
        block: bool = False,
        timeout_secs: Optional[int] = None,
        verbose: bool = False,
    ) -> Any:
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
            data_ptr_domain_1.request(name="My Request", reason="Research project.")

        :param name: The title of the request that the data owner is going to see.
        :type name: str
        :param reason: The description of the request. This is the reason why you want to have
            access to the data.
        :type reason: str

        .. note::
            This method should be used when the remote data associated with the pointer wants to be
            downloaded locally (or use .get() on the pointer).
        """

        # relative
        from ..node.common.node_service.request_receiver.request_receiver_messages import (
            RequestMessage,
        )

        # if you request non-blocking you don't need a timeout
        # if you request blocking you need a timeout, so lets set a default on here
        # a timeout of 0 would be a way to say don't block my local notebook but if the
        # duet partner has a rule configured it will get executed first before the
        # request would time out
        if timeout_secs is None and block is False:
            timeout_secs = -1  # forever

        msg = RequestMessage(
            request_description=reason,
            address=self.client.address,
            owner_address=self.client.address,
            object_id=self.id_at_location,
            object_type=self.object_type,
            requester_verify_key=self.client.verify_key,
            timeout_secs=timeout_secs,
        )

        self.client.send_immediate_msg_without_reply(msg=msg)

        # wait long enough for it to arrive and trigger a handler
        time.sleep(0.1)

        if not block:
            return None
        else:
            if timeout_secs is None:
                timeout_secs = 30  # default if not explicitly set

            # relative
            from ..node.common.node_service.request_answer.request_answer_service import (
                RequestAnswerMessage,
            )
            from ..node.common.node_service.request_receiver.request_receiver_messages import (
                RequestStatus,
            )

            output_string = "> Waiting for Blocking Request: "
            output_string += f"  {self.id_at_location}"
            if len(reason) > 0:
                output_string += f": {reason}"
            if len(output_string) > 0 and output_string[-1] != ".":
                output_string += "."
            debug(output_string)
            status = None
            start = time.time()

            last_check: float = 0.0
            while True:
                now = time.time()
                try:
                    # won't run on the first pass because status is None which allows
                    # for remote request handlers to auto respond before timeout
                    if now - start > timeout_secs:
                        log = (
                            f"\n> Blocking Request Timeout after {timeout_secs} seconds"
                        )
                        debug(log)
                        return status

                    # only check once every second
                    if now - last_check > 1:
                        last_check = now
                        debug(f"> Sending another Request Message {now - start}")
                        status_msg = RequestAnswerMessage(
                            request_id=msg.id,
                            address=self.client.address,
                            reply_to=self.client.address,
                        )
                        response = self.client.send_immediate_msg_with_reply(
                            msg=status_msg
                        )
                        status = response.status
                        if response.status == RequestStatus.Pending:
                            time.sleep(0.1)
                            continue
                        else:
                            # accepted or rejected lets exit
                            status_text = "REJECTED"
                            if status == RequestStatus.Accepted:
                                status_text = "ACCEPTED"
                            log = f" {status_text}"
                            debug(log)
                            return status
                except Exception as e:
                    error(f"Exception while running blocking request. {e}")
                    # escape the while loop
                    return status

    @property
    def searchable(self) -> bool:
        msg = "`searchable` is deprecated please use `pointable` in future"
        warning(msg, print=True)
        warnings.warn(
            msg,
            DeprecationWarning,
        )
        return self._pointable

    @searchable.setter
    def searchable(self, value: bool) -> None:
        msg = "`searchable` is deprecated please use `pointable` in future"
        warning(msg, print=True)
        warnings.warn(
            msg,
            DeprecationWarning,
        )
        self.pointable = value

    @property
    def pointable(self) -> bool:
        return self._pointable

    @pointable.setter
    def pointable(self, value: bool) -> None:
        if value != self._pointable:
            self.update_searchability(not self._pointable)

    def update_searchability(
        self,
        pointable: bool = True,
        target_verify_key: Optional[VerifyKey] = None,
        searchable: Optional[bool] = None,
    ) -> None:
        """Make the object pointed at pointable or not for other people. If
        target_verify_key is not specified, the searchability for the VERIFYALL group
        will be toggled.

        :param pointable: If the target object should be made pointable or not.
        :type target_verify_key: bool
        :param target_verify_key: The verify_key of the client to which we want to give
               search permission.
        :type target_verify_key: Optional[VerifyKey]
        """

        if searchable is not None:
            warn_msg = "`searchable` is deprecated please use `pointable` in future"
            warning(warn_msg, print=True)
            warnings.warn(
                warn_msg,
                DeprecationWarning,
            )
            pointable = searchable

        self._pointable = pointable
        msg = ObjectSearchPermissionUpdateMessage(
            add_instead_of_remove=pointable,
            target_verify_key=target_verify_key,
            target_object_id=self.id_at_location,
            address=self.client.address,
        )
        self.client.send_immediate_msg_without_reply(msg=msg)

    def check_access(self, node: AbstractNode, request_id: UID) -> any:  # type: ignore
        """Method that checks the status of an already made request. There are three
        possible outcomes when requesting access:

        1. RequestStatus.Accepted - your request has been approved, you can not .get() your data.
        2. RequestStatus.Pending - your request has not been reviewed yet.
        3. RequestStatus.Rejected - your request has been rejected.

        :param node: The node that queries the request status.
        :type node: AbstractNode
        :param request_id: The request on which you are querying the status.
        :type request_id: UID
        """

        # relative
        from ..node.common.node_service.request_answer.request_answer_messages import (
            RequestAnswerMessage,
        )

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

        # Check/Remove it if this pointer is still in processing_pointers dict
        self.client.processing_pointers.pop(self.id_at_location, None)

        # if self.gc_enabled:
        #     # this is not being used in the node currenetly
        #     self.client.gc.apply(self)
