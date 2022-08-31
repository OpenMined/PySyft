# stdlib
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
import warnings

# third party
from nacl.signing import VerifyKey
import requests

# relative
from ...logger import debug
from ...logger import error
from ...logger import warning
from ..common.pointer import AbstractPointer
from ..common.serde import _serialize
from ..common.serde.deserialize import _deserialize
from ..common.serde.serializable import serializable
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
from ..store.storeable_object import StorableObject


# TODO: Fix the Client, Address, Location confusion
@serializable(recursive_serde=True)
class Pointer(AbstractPointer):
    __attr_allowlist__ = [
        "id_at_location",
        "client",
        "tags",
        "description",
        "object_type",
        "_exhausted",
        "gc_enabled",
    ]
    __serde_overrides__: Dict[str, Sequence[Callable]] = {
        "client": (
            lambda client: _serialize(client.address, to_bytes=True),
            lambda blob: _deserialize(blob, from_bytes=True),
        )
    }

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
        self.gc_enabled = True

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
        return f"<{self.__name__} -> {self.client.name}:{self.id_at_location.no_dash}>"

    def _get(
        self, delete_obj: bool = True, verbose: bool = False, proxy_only: bool = False
    ) -> StorableObject:
        """Method to download a remote object from a pointer object if you have the right
        permissions.

        :return: returns the downloaded data
        :rtype: StorableObject
        """

        # relative
        from ...core.node.common.client import GET_OBJECT_TIMEOUT

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

        obj = self.client.send_immediate_msg_with_reply(
            msg=obj_msg, timeout=GET_OBJECT_TIMEOUT
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

    def publish(self, sigma: float = 1.5) -> Any:

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
        )

        self.client.send_immediate_msg_without_reply(msg=obj_msg)
        # create pointer which will point to float result

        ptr = self.client.lib_ast.query("syft.lib.python.Any").pointer_type(
            client=self.client
        )
        ptr.id_at_location = id_at_location
        ptr._pointable = True

        # return pointer
        return ptr

    def get(
        self,
        request_block: bool = False,
        timeout_secs: int = 600,
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
                delete_obj=delete_obj, verbose=verbose, proxy_only=proxy_only
            )
        else:
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
                result = self._get(delete_obj=delete_obj, verbose=verbose)
            else:
                return None

        if result is not None and delete_obj:
            self.gc_enabled = False
            self._exhausted = True

        return result

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

        if self.gc_enabled:
            # this is not being used in the node currenetly
            self.client.gc.apply(self)
