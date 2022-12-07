# DOs and Don's of this class:
# - Do NOT use absolute syft imports (i.e. import syft.core...) Use relative ones.
# - Do NOT put multiple imports on the same line (i.e. from <x> import a, b, c). Use separate lines
# - Do sort imports by length
# - Do group imports by where they come from

# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey
from typing_extensions import final

# relative
from ......logger import error
from ......util import obj2pointer_type
from ......util import traceback_and_raise
from .....common.group import VERIFYALL
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address
from .....pointer.pointer import Pointer
from .....store.proxy_dataset import ProxyDataset
from ....abstract.node import AbstractNode
from ..node_service import ImmediateNodeServiceWithReply


@serializable(recursive_serde=True)
@final
class ObjectSearchMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "reply_to", "obj_id"]

    def __init__(
        self,
        address: Address,
        reply_to: Address,
        obj_id: Optional[UID] = None,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        """By default this message just returns pointers to all the objects
        the sender is allowed to see. In the future we'll add support so that
        we can query for subsets."""

        # if you specify an object id then search will return a pointer to that
        self.obj_id = obj_id


@serializable(recursive_serde=True)
@final
class ObjectSearchReplyMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "results"]

    def __init__(
        self,
        results: List[Pointer],
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        """By default this message just returns pointers to all the objects
        the sender is allowed to see. In the future we'll add support so that
        we can query for subsets."""
        self.results = results


class ImmediateObjectSearchService(ImmediateNodeServiceWithReply):
    @staticmethod
    def process(
        node: AbstractNode,
        msg: ObjectSearchMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> ObjectSearchReplyMessage:
        results: List[Pointer] = list()

        if verify_key is None:
            traceback_and_raise(
                "Can't process an ImmediateObjectSearchService with no "
                "verification key."
            )

        try:
            # if no object is specified, return all objects
            # TODO change this to a get all keys and metadata method which does not
            # require pulling out all data from the database
            if msg.obj_id is None:
                objs = node.store.get_objects_of_type(obj_type=object)

            # if object id is specified - return just that object
            else:
                objs = [node.store.get(msg.obj_id, proxy_only=True)]

            for obj in objs:
                # if this tensor allows anyone to search for it, then one of its keys
                # has a VERIFYALL in it.
                contains_all_in_permissions = any(
                    key is VERIFYALL for key in obj.search_permissions.keys()
                )
                if (
                    verify_key in obj.search_permissions.keys()
                    or verify_key == node.root_verify_key
                    or contains_all_in_permissions
                ):
                    if obj.is_proxy:
                        proxy_obj: ProxyDataset = obj.data  # type: ignore
                        ptr_constructor = obj2pointer_type(
                            fqn=proxy_obj.data_fully_qualified_name
                        )

                        ptr = ptr_constructor(
                            client=node,
                            id_at_location=obj.id,
                            object_type=obj.object_type,
                            tags=obj.tags,
                            description=obj.description,
                            **proxy_obj.obj_public_kwargs,
                        )

                    else:
                        if hasattr(obj.data, "init_pointer"):
                            ptr_constructor = obj.data.init_pointer  # type: ignore
                        else:
                            ptr_constructor = obj2pointer_type(obj=obj.data)

                        ptr = ptr_constructor(
                            client=node,
                            id_at_location=obj.id,
                            object_type=obj.object_type,
                            tags=obj.tags,
                            description=obj.description,
                        )

                    results.append(ptr)
        except Exception as e:
            error(f"Error searching store. {e}")

        return ObjectSearchReplyMessage(address=msg.reply_to, results=results)

    @staticmethod
    def message_handler_types() -> List[Type[ObjectSearchMessage]]:
        return [ObjectSearchMessage]
