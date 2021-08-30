# stdlib
from typing import Dict as TypeDict
from typing import List as TypeList
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from ......lib.python import List  # type: ignore
from ......logger import traceback_and_raise  # type: ignore
from .....adp.publish import publish  # type: ignore
from .....common.uid import UID  # type: ignore
from .....store.storeable_object import StorableObject  # type: ignore
from .....tensor.tensor import PassthroughTensor  # type: ignore
from ....abstract.node import AbstractNode  # type: ignore
from ..node_service import ImmediateNodeServiceWithoutReply  # type: ignore
from .publish_messages import PublishScalarsAction  # type: ignore


class PublishScalarsService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    def process(
        node: AbstractNode, msg: PublishScalarsAction, verify_key: VerifyKey
    ) -> None:

        # get scalar objects from store
        results = List()
        for publish_id in msg.publish_ids_at_location:
            print(publish_id)
            try:
                publish_object = node.store[publish_id]

                if isinstance(publish_object.data, PassthroughTensor):
                    result = publish_object.data.publish(
                        acc=node.acc, sigma=msg.sigma, user_key=verify_key
                    )
                else:
                    result = publish(
                        [publish_object.data], node.acc, msg.sigma, user_key=verify_key
                    )
                results.append(result)
            except Exception as e:
                log = (
                    f"Unable to Get Object with ID {publish_id} from store. "
                    + f"Possible dangling Pointer. {e}"
                )
                traceback_and_raise(Exception(log))

        # give the caller permission to download this
        read_permissions: TypeDict[VerifyKey, UID] = {verify_key: None}
        search_permissions: TypeDict[VerifyKey, Optional[UID]] = {verify_key: None}

        if len(results) == 1:
            results = results[0]

        storable = StorableObject(
            id=msg.id_at_location,
            data=results,
            description=f"Approved AutoDP Result: {msg.id_at_location}",
            read_permissions=read_permissions,
            search_permissions=search_permissions,
        )

        node.store[msg.id_at_location] = storable

    @staticmethod
    def message_handler_types() -> TypeList[Type[PublishScalarsAction]]:
        return [PublishScalarsAction]
