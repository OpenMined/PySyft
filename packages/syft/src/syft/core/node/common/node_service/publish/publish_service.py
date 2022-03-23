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
from .....adp.data_subject_ledger import DataSubjectLedger  # type: ignore
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

        print("PublishScalarsService:28")

        # get scalar objects from store
        results = List()
        for publish_id in msg.publish_ids_at_location:
            print("PublishScalarsService:33")
            print(publish_id)
            try:
                print(
                    "PublishScalarsService:36: TRY: publish_object = node.store.get(publish_id)"
                )
                publish_object = node.store.get(publish_id)
                print(
                    "PublishScalarsService:38: SUCCESS: publish_object = node.store.get(publish_id)"
                )
                if isinstance(publish_object.data, PassthroughTensor):
                    print(
                        "PublishScalarsService:40: TRY: publish_object.data.publish()"
                    )
                    try:
                        print("Trying to get DataSubjectLedger in PublishService")
                        ledger = DataSubjectLedger.get_or_create(
                            store=node.ledger_store, user_key=verify_key
                        )
                        if ledger is None:
                            raise Exception("Unable to get ledger so we cannot publish")
                    except Exception as e:
                        print(f"Failed to get a ledger. {e}")
                        raise e

                    print("Publishing!")
                    print("Publish_object.data: ")
                    print(type(publish_object))
                    print("Publish_object.data: ")
                    print(type(publish_object.data))
                    print("Publish_object.data.child: ")
                    print(type(publish_object.data.child))
                    print("Ledger: ")
                    print(ledger)

                    result = publish_object.data.child.publish(
                        node=node, ledger=ledger, sigma=msg.sigma
                    )
                    # result = publish_object.data.publish(
                    #     acc=node.acc, sigma=msg.sigma, user_key=verify_key
                    # )
                    print(
                        "PublishScalarsService:44: SUCCESS: publish_object.data.publish()"
                    )
                else:
                    print(
                        "PublishScalarsService:46: TRY: publish([publish_object.data])"
                    )
                    result = publish(
                        [publish_object.data], node.acc, msg.sigma, user_key=verify_key
                    )
                    print(
                        "PublishScalarsService:50: SUCCESS: publish([publish_object.data])"
                    )
                results.append(result)
            except Exception as e:
                print("PublishScalarsService:53: EXCEPTION - missing id")
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
        print("PublishScalarsService:67: storable = StorableObject(")
        print(type(results))
        storable = StorableObject(
            id=msg.id_at_location,
            data=results,
            description=f"Approved AutoDP Result: {msg.id_at_location}",
            read_permissions=read_permissions,
            search_permissions=search_permissions,
        )
        print(
            "PublishScalarsService:74:  msg.id_at_location == "
            + str(msg.id_at_location)
            + " obj_type:"
            + str(type(storable))
        )
        print(
            "PublishScalarsService:75: TRY: node.store[msg.id_at_location] = storable"
        )
        node.store[msg.id_at_location] = storable
        print(
            "PublishScalarsService:77: SUCCESS: node.store[msg.id_at_location] = storable"
        )

    @staticmethod
    def message_handler_types() -> TypeList[Type[PublishScalarsAction]]:
        return [PublishScalarsAction]
