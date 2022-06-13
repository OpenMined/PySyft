# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from ..... import lib
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply


@serializable(recursive_serde=True)
class BeaverPrimitiveAction(ImmediateActionWithoutReply):
    """
    When executing a BeaverPrimitiveAction, a :class:`Node` will run a method defined
    by the action's path attribute and keep the returned
    value in its store.

    Attributes:
         path: the dotted path to the method to call
         args: args to pass to the function. They should be pointers to objects
            located on the :class:`Node` that will execute the action.
         kwargs: kwargs to pass to the function. They should be pointers to objects
            located on the :class:`Node` that will execute the action.
    """

    __attr_allowlist__ = [
        "path",
        "args",
        "kwargs",
        "id_at_location",
        "address",
        "msg_id",
        "_id",
    ]

    def __init__(
        self,
        path: str,
        args: List[Any],
        kwargs: Dict[Any, Any],
        id_at_location: UID,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        self.path = path
        self.args = args
        self.kwargs = kwargs
        self.id_at_location = id_at_location
        # logging needs .path to exist before calling
        # this which is why i've put this super().__init__ down here
        super().__init__(address=address, msg_id=msg_id)

    @property
    def pprint(self) -> str:
        return f"BeaverPrimitiveAction({self.path})"

    def __repr__(self) -> str:
        arg_names = ",".join([a.__class__.__name__ for a in self.args])
        kwargs_names = ",".join(
            [f"{k}={v.__class__.__name__}" for k, v in self.kwargs.items()]
        )
        return f"BeaverPrimitiveAction ({arg_names}, {kwargs_names})"

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        # relative
        from ....tensor.smpc.share_tensor import populate_store

        (
            upcasted_args,
            upcasted_kwargs,
        ) = lib.python.util.upcast_args_and_kwargs(self.args, self.kwargs)

        result = populate_store(*upcasted_args, **upcasted_kwargs)  # type: ignore

        result_read_permissions = {
            node.verify_key: node.id,
            verify_key: None,  # we dont have the passed in sender's UID
        }
        result_write_permissions = {
            node.verify_key: node.id,
            verify_key: None,  # we dont have the passed in sender's UID
        }

        if not isinstance(result, StorableObject):
            store_obj = StorableObject(
                id=self.id_at_location,
                data=result,
                read_permissions=result_read_permissions,
                write_permissions=result_write_permissions,
            )

        node.store[self.id_at_location] = store_obj
