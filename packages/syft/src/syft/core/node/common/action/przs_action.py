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
from ....tensor.smpc.share_tensor import ShareTensor
from ....tensor.smpc.utils import ispointer
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply
from .greenlets_switch import retrieve_object


@serializable(recursive_serde=True)
class PRZSAction(ImmediateActionWithoutReply):
    """
    When executing a PRZSAction, a :class:`Node` will run a ShareTensor przs method
    and keep the returned value in its store.

    Attributes:
         args: args to pass to the function. They should be pointers to objects
            located on the :class:`Node` that will execute the action.
         kwargs: kwargs to pass to the function. They should be pointers to objects
            located on the :class:`Node` that will execute the action
        is_dp_tensor: Set to True, if the tensor chain contains a DP Tensor
    """

    __attr_allowlist__ = [
        "path",
        "args",
        "kwargs",
        "id_at_location",
        "is_dp_tensor",
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
        is_dp_tensor: bool,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        self.path = path
        self.args = args
        self.kwargs = kwargs
        self.id_at_location = id_at_location
        self.is_dp_tensor = is_dp_tensor
        # logging needs .path to exist before calling
        # this which is why i've put this super().__init__ down here
        super().__init__(address=address, msg_id=msg_id)

    @property
    def pprint(self) -> str:
        return f"PRZSAction({self.path})"

    def __repr__(self) -> str:
        arg_names = ",".join([a.__class__.__name__ for a in self.args])
        kwargs_names = ",".join(
            [f"{k}={v.__class__.__name__}" for k, v in self.kwargs.items()]
        )
        return f"PRZSAction ({arg_names}, {kwargs_names})"

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:

        (
            upcasted_args,
            upcasted_kwargs,
        ) = lib.python.util.upcast_args_and_kwargs(self.args, self.kwargs)

        if "value" in upcasted_kwargs and ispointer(upcasted_kwargs["value"]):
            upcasted_kwargs["value"] = retrieve_object(
                node, upcasted_kwargs["value"].id_at_location, self.path
            ).data

        if not self.is_dp_tensor:
            result = ShareTensor.generate_przs(*upcasted_args, **upcasted_kwargs)
        else:
            result = ShareTensor.generate_przs_on_dp_tensor(
                *upcasted_args, **upcasted_kwargs
            )

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
