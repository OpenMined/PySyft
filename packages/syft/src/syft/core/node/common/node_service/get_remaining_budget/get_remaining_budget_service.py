# stdlib
from typing import List as TypeList
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from ......logger import traceback_and_raise  # type: ignore
from ..auth import service_auth

# syft relative
from .....adp.publish import get_remaining_budget  # type: ignore
from ....abstract.node import AbstractNode
from ..node_service import ImmediateNodeServiceWithReply
from .get_remaining_budget_messages import GetRemainingBudgetAction  # type: ignore
from .get_remaining_budget_messages import GetRemainingBudgetMessage


class GetRemainingBudgetService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(root_only=False)
    def process(
        node: AbstractNode, 
        msg: GetRemainingBudgetMessage, 
        verify_key: Optional[VerifyKey] = None,
    ) -> GetRemainingBudgetAction:

        if verify_key is None:
            traceback_and_raise(
                "Can't process GetRemainingBudgetService with no verification key."
            )

        try:
            result = get_remaining_budget(node.acc, verify_key)
            return GetRemainingBudgetAction(budget=result, address=msg.reply_to)
        except Exception as e:
            log = (
                f"Unable to get remaining budget for {verify_key}. " + f"Possible dangling Pointer. {e}"
            )
            traceback_and_raise(Exception(log))

    @staticmethod
    def message_handler_types() -> TypeList[Type[GetRemainingBudgetMessage]]:
        return [GetRemainingBudgetMessage]
