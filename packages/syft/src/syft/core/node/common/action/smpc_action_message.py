# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# relative
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ....io.address import Address


@serializable(recursive_serde=True)
class SMPCActionMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = [
        "name_action",
        "self_id",
        "args_id",
        "kwargs_id",
        "kwargs",
        "id_at_location",
        "address",
        "id",
    ]

    def __init__(
        self,
        name_action: str,
        self_id: UID,
        args_id: List[UID],
        kwargs_id: Dict[str, UID],
        result_id: UID,
        address: Address,
        kwargs: Optional[Dict[str, Any]] = None,
        ranks_to_run_action: Optional[List[int]] = None,
        msg_id: Optional[UID] = None,
    ) -> None:
        self.name_action = name_action
        self.self_id = self_id
        self.args_id = args_id
        self.kwargs_id = kwargs_id
        if kwargs is None:
            self.kwargs = {}
        else:
            self.kwargs = kwargs
        self.id_at_location = result_id
        self.ranks_to_run_action = ranks_to_run_action if ranks_to_run_action else []
        super().__init__(address=address, msg_id=msg_id)

    @staticmethod
    def filter_actions_after_rank(
        rank: int, actions: List[SMPCActionMessage]
    ) -> List[SMPCActionMessage]:
        """
        Filter the actions depending on the rank of each party

        Arguments:
            rank (int): the rank of the party
            actions (List[SMPCActionMessage]):

        """
        res_actions = []
        for action in actions:
            if rank in action.ranks_to_run_action:
                res_actions.append(action)

        return res_actions

    def __str__(self) -> str:
        res = f"SMPCAction: {self.name_action}, "
        res = f"{res}Self ID: {self.self_id}, "
        res = f"{res}Args IDs: {self.args_id}, "
        res = f"{res}Kwargs IDs: {self.kwargs_id}, "
        res = f"{res}Kwargs : {self.kwargs}, "
        res = f"{res}Result ID: {self.id_at_location}, "
        res = f"{res}Ranks to run action: {self.ranks_to_run_action}"
        return res

    __repr__ = __str__
