# stdlib
from typing import Optional

# relative
from ..abstract_node import AbstractNode
from ..node.credentials import SyftCredentials
from ..serde.serializable import serializable
from ..types.syft_object import Context
from .response import SyftWarning
from .user.user_roles import ServiceRole


class WarningContext(
    Context,
):
    node: Optional[AbstractNode]
    credentials: Optional[SyftCredentials]
    role: ServiceRole


@serializable()
class APIEndpointWarning(SyftWarning):
    confirmation: bool

    def message_from(self, context: Optional[WarningContext]):
        raise NotImplementedError


@serializable()
class CRUDWarning(APIEndpointWarning):
    message: Optional[str]

    def message_from(self, context: Optional[WarningContext] = None):
        if self._confirmation:
            pass

        self.message = ""
        if context is not None:
            node = context.node
            node_side_type = node.node_side_type
            message = (
                f"You're performing an operation on "
                f"{node_side_type.value} side {node.node_type.value}."
            )
            self.message = message
