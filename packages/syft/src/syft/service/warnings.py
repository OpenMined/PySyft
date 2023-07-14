# stdlib
from typing import Optional

# third party
from IPython.display import display
from rich.prompt import Confirm

# relative
from ..abstract_node import AbstractNode
from ..node.credentials import SyftCredentials
from ..serde.serializable import serializable
from ..types.syft_object import Context
from ..util.experimental_flags import flags
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

    def show(self):
        display(self)
        if self.confirmation and flags.PROMPT_ENABLED:
            allowed = Confirm.ask("Would you like to proceed?")
            if not allowed:
                display("Aborted !!")
                return False
        return True


@serializable()
class CRUDWarning(APIEndpointWarning):
    message: Optional[str]

    def message_from(self, context: Optional[WarningContext] = None):
        self.message = ""
        if context is not None:
            node = context.node
            node_side_type = node.node_side_type
            node_type = node.node_type.value
            message = (
                "You're performing an operation on "
                f"{node_side_type.value} side {node_type}."
            )
            self.message = message
