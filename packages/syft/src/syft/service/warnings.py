# stdlib
from typing import Any
from typing import Optional

# third party
from IPython.display import display
from rich.prompt import Confirm
from typing_extensions import Self

# relative
from ..abstract_node import AbstractNode
from ..abstract_node import NodeSideType
from ..node.credentials import SyftCredentials
from ..serde.serializable import serializable
from ..types.base import SyftBaseModel
from ..types.syft_object import Context
from .user.user_roles import ServiceRole


class WarningContext(
    Context,
):
    node: Optional[AbstractNode]
    credentials: Optional[SyftCredentials]
    role: ServiceRole


@serializable()
class APIEndpointWarning(SyftBaseModel):
    confirmation: bool = False
    message: Optional[str] = None
    enabled: bool = True

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, APIEndpointWarning):
            return self.message == other.message and self._bool == other._bool
        return self._bool == other

    @property
    def _repr_html_class_(self) -> str:
        return "alert-warning"

    def __repr__(self) -> str:
        return f"SyftWarning: {self.message}"

    def __str__(self) -> str:
        return self.__repr__()

    def _repr_html_(self) -> str:
        return (
            f'<div class="{self._repr_html_class_}" style="padding:5px;">'
            + f"<strong>SyftWarning</strong>: {self.message}</div><br />"
        )

    def message_from(self, context: Optional[WarningContext]) -> Self:
        raise NotImplementedError

    def show(self) -> bool:
        if not self.enabled or not self.message:
            return True
        display(self)
        if self.confirmation:
            allowed = Confirm.ask("Would you like to proceed?")
            if not allowed:
                display("Aborted !!")
                return False
        return True


@serializable()
class CRUDWarning(APIEndpointWarning):
    def message_from(self, context: Optional[WarningContext] = None) -> Self:
        message = None
        confirmation = self.confirmation
        if context is not None:
            node = context.node
            if node is not None:
                node_side_type = node.node_side_type
                node_type = node.node_type
                _msg = (
                    "which could host datasets with private information."
                    if node_side_type.value == NodeSideType.HIGH_SIDE.value
                    else "which only hosts mock or synthetic data."
                )
                message = (
                    "You're performing an operation on "
                    f"{node_side_type.value} side {node_type.value}, {_msg}"
                )
                confirmation = node_side_type.value == NodeSideType.HIGH_SIDE.value

        return CRUDWarning(confirmation=confirmation, message=message)


@serializable()
class CRUDReminder(CRUDWarning):
    confirmation: bool = False

    def message_from(self, context: Optional[WarningContext] = None) -> Self:
        message = None
        confirmation = self.confirmation
        if context is not None:
            node = context.node
            if node is not None:
                node_side_type = node.node_side_type
                node_type = node.node_type
                _msg = (
                    "which could host datasets with private information."
                    if node_side_type.value == NodeSideType.HIGH_SIDE.value
                    else "which only hosts mock or synthetic data."
                )
                message = (
                    "You're performing an operation on "
                    f"{node_side_type.value} side {node_type.value}, {_msg}"
                )

        return CRUDReminder(confirmation=confirmation, message=message)


@serializable()
class LowSideCRUDWarning(APIEndpointWarning):
    def message_from(self, context: Optional[WarningContext] = None) -> Self:
        confirmation = self.confirmation
        message = None
        if context is not None:
            node = context.node
            if node is not None:
                node_side_type = node.node_side_type
                node_type = node.node_type
                if node_side_type.value == NodeSideType.LOW_SIDE.value:
                    message = (
                        "You're performing an operation on "
                        f"{node_side_type.value} side {node_type.value} "
                        "which only hosts mock or synthetic data."
                    )

        return LowSideCRUDWarning(confirmation=confirmation, message=message)


@serializable()
class HighSideCRUDWarning(APIEndpointWarning):
    def message_from(self, context: Optional[WarningContext] = None) -> Self:
        confirmation = self.confirmation
        message = None
        if context is not None:
            node = context.node
            if node is not None:
                node_side_type = node.node_side_type
                node_type = node.node_type
                if node_side_type.value == NodeSideType.HIGH_SIDE.value:
                    message = (
                        "You're performing an operation on "
                        f"{node_side_type.value} side {node_type.value} "
                        "which could host datasets with private information."
                    )

        return HighSideCRUDWarning(confirmation=confirmation, message=message)
