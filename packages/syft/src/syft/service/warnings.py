# stdlib
from typing import Any
from typing import cast

# third party
from IPython.display import display
from rich.prompt import Confirm
from typing_extensions import Self

# relative
from ..abstract_server import AbstractServer
from ..abstract_server import ServerSideType
from ..abstract_server import ServerType
from ..serde.serializable import serializable
from ..server.credentials import SyftCredentials
from ..types.base import SyftBaseModel
from ..types.syft_object import Context
from .user.user_roles import ServiceRole


class WarningContext(
    Context,
):
    server: AbstractServer | None = None
    credentials: SyftCredentials | None = None
    role: ServiceRole


@serializable(canonical_name="APIEndpointWarning", version=1)
class APIEndpointWarning(SyftBaseModel):
    confirmation: bool = False
    message: str | None = None
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

    def message_from(self, context: WarningContext | None) -> Self:
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


@serializable(canonical_name="CRUDWarning", version=1)
class CRUDWarning(APIEndpointWarning):
    def message_from(self, context: WarningContext | None = None) -> Self:
        message = None
        confirmation = self.confirmation
        if context is not None:
            server = context.server
            if server is not None:
                server_side_type = cast(ServerSideType, server.server_side_type)
                server_type = server.server_type

                _msg = (
                    "which could host datasets with private information."
                    if server_side_type.value == ServerSideType.HIGH_SIDE.value
                    else "which only hosts mock or synthetic data."
                )
                if server_type is not None:
                    message = (
                        "You're performing an operation on "
                        f"{server_side_type.value} side {server_type.value}, {_msg}"
                    )
                confirmation = server_side_type.value == ServerSideType.HIGH_SIDE.value

        return CRUDWarning(confirmation=confirmation, message=message)


@serializable(canonical_name="CRUDReminder", version=1)
class CRUDReminder(CRUDWarning):
    confirmation: bool = False

    def message_from(self, context: WarningContext | None = None) -> Self:
        message = None
        confirmation = self.confirmation
        if context is not None:
            server = context.server
            if server is not None:
                server_side_type = cast(ServerSideType, server.server_side_type)
                server_type = server.server_type

                _msg = (
                    "which could host datasets with private information."
                    if server_side_type.value == ServerSideType.HIGH_SIDE.value
                    else "which only hosts mock or synthetic data."
                )
                if server_type is not None:
                    message = (
                        "You're performing an operation on "
                        f"{server_side_type.value} side {server_type.value}, {_msg}"
                    )

        return CRUDReminder(confirmation=confirmation, message=message)


@serializable(canonical_name="LowSideCRUDWarning", version=1)
class LowSideCRUDWarning(APIEndpointWarning):
    def message_from(self, context: WarningContext | None = None) -> Self:
        confirmation = self.confirmation
        message = None
        if context is not None:
            server = context.server
            if server is not None:
                server_side_type = cast(ServerSideType, server.server_side_type)
                server_type = cast(ServerType, server.server_type)
                if server_side_type.value == ServerSideType.LOW_SIDE.value:
                    message = (
                        "You're performing an operation on "
                        f"{server_side_type.value} side {server_type.value} "
                        "which only hosts mock or synthetic data."
                    )

        return LowSideCRUDWarning(confirmation=confirmation, message=message)


@serializable(canonical_name="HighSideCRUDWarning", version=1)
class HighSideCRUDWarning(APIEndpointWarning):
    def message_from(self, context: WarningContext | None = None) -> Self:
        confirmation = self.confirmation
        message = None
        if context is not None:
            server = context.server
            if server is not None:
                server_side_type = cast(ServerSideType, server.server_side_type)
                server_type = cast(ServerType, server.server_type)
                if server_side_type.value == ServerSideType.HIGH_SIDE.value:
                    message = (
                        "You're performing an operation on "
                        f"{server_side_type.value} side {server_type.value} "
                        "which could host datasets with private information."
                    )

        return HighSideCRUDWarning(confirmation=confirmation, message=message)
