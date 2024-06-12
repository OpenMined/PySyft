# stdlib
from typing import Any

# relative
from ..service.context import AuthedServiceContext
from ..service.user.user_roles import ServiceRole


class SyftException(Exception):
    """
    A Syft custom exception class with distinct public and private messages.

    Attributes:
        private_message (str): Detailed error message intended for administrators.
        public_message (str): General error message for end-users.
    """

    public_message = "An error occurred. Contact the admin for more information."

    def __init__(
        self,
        private_message: str,
        public_message: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if public_message:
            self.public_message = public_message
        self._private_message = private_message
        super().__init__(self.public, *args, **kwargs)

    @property
    def public(self) -> str:
        """
        Returns the public error message.

        Returns:
            str: The public error message.
        """
        return self.public_message

    def get_message(self, context: AuthedServiceContext) -> str:
        """
        Retrieves the appropriate message based on the user's role, obtained via
        `context.role`.

        Args:
            context (AuthedServiceContext): The context containing user role information.

        Returns:
            str: The private or public message based on the role.
        """
        if context.role.value >= ServiceRole.DATA_OWNER.value:
            return self._private_message
        return self.public
