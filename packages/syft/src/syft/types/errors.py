# relative
from ..service.context import AuthedServiceContext
from ..service.user.user_roles import ServiceRole


class SyftException(Exception):
    def public(self, context: AuthedServiceContext) -> str:
        return "An error occurred. Contact your admininstrator for more information."

    def get_message(self, context: AuthedServiceContext):
        if context.role.value <= ServiceRole.DATA_OWNER.value:
            return self.public(context)
