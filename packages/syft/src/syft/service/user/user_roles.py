# stdlib
from enum import Enum
from typing import Any
from typing import List
from typing import Tuple
from typing import Union

# third party
from typing_extensions import Self

# relative
from ...serde.serializable import serializable


class ServiceRoleCapability(Enum):
    CAN_MAKE_DATA_REQUESTS = 1
    CAN_TRIAGE_DATA_REQUESTS = 2
    CAN_MANAGE_PRIVACY_BUDGET = 4
    CAN_CREATE_USERS = 8
    CAN_MANAGE_USERS = 16
    CAN_EDIT_ROLES = 32
    CAN_MANAGE_INFRASTRUCTURE = 64
    CAN_UPLOAD_DATA = 128
    CAN_UPLOAD_LEGAL_DOCUMENT = 256
    CAN_EDIT_DOMAIN_SETTINGS = 512


@serializable()
class ServiceRole(Enum):
    NONE = 0
    GUEST = 1
    DATA_SCIENTIST = 2
    DATA_OWNER = 32
    ADMIN = 128

    # Disabling it, as both property and classmethod only works for python >= 3.9
    # @property
    @classmethod
    def roles_descending(cls) -> List[Tuple[int, Self]]:
        tuples = []
        for x in cls:
            tuples.append((x.value, x))
        return list(reversed(sorted(tuples)))

    @staticmethod
    def roles_for_level(level: Union[int, Self]) -> List[Self]:
        if isinstance(level, ServiceRole):
            level = level.value
        roles = []
        level_float = float(level)
        service_roles = ServiceRole.roles_descending()
        for role in service_roles:
            role_num = role[0]
            if role_num == 0:
                continue
            role_enum = role[1]
            if level_float / role_num >= 1:
                roles.append(role_enum)
                level_float = level_float % role_num
        return roles

    def capabilities(self):
        return ROLE_TO_CAPABILITIES[self]

    def __add__(self, other: Any) -> int:
        if isinstance(other, ServiceRole):
            return self.value + other.value
        return self.value + other

    def __radd__(self, other: Any) -> int:
        return self.__add__(other)

    def __ge__(self, other: Self) -> bool:
        return self.value >= other.value

    def __le__(self, other: Self) -> bool:
        return self.value <= other.value

    def __gt__(self, other: Self) -> bool:
        return self.value > other.value

    def __lt__(self, other: Self) -> bool:
        return self.value < other.value


GUEST_ROLE_LEVEL = ServiceRole.roles_for_level(
    ServiceRole.GUEST
    + ServiceRole.DATA_SCIENTIST
    + ServiceRole.DATA_OWNER
    + ServiceRole.ADMIN
)

DATA_SCIENTIST_ROLE_LEVEL = ServiceRole.roles_for_level(
    ServiceRole.DATA_SCIENTIST + ServiceRole.DATA_OWNER + ServiceRole.ADMIN
)

ONLY_DATA_SCIENTIST_ROLE_LEVEL = ServiceRole.roles_for_level(ServiceRole.DATA_SCIENTIST)

DATA_OWNER_ROLE_LEVEL = ServiceRole.roles_for_level(
    ServiceRole.DATA_OWNER + ServiceRole.ADMIN
)

ADMIN_ROLE_LEVEL = ServiceRole.roles_for_level(ServiceRole.ADMIN)

ROLE_TO_CAPABILITIES = {
    ServiceRole.NONE: [],
    ServiceRole.GUEST: [
        ServiceRoleCapability.CAN_MAKE_DATA_REQUESTS,
    ],
    ServiceRole.DATA_SCIENTIST: [
        ServiceRoleCapability.CAN_MAKE_DATA_REQUESTS,
    ],
    ServiceRole.DATA_OWNER: [
        ServiceRoleCapability.CAN_MAKE_DATA_REQUESTS,
        ServiceRoleCapability.CAN_TRIAGE_DATA_REQUESTS,
        ServiceRoleCapability.CAN_MANAGE_PRIVACY_BUDGET,
        ServiceRoleCapability.CAN_CREATE_USERS,
        ServiceRoleCapability.CAN_EDIT_ROLES,
        ServiceRoleCapability.CAN_UPLOAD_DATA,
    ],
    ServiceRole.ADMIN: list(ServiceRoleCapability),
}
