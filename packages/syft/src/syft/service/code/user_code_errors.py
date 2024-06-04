# relative
from ...types.errors import SyftException


class UserCodeException(SyftException): ...


class UserCodeNotApprovedException(UserCodeException): ...


class UserCodeBadInputPolicyException(UserCodeException): ...


class UserCodeInvalidRequestException(UserCodeException): ...
