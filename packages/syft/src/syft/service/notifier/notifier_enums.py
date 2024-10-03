# stdlib
from enum import Enum
from enum import auto

# relative
from ...serde.serializable import serializable


@serializable(canonical_name="EMAIL_TYPES", version=1)
class EMAIL_TYPES(Enum):
    PASSWORD_RESET_EMAIL = "PasswordResetTemplate"  # nosec
    ONBOARD_EMAIL = "OnBoardEmailTemplate"
    REQUEST_EMAIL = "RequestEmailTemplate"
    REQUEST_UPDATE_EMAIL = "RequestUpdateEmailTemplate"


@serializable(canonical_name="NOTIFIERS", version=1)
class NOTIFIERS(Enum):
    EMAIL = auto()
    SMS = auto()
    SLACK = auto()
    APP = auto()


@serializable(canonical_name="NOTIFICATION_FREQUENCY", version=1)
class NOTIFICATION_FREQUENCY(Enum):
    INSTANT = auto()
    SIX_HOURS = auto()
    TWELVE_HOURS = auto()
    DAILY = auto()
    WEEKLY = auto()
