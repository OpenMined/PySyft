# stdlib
import os
import secrets
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
from pydantic import AnyHttpUrl
from pydantic import BaseSettings
from pydantic import EmailStr
from pydantic import HttpUrl
from pydantic import validator

_truthy = {"yes", "y", "true", "t", "on", "1"}
_falsy = {"no", "n", "false", "f", "off", "0"}


def _distutils_strtoint(s: str) -> int:
    """implements the deprecated distutils.util.strtoint"""
    ls = s.lower()
    if ls in _truthy:
        return 1
    if ls in _falsy:
        return 0
    raise ValueError(f"invalid truth value '{s}'")


def str_to_int(bool_str: Any) -> int:
    try:
        return _distutils_strtoint(str(bool_str))
    except ValueError:
        return 0


def str_to_bool(bool_str: Any) -> bool:
    return bool(str_to_int(bool_str))


class Settings(BaseSettings):
    API_V2_STR: str = "/api/v2"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    SERVER_HOST: str = "https://localhost"
    # BACKEND_CORS_ORIGINS is a JSON-formatted list of origins
    # e.g: '["http://localhost", "http://localhost:4200", "http://localhost:3000", \
    # "http://localhost:8080", "http://local.dockertoolbox.tiangolo.com"]'
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    PROJECT_NAME: str = "grid"

    SENTRY_DSN: Optional[HttpUrl] = None

    @validator("SENTRY_DSN", pre=True)
    def sentry_dsn_can_be_blank(cls, v: str) -> Optional[str]:
        if v is None or len(v) == 0:
            return None
        return v

    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = None
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[EmailStr] = None
    EMAILS_FROM_NAME: Optional[str] = None

    @validator("EMAILS_FROM_NAME")
    def get_project_name(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        if not v:
            return values["PROJECT_NAME"]
        return v

    EMAIL_RESET_TOKEN_EXPIRE_HOURS: int = 48
    EMAIL_TEMPLATES_DIR: str = os.path.expandvars(
        "$HOME/app/grid/email-templates/build"
    )
    EMAILS_ENABLED: bool = False

    @validator("EMAILS_ENABLED", pre=True)
    def get_emails_enabled(cls, v: bool, values: Dict[str, Any]) -> bool:
        return bool(
            values.get("SMTP_HOST")
            and values.get("SMTP_PORT")
            and values.get("EMAILS_FROM_EMAIL")
        )

    DEFAULT_ROOT_EMAIL: EmailStr = EmailStr("info@openmined.org")
    DEFAULT_ROOT_PASSWORD: str = "changethis"
    USERS_OPEN_REGISTRATION: bool = False

    NODE_NAME: str = "default_node_name"
    STREAM_QUEUE: bool = False
    NODE_TYPE: str = "domain"

    OPEN_REGISTRATION: bool = True

    DOMAIN_ASSOCIATION_REQUESTS_AUTOMATICALLY_ACCEPTED: bool = True
    USE_BLOB_STORAGE: bool = (
        True if os.getenv("USE_BLOB_STORAGE", "false").lower() == "true" else False
    )
    S3_ENDPOINT: str = os.getenv("S3_ENDPOINT", "seaweedfs")
    S3_PORT: int = int(os.getenv("S3_PORT", 8333))
    S3_ROOT_USER: str = os.getenv("S3_ROOT_USER", "admin")
    S3_ROOT_PWD: Optional[str] = os.getenv("S3_ROOT_PWD", "admin")
    S3_REGION: str = os.getenv("S3_REGION", "us-east-1")
    S3_PRESIGNED_TIMEOUT_SECS: int = int(
        os.getenv("S3_PRESIGNED_TIMEOUT_SECS", 1800)
    )  # 30 minutes in seconds
    SEAWEED_MOUNT_PORT: int = int(os.getenv("SEAWEED_MOUNT_PORT", 4001))

    REDIS_HOST: str = str(os.getenv("REDIS_HOST", "redis"))
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_STORE_DB_ID: int = int(os.getenv("REDIS_STORE_DB_ID", 0))
    REDIS_LEDGER_DB_ID: int = int(os.getenv("REDIS_LEDGER_DB_ID", 1))
    STORE_DB_ID: int = int(os.getenv("STORE_DB_ID", 0))
    LEDGER_DB_ID: int = int(os.getenv("LEDGER_DB_ID", 1))
    NETWORK_CHECK_INTERVAL: int = int(os.getenv("NETWORK_CHECK_INTERVAL", 60))
    DOMAIN_CHECK_INTERVAL: int = int(os.getenv("DOMAIN_CHECK_INTERVAL", 60))
    CONTAINER_HOST: str = str(os.getenv("CONTAINER_HOST", "docker"))
    MONGO_HOST: str = str(os.getenv("MONGO_HOST", ""))
    MONGO_PORT: int = int(os.getenv("MONGO_PORT", 0))
    MONGO_USERNAME: str = str(os.getenv("MONGO_USERNAME", ""))
    MONGO_PASSWORD: str = str(os.getenv("MONGO_PASSWORD", ""))
    DEV_MODE: bool = True if os.getenv("DEV_MODE", "false").lower() == "true" else False
    # ZMQ stuff
    QUEUE_PORT: int = int(os.getenv("QUEUE_PORT", 0))
    CREATE_PRODUCER: bool = (
        True if os.getenv("CREATE_PRODUCER", "false").lower() == "true" else False
    )
    N_CONSUMERS: int = int(os.getenv("N_CONSUMERS", 1))
    SQLITE_PATH: str = os.path.expandvars("$HOME/data/db/")
    SINGLE_CONTAINER_MODE: bool = str_to_bool(os.getenv("SINGLE_CONTAINER_MODE", False))
    CONSUMER_SERVICE_NAME: Optional[str] = os.getenv("CONSUMER_SERVICE_NAME")
    INMEMORY_WORKERS: bool = str_to_bool(os.getenv("INMEMORY_WORKERS", True))

    TEST_MODE: bool = (
        True if os.getenv("TEST_MODE", "false").lower() == "true" else False
    )
    ASSOCIATION_TIMEOUT: int = 10

    class Config:
        case_sensitive = True


settings = Settings()
