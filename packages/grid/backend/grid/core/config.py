# stdlib
import os
import secrets
from typing import Any

# third party
from pydantic import AnyHttpUrl
from pydantic import EmailStr
from pydantic import HttpUrl
from pydantic import field_validator
from pydantic import model_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
from typing_extensions import Self

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
    BACKEND_CORS_ORIGINS: list[AnyHttpUrl] = []

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: str | list[str]) -> list[str] | str:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list | str):
            return v
        raise ValueError(v)

    PROJECT_NAME: str = "grid"

    SENTRY_DSN: HttpUrl | None = None

    @field_validator("SENTRY_DSN", mode="before")
    @classmethod
    def sentry_dsn_can_be_blank(cls, v: str) -> str | None:
        if v is None or len(v) == 0:
            return None
        return v

    EMAILS_FROM_EMAIL: EmailStr | None = None
    EMAILS_FROM_NAME: str | None = None

    @model_validator(mode="after")
    def get_project_name(self) -> Self:
        if not self.EMAILS_FROM_NAME:
            self.EMAILS_FROM_NAME = self.PROJECT_NAME

        return self

    EMAIL_RESET_TOKEN_EXPIRE_HOURS: int = 48
    EMAIL_TEMPLATES_DIR: str = os.path.expandvars(
        "$HOME/app/grid/email-templates/build"
    )
    EMAILS_ENABLED: bool = False

    @model_validator(mode="after")
    def get_emails_enabled(self) -> Self:
        self.EMAILS_ENABLED = bool(
            self.SMTP_HOST and self.SMTP_PORT and self.EMAILS_FROM_EMAIL
        )

        return self

    DEFAULT_ROOT_EMAIL: EmailStr = "info@openmined.org"
    DEFAULT_ROOT_PASSWORD: str = "changethis"
    USERS_OPEN_REGISTRATION: bool = False

    SERVER_NAME: str = "default_server_name"
    STREAM_QUEUE: bool = False
    SERVER_TYPE: str = "datasite"

    OPEN_REGISTRATION: bool = True

    # DATASITE_ASSOCIATION_REQUESTS_AUTOMATICALLY_ACCEPTED: bool = True
    USE_BLOB_STORAGE: bool = (
        True if os.getenv("USE_BLOB_STORAGE", "false").lower() == "true" else False
    )
    S3_ENDPOINT: str = os.getenv("S3_ENDPOINT", "seaweedfs")
    S3_PORT: int = int(os.getenv("S3_PORT", 8333))
    S3_ROOT_USER: str = os.getenv("S3_ROOT_USER", "admin")
    S3_ROOT_PWD: str | None = os.getenv("S3_ROOT_PWD", "admin")
    S3_REGION: str = os.getenv("S3_REGION", "us-east-1")
    S3_PRESIGNED_TIMEOUT_SECS: int = int(
        os.getenv("S3_PRESIGNED_TIMEOUT_SECS", 1800)
    )  # 30 minutes in seconds
    SEAWEED_MOUNT_PORT: int = int(os.getenv("SEAWEED_MOUNT_PORT", 4001))

    # REDIS_HOST: str = str(os.getenv("REDIS_HOST", "redis"))
    # REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    # REDIS_STORE_DB_ID: int = int(os.getenv("REDIS_STORE_DB_ID", 0))
    # REDIS_LEDGER_DB_ID: int = int(os.getenv("REDIS_LEDGER_DB_ID", 1))
    # STORE_DB_ID: int = int(os.getenv("STORE_DB_ID", 0))
    # LEDGER_DB_ID: int = int(os.getenv("LEDGER_DB_ID", 1))
    # NETWORK_CHECK_INTERVAL: int = int(os.getenv("NETWORK_CHECK_INTERVAL", 60))
    # DATASITE_CHECK_INTERVAL: int = int(os.getenv("DATASITE_CHECK_INTERVAL", 60))
    CONTAINER_HOST: str = str(os.getenv("CONTAINER_HOST", "docker"))
    POSTGRESQL_DBNAME: str = str(os.getenv("POSTGRESQL_DBNAME", ""))
    POSTGRESQL_HOST: str = str(os.getenv("POSTGRESQL_HOST", ""))
    POSTGRESQL_PORT: int = int(os.getenv("POSTGRESQL_PORT", 5432))
    POSTGRESQL_USERNAME: str = str(os.getenv("POSTGRESQL_USERNAME", ""))
    POSTGRESQL_PASSWORD: str = str(os.getenv("POSTGRESQL_PASSWORD", ""))
    DEV_MODE: bool = True if os.getenv("DEV_MODE", "false").lower() == "true" else False
    # ZMQ stuff
    QUEUE_PORT: int = int(os.getenv("QUEUE_PORT", 5556))
    CREATE_PRODUCER: bool = (
        True if os.getenv("CREATE_PRODUCER", "false").lower() == "true" else False
    )
    N_CONSUMERS: int = int(os.getenv("N_CONSUMERS", 1))
    SQLITE_PATH: str = os.path.expandvars("/tmp/data/db")
    SINGLE_CONTAINER_MODE: bool = str_to_bool(os.getenv("SINGLE_CONTAINER_MODE", False))
    CONSUMER_SERVICE_NAME: str | None = os.getenv("CONSUMER_SERVICE_NAME")
    INMEMORY_WORKERS: bool = str_to_bool(os.getenv("INMEMORY_WORKERS", True))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
    EMAIL_SENDER: str = os.getenv("EMAIL_SENDER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_TLS: bool = True
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", 587))
    SMTP_HOST: str = os.getenv("SMTP_HOST", "")

    TEST_MODE: bool = (
        True if os.getenv("TEST_MODE", "false").lower() == "true" else False
    )
    ASSOCIATION_TIMEOUT: int = 10
    ASSOCIATION_REQUEST_AUTO_APPROVAL: bool = str_to_bool(
        os.getenv("ASSOCIATION_REQUEST_AUTO_APPROVAL", "False")
    )
    MIN_SIZE_BLOB_STORAGE_MB: int = int(os.getenv("MIN_SIZE_BLOB_STORAGE_MB", 1))
    REVERSE_TUNNEL_ENABLED: bool = str_to_bool(
        os.getenv("REVERSE_TUNNEL_ENABLED", "false")
    )
    TRACING_ENABLED: bool = str_to_bool(os.getenv("TRACING", "False"))

    model_config = SettingsConfigDict(case_sensitive=True)


settings = Settings()
