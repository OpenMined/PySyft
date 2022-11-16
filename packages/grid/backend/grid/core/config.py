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
from pydantic import PostgresDsn
from pydantic import validator


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
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

    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "user"
    POSTGRES_PASSWORD: Optional[str] = None
    POSTGRES_DB: str = "db"
    SQLALCHEMY_DATABASE_URI: Optional[Union[PostgresDsn, str]] = None

    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql",
            user=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )

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
    EMAIL_TEMPLATES_DIR: str = "/app/grid/email-templates/build"
    EMAILS_ENABLED: bool = False

    @validator("EMAILS_ENABLED", pre=True)
    def get_emails_enabled(cls, v: bool, values: Dict[str, Any]) -> bool:
        return bool(
            values.get("SMTP_HOST")
            and values.get("SMTP_PORT")
            and values.get("EMAILS_FROM_EMAIL")
        )

    FIRST_SUPERUSER: EmailStr = EmailStr("info@openmined.org")
    FIRST_SUPERUSER_PASSWORD: str = "changethis"
    USERS_OPEN_REGISTRATION: bool = False

    DOMAIN_NAME: str = "default_node_name"
    STREAM_QUEUE: bool = False
    NODE_TYPE: str = "Domain"

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

    REDIS_HOST: str = str(os.getenv("REDIS_HOST", "redis"))
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_STORE_DB_ID: int = int(os.getenv("REDIS_STORE_DB_ID", 0))
    REDIS_LEDGER_DB_ID: int = int(os.getenv("REDIS_LEDGER_DB_ID", 1))
    STORE_DB_ID: int = int(os.getenv("STORE_DB_ID", 0))
    LEDGER_DB_ID: int = int(os.getenv("LEDGER_DB_ID", 1))
    NETWORK_CHECK_INTERVAL: int = int(os.getenv("NETWORK_CHECK_INTERVAL", 60))
    DOMAIN_CHECK_INTERVAL: int = int(os.getenv("DOMAIN_CHECK_INTERVAL", 60))
    CONTAINER_HOST: str = str(os.getenv("CONTAINER_HOST", "docker"))
    TEST_MODE: bool = (
        True if os.getenv("TEST_MODE", "false").lower() == "true" else False
    )

    class Config:
        case_sensitive = True


settings = Settings()
