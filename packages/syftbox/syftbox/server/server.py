import contextlib
import platform
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from loguru import logger
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor
from typing_extensions import AsyncGenerator, Optional

from syftbox import __version__
from syftbox.server.api.v1.main_router import main_router
from syftbox.server.api.v1.sync_router import router as sync_router
from syftbox.server.emails.router import router as emails_router
from syftbox.server.logger import setup_logger
from syftbox.server.middleware import LoguruMiddleware, RequestSizeLimitMiddleware, VersionCheckMiddleware
from syftbox.server.settings import ServerSettings
from syftbox.server.telemetry import (
    server_request_hook,
    setup_otel_exporter,
)
from syftbox.server.users.router import router as users_router

current_dir = Path(__file__).parent


def _server_setup(app: FastAPI, settings: ServerSettings) -> dict[str, Any]:
    setup_logger(logs_folder=settings.logs_folder)

    logger.info(f"Starting SyftBox Server {__version__}. Python {platform.python_version()}")
    logger.info(settings)

    if settings.otel_enabled:
        logger.info("OTel Exporter is ENABLED")
        setup_otel_exporter(settings.env.value)
    else:
        logger.info("OTel Exporter is DISABLED")

    return {
        "server_settings": settings,
    }


def _server_shutdown(app: FastAPI) -> None:
    logger.info("Shutting down server")


def create_server(settings: Optional[ServerSettings] = None) -> FastAPI:
    settings = settings or ServerSettings()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[Dict[str, Any], None]:
        state = _server_setup(app, settings)
        yield state
        _server_shutdown(app)

    app = FastAPI(lifespan=lifespan)
    app.include_router(main_router)
    app.include_router(emails_router)
    app.include_router(sync_router)
    app.include_router(users_router)

    app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)
    app.add_middleware(LoguruMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware)
    app.add_middleware(VersionCheckMiddleware)

    FastAPIInstrumentor.instrument_app(
        app,
        http_capture_headers_server_request=[".*"],
        server_request_hook=server_request_hook,
    )
    SQLite3Instrumentor().instrument()

    return app


# Global instance for backwards compatibility
app = create_server()
