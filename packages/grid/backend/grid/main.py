# stdlib
from contextlib import asynccontextmanager
import logging
from typing import Any

# third party
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# syft absolute
from syft.util.telemetry import instrument_fastapi

# server absolute
from grid.api.router import api_router
from grid.core.config import settings
from grid.core.server import worker

# logger => grid.main
logger = logging.getLogger(__name__)


class FastAPILogFilter(logging.Filter):
    HEALTHCHECK_ENDPOINT = f"{settings.API_V2_STR}/?probe="

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find(self.HEALTHCHECK_ENDPOINT) == -1


def on_app_startup(app: FastAPI) -> None:
    if settings.DEV_MODE:
        # syft absolute
        from syft.protocol.data_protocol import stage_protocol_changes

        logger.info("Staging protocol changes...")
        status = stage_protocol_changes()
        logger.info(f"Staging protocol result: {status}")


def on_app_shutdown(app: FastAPI) -> None:
    worker.stop()
    logger.info("Worker Stopped")


def get_middlewares() -> FastAPI:
    middlewares = []

    # Set all CORS enabled origins
    if settings.BACKEND_CORS_ORIGINS:
        middlewares.append(
            Middleware(
                CORSMiddleware,
                allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        )

    return middlewares


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    try:
        on_app_startup(app)
        yield
    finally:
        on_app_shutdown(app)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V2_STR}/openapi.json",
        lifespan=lifespan,
        middleware=get_middlewares(),
        docs_url=None,
        redoc_url=None,
    )

    # instrument app
    instrument_fastapi(app)

    # patch logger to ignore healthcheck logs
    logging.getLogger("uvicorn.access").addFilter(FastAPILogFilter())

    # add Syft API routes
    app.include_router(api_router, prefix=settings.API_V2_STR)

    return app


app = create_app()


@app.get(
    "/",
    name="healthcheck",
    status_code=200,
    response_class=JSONResponse,
)
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
