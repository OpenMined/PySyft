# stdlib
from contextlib import asynccontextmanager
import logging
from typing import Any

# third party
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

# syft absolute
from syft.protocol.data_protocol import stage_protocol_changes

# server absolute
from grid.api.router import api_router
from grid.core.config import settings
from grid.core.server import worker


class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/api/v2/?probe=livenessProbe") == -1


logger = logging.getLogger("uvicorn.error")
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    try:
        yield
    finally:
        worker.stop()
        logger.info("Worker Stop")


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V2_STR}/openapi.json",
    lifespan=lifespan,
)


# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V2_STR)
logger.info("Included routes, app should now be reachable")


if settings.DEV_MODE:
    logger.info("Staging protocol changes...")
    status = stage_protocol_changes()
    logger.info(f"Staging protocol result: {status}")


# needed for Google Kubernetes Engine LoadBalancer Healthcheck
@app.get(
    "/",
    name="healthcheck",
    status_code=200,
    response_class=JSONResponse,
)
def healthcheck() -> dict[str, str]:
    """
    Currently, all service backends must satisfy either of the following requirements to
    pass the HTTP health checks sent to it from the GCE loadbalancer: 1. Respond with a
    200 on '/'. The content does not matter. 2. Expose an arbitrary url as a readiness
    probe on the pods backing the Service.
    """
    return {"status": "ok"}


if settings.TRACING_ENABLED:
    # third party
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)
