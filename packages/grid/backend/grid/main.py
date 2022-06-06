# stdlib
from typing import Dict

# third party
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from starlette.middleware.cors import CORSMiddleware

# grid absolute
from grid.api.router import api_router
from grid.core.config import settings
from grid.logger.handler import get_log_handler

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

sentry_sdk.init(
    "https://bd20175e36374f1c9edef90c9b0ba94c@o488706.ingest.sentry.io/6465580",
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0,
    send_default_pii=True,
)

# app = SentryAsgiMiddleware(app)
app.add_middleware(SentryAsgiMiddleware)

app.add_event_handler("startup", get_log_handler().init_logger)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)


# needed for Google Kubernetes Engine LoadBalancer Healthcheck
@app.get(
    "/",
    name="healthcheck",
    status_code=200,
    response_class=JSONResponse,
)
def healthcheck() -> Dict[str, str]:
    """
    Currently, all service backends must satisfy either of the following requirements to
    pass the HTTP health checks sent to it from the GCE loadbalancer: 1. Respond with a
    200 on '/'. The content does not matter. 2. Expose an arbitrary url as a readiness
    probe on the pods backing the Service.
    """
    return {"status": "ok"}
