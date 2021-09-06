# third party
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

# grid absolute
from grid.api.router import api_router
from grid.core.config import settings
from grid.logger.handler import get_log_handler

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

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
