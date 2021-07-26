# third party
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

# grid absolute
from app.api.api_v1.api import api_router
from app.core.config import settings
from app.logger.log import LogHandler

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
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

logHandler = LogHandler()

app.include_router(api_router, prefix=settings.API_V1_STR)
app.add_event_handler("startup", logHandler.init)
