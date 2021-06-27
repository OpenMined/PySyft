from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.api_v1.api import api_router
from app.core.config import settings

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

app.include_router(api_router, prefix=settings.API_V1_STR)

# HOT RELOADING PYSYFT - so that when we make changes to the Syft director
# it rebuilds the code, but it ignores syft if we didn't make any changes

from checksumdir import dirhash
import pip

try:
    f = open('/syft_md5_hash','r')
    prev_hash = f.readlines()[0]
    f.close()
except Exception as e:
    prev_hash = ""

directory  = './app/syft'
md5hash    = dirhash(directory, 'md5')

f = open('/syft_md5_hash','w')
f.write(md5hash)
f.close()

def install(package):
    pip.main(['install', package])

if prev_hash != md5hash:
    print("INFO:     SYFT CHANGED! - rebuilding...")
    install('./app/syft')
else:
    print("INFO:     PySyft didn't change - ignoring")