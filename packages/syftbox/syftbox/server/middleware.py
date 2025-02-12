import time
from typing import Callable

from fastapi import Request, Response, status
from loguru import logger
from packaging import version
from starlette.middleware.base import BaseHTTPMiddleware

from syftbox import __version__
from syftbox.lib.http import (
    HEADER_SYFTBOX_VERSION,
)
from syftbox.lib.version_utils import get_range_for_version


class LoguruMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        logger.info(f"{request.method} {request.url.path} {response.status_code} {duration:.2f}s")

        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_size_limit_in_mb = request.state.server_settings.request_size_limit_in_mb
        request_size_limit_in_bytes = request_size_limit_in_mb * 1024 * 1024

        content_length = request.headers.get("content-length")

        # If content-length header is present, check it first
        if content_length:
            if int(content_length) > request_size_limit_in_bytes:
                return Response(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content=f"Request Denied. Message size is greater than {request_size_limit_in_mb} MB",
                )

        # If content-length header is not present, read the request body and check its size.
        # TODO: This is susceptible to DoS attacks like Slowloris and body flooding. We should check
        # the request stream and terminate early as soon as the size exceeds the limit.
        request_body = await request.body()
        if len(request_body) > request_size_limit_in_bytes:
            return Response(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content=f"Request Denied. Message size is greater than {request_size_limit_in_mb} MB",
            )

        response = await call_next(request)
        return response


class VersionCheckMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        user_agent = request.headers.get("User-Agent")
        if user_agent.startswith("SyftBox"):
            client_version = request.headers.get(HEADER_SYFTBOX_VERSION)

            if not client_version:
                return Response(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content="Client version not provided. Please include the 'Version' header.",
                )

            version_range = get_range_for_version(client_version)

            if isinstance(version_range, str):
                logger.info(version_range)
            else:
                lower_bound_version = version_range[0]

                if version.parse(client_version) < version.parse(lower_bound_version):
                    return Response(
                        status_code=status.HTTP_426_UPGRADE_REQUIRED,
                        content=f"Client version is too old. Minimum version required is {lower_bound_version}",
                    )

        response = await call_next(request)
        response.headers[HEADER_SYFTBOX_VERSION] = __version__
        return response
