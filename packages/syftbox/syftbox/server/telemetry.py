from loguru import logger
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import OTELResourceDetector, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span
from typing_extensions import Any

from syftbox import __version__
from syftbox.lib.http import (
    HEADER_OS_ARCH,
    HEADER_OS_NAME,
    HEADER_OS_VERSION,
    HEADER_SYFTBOX_PYTHON,
    HEADER_SYFTBOX_USER,
    HEADER_SYFTBOX_VERSION,
)

OTEL_ATTR_CLIENT_VERSION = "syftbox.client.version"
OTEL_ATTR_CLIENT_PYTHON = "syftbox.client.python"
OTEL_ATTR_CLIENT_USER = "syftbox.client.user"
OTEL_ATTR_CLIENT_USER_LOC = "syftbox.client.user.location"
OTEL_ATTR_CLIENT_OS_NAME = "syftbox.client.os.name"
OTEL_ATTR_CLIENT_OS_VER = "syftbox.client.os.version"
OTEL_ATTR_CLIENT_OS_ARCH = "syftbox.client.os.arch"
OTEL_ATTR_SERVER_VERSION = "syftbox.server.version"


def setup_otel_exporter(env: str) -> None:
    exporter = OTLPSpanExporter()
    span_processor = BatchSpanProcessor(exporter)

    resource = Resource(
        {
            "service.name": "syftbox-server",
            "deployment.environment": env.lower(),
            OTEL_ATTR_SERVER_VERSION: __version__,
        }
    )
    resource = resource.merge(OTELResourceDetector().detect())

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(span_processor)
    trace.set_tracer_provider(tracer_provider)

    logger.info(f"OTEL Exporter: {exporter._endpoint}")
    logger.info(f"OTEL Resource: {tracer_provider.resource.attributes}")


def server_request_hook(span: Span, scope: dict[str, Any]) -> None:
    if not span.is_recording():
        return
    # headers k/v pairs are bytes
    headers: dict[bytes, bytes] = dict(scope.get("headers", {}))
    span.set_attribute(OTEL_ATTR_CLIENT_VERSION, headers.get(HEADER_SYFTBOX_VERSION.encode(), ""))
    span.set_attribute(OTEL_ATTR_CLIENT_PYTHON, headers.get(HEADER_SYFTBOX_PYTHON.encode(), ""))
    span.set_attribute(OTEL_ATTR_CLIENT_USER, headers.get(HEADER_SYFTBOX_USER.encode(), ""))
    span.set_attribute(OTEL_ATTR_CLIENT_OS_NAME, headers.get(HEADER_OS_NAME.encode(), ""))
    span.set_attribute(OTEL_ATTR_CLIENT_OS_VER, headers.get(HEADER_OS_VERSION.encode(), ""))
    span.set_attribute(OTEL_ATTR_CLIENT_OS_ARCH, headers.get(HEADER_OS_ARCH.encode(), ""))
