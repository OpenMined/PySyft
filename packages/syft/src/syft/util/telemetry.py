# stdlib
from collections.abc import Callable
import logging
import os
from typing import Any
from typing import TypeVar

# relative
from .util import str_to_bool

__all__ = ["TRACING_ENABLED", "instrument"]

logger = logging.getLogger(__name__)

TRACING_ENABLED = str_to_bool(os.environ.get("TRACING", "False"))

T = TypeVar("T", bound=Callable | type)


def noop(__func_or_class: T, /, *args: Any, **kwargs: Any) -> T:
    return __func_or_class


if not TRACING_ENABLED:
    instrument = noop
else:
    try:
        # third party
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # relative
        from .trace_decorator import instrument as _instrument

        service_name = os.environ.get("OTEL_SERVICE_NAME", "syft-backend")
        trace.set_tracer_provider(
            TracerProvider(resource=Resource.create({"service.name": service_name}))
        )

        # configured through env:OTEL_EXPORTER_OTLP_ENDPOINT
        otlp_exporter = OTLPSpanExporter()
        span_processor = BatchSpanProcessor(otlp_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

        instrument = _instrument
    except Exception as e:
        logger.error("Failed to import opentelemetry", exc_info=e)
        instrument = noop
