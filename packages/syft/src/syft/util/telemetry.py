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


def noop(__func_or_class: T | None = None, /, *args: Any, **kwargs: Any) -> T:
    def noop_wrapper(__func_or_class: T) -> T:
        return __func_or_class

    if __func_or_class is None:
        return noop_wrapper  # type: ignore
    else:
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

        # create a provider
        service_name = os.environ.get("OTEL_SERVICE_NAME", "syft-backend")
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        # create a span processor
        otlp_exporter = OTLPSpanExporter()
        span_processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(span_processor)

        # set the global trace provider
        trace.set_tracer_provider(provider)

        # expose the instrument decorator
        instrument = _instrument
    except Exception as e:
        logger.error("Failed to import opentelemetry", exc_info=e)
        instrument = noop
