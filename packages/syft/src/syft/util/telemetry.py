# stdlib
from collections.abc import Callable
import logging
import os
from typing import Any
from typing import TYPE_CHECKING
from typing import TypeVar

# relative
from .. import __version__
from .util import str_to_bool

__all__ = [
    "TRACING_ENABLED",
    "instrument",
    "instrument_fastapi",
    "instrument_botocore",
]

TRACING_ENABLED = str_to_bool(os.environ.get("TRACING", "False"))
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Callable | type)


def no_instrument(__func_or_class: T | None = None, /, *args: Any, **kwargs: Any) -> T:
    def noop_wrapper(__func_or_class: T) -> T:
        return __func_or_class

    if __func_or_class is None:
        return noop_wrapper  # type: ignore
    else:
        return __func_or_class


def setup_instrumenter() -> Any:
    if not TRACING_ENABLED:
        return no_instrument

    try:
        # third party
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import OTELResourceDetector
        from opentelemetry.sdk.resources import ProcessResourceDetector
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # relative
        from .trace_decorator import instrument

        # create a resource
        resource = Resource({"syft.version": __version__})
        resource = resource.merge(OTELResourceDetector().detect())
        resource = resource.merge(ProcessResourceDetector().detect())
        logger.debug(f"OTEL resource : {resource.__dict__}")

        # create a trace provider from the resource
        provider = TracerProvider(resource=resource)

        # create a span processor
        otlp_exporter = OTLPSpanExporter()
        span_processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(span_processor)

        # set the global trace provider
        trace.set_tracer_provider(provider)

        logger.info("Added TracerProvider with BatchSpanProcessor")
        return instrument
    except Exception as e:
        logger.error("Failed to import opentelemetry", exc_info=e)
        return no_instrument


def instrument_fastapi(app: Any) -> None:
    if not TRACING_ENABLED:
        return

    try:
        # third party
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor().instrument_app(app)
        logger.info("Added OTEL FastAPIInstrumentor")
    except Exception as e:
        logger.error(f"Failed to load FastAPIInstrumentor. {e}")


def instrument_botocore() -> None:
    if not TRACING_ENABLED:
        return

    try:
        # third party
        from opentelemetry.instrumentation.botocore import BotocoreInstrumentor

        BotocoreInstrumentor().instrument()
        logger.info("Added OTEL BotocoreInstrumentor")
    except Exception as e:
        logger.error(f"Failed to load BotocoreInstrumentor. {e}")


def instrument_threads() -> None:
    if not TRACING_ENABLED:
        return

    try:
        # third party
        from opentelemetry.instrumentation.threading import ThreadingInstrumentor

        ThreadingInstrumentor().instrument()
        logger.info("Added OTEL ThreadingInstrumentor")
    except Exception as e:
        logger.error(f"Failed to load ThreadingInstrumentor. {e}")


def instrument_sqlalchemny() -> None:
    if not TRACING_ENABLED:
        return

    try:
        # third party
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument(enable_commenter=True, commenter_options={})
        logger.info("Added OTEL SQLAlchemyInstrumentor")
    except Exception as e:
        logger.error(f"Failed to load SQLAlchemyInstrumentor. {e}")


if TYPE_CHECKING:
    # To let static type checker know the returntype of instrument decorators
    instrument = no_instrument
else:
    instrument = setup_instrumenter()
