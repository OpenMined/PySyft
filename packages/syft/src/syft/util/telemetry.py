# stdlib
from collections.abc import Callable
import os
from typing import Any
from typing import TypeVar


def str_to_bool(bool_str: str | None) -> bool:
    result = False
    bool_str = str(bool_str).lower()
    if bool_str == "true" or bool_str == "1":
        result = True
    return result


TRACE_MODE = str_to_bool(os.environ.get("TRACE", "False"))


T = TypeVar("T", bound=Callable | type)


def noop(__func_or_class: T, /, *args: Any, **kwargs: Any) -> T:
    return __func_or_class


if not TRACE_MODE:
    instrument = noop
else:
    try:
        print("OpenTelemetry Tracing enabled")
        service_name = os.environ.get("SERVICE_NAME", "client")
        jaeger_host = os.environ.get("JAEGER_HOST", "localhost")
        jaeger_port = int(os.environ.get("JAEGER_PORT", "14268"))

        # third party
        from opentelemetry import trace
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.resources import SERVICE_NAME
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        trace.set_tracer_provider(
            TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
        )
        jaeger_exporter = JaegerExporter(
            # agent_host_name=jaeger_host,
            # agent_port=jaeger_port,
            collector_endpoint=f"http://{jaeger_host}:{jaeger_port}/api/traces?format=jaeger.thrift",
            # udp_split_oversized_batches=True,
        )

        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(jaeger_exporter)
        )

        # from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        # console_exporter = ConsoleSpanExporter()
        # span_processor = BatchSpanProcessor(console_exporter)
        # trace.get_tracer_provider().add_span_processor(span_processor)

        # third party
        import opentelemetry.instrumentation.requests

        opentelemetry.instrumentation.requests.RequestsInstrumentor().instrument()

        # relative
        # from opentelemetry.instrumentation.digma.trace_decorator import (
        #     instrument as _instrument,
        # )
        #
        # until this is merged:
        # https://github.com/digma-ai/opentelemetry-instrumentation-digma/pull/41
        from .trace_decorator import instrument as _instrument

        instrument = _instrument
    except Exception:  # nosec
        print("Failed to import opentelemetry")
        instrument = noop
