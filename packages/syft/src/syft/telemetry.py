# stdlib
import os
from typing import Any
from typing import Optional


def str_to_bool(bool_str: Optional[str]) -> bool:
    result = False
    bool_str = str(bool_str).lower()
    if bool_str == "true" or bool_str == "1":
        result = True
    return result


TRACE_MODE = str_to_bool(os.environ.get("TRACE", "False"))


def setup_tracer() -> Any:
    if not TRACE_MODE:

        def noop(func: Any) -> Any:
            return func

        return noop

    print("OpenTelemetry Tracing enabled")
    service_name = os.environ.get("SERVICE_NAME", "client")
    jaeger_host = os.environ.get("JAEGER_HOST", "localhost")
    # jaeger_port = int(os.environ.get("JAEGER_PORT", "6831"))

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
        collector_endpoint=f"http://{jaeger_host}:14268/api/traces?format=jaeger.thrift",
        # udp_split_oversized_batches=True,
    )

    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(jaeger_exporter))

    # from opentelemetry.sdk.trace.export import ConsoleSpanExporter
    # console_exporter = ConsoleSpanExporter()
    # span_processor = BatchSpanProcessor(console_exporter)
    # trace.get_tracer_provider().add_span_processor(span_processor)

    # third party
    from opentelemetry.instrumentation.digma.trace_decorator import (
        instrument as _instrument,
    )

    return _instrument


instrument = setup_tracer()
