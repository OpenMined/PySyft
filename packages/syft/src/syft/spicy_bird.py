import os
import requests
from contextlib import contextmanager
from pathlib import Path


def str_to_bool(bool_str: Optional[str]) -> bool:
    result = False
    bool_str = str(bool_str).lower()
    if bool_str == "true" or bool_str == "1":
        result = True
    return result


def download_file(url: str, full_path: Union[str, Path]) -> Path:
    if not os.path.exists(full_path):
        # TODO: Rasswanth fix the SSL Error.
        r = requests.get(url, allow_redirects=True, verify=verify_tls())  # nosec
        path = os.path.dirname(full_path)
        os.makedirs(path, exist_ok=True)
        with open(full_path, "wb") as f:
            f.write(r.content)
    return Path(full_path)


def get_root_data_path() -> Path:
    # get the PySyft / data directory to share datasets between notebooks
    # on Linux and MacOS the directory is: ~/.syft/data"
    # on Windows the directory is: C:/Users/$USER/.syft/data

    data_dir = Path.home() / ".syft" / "data"

    os.makedirs(data_dir, exist_ok=True)
    return data_dir


_tracer = None


def get_tracer(service_name: Optional[str] = None) -> Any:
    global _tracer
    if _tracer is not None:
        return _tracer

    PROFILE_MODE = str_to_bool(os.environ.get("PROFILE", "False"))
    PROFILE_MODE = False
    if not PROFILE_MODE:

        class NoopTracer:
            @contextmanager
            def start_as_current_span(*args: Any, **kwargs: Any) -> Any:
                yield None

        _tracer = NoopTracer()
        return _tracer

    print("Profile mode with OpenTelemetry enabled")
    if service_name is None:
        service_name = os.environ.get("SERVICE_NAME", "client")

    jaeger_host = os.environ.get("JAEGER_HOST", "localhost")
    jaeger_port = int(os.environ.get("JAEGER_PORT", "6831"))

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
        agent_host_name=jaeger_host,
        agent_port=jaeger_port,
    )

    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(jaeger_exporter))

    _tracer = trace.get_tracer(__name__)
    return _tracer