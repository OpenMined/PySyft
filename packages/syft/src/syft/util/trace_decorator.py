# until this is merged:
# https://github.com/digma-ai/opentelemetry-instrumentation-digma/pull/41

# stdlib
import asyncio
from collections.abc import Callable
from functools import wraps
import inspect
import threading
from typing import Any
from typing import ClassVar
from typing import TypeVar

# third party
from opentelemetry import trace
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.trace import Tracer
from opentelemetry.trace.span import Span

__all__ = ["instrument"]

T = TypeVar("T", bound=Callable | type)


class TracingDecoratorOptions:
    class NamingSchemes:
        @staticmethod
        def function_qualified_name(func: Callable) -> str:
            return func.__qualname__

        default_scheme = function_qualified_name

    naming_scheme: ClassVar[Callable[[Callable], str]] = NamingSchemes.default_scheme
    default_attributes: ClassVar[dict[str, str]] = {}

    @classmethod
    def set_naming_scheme(cls, naming_scheme: Callable[[Callable], str]) -> None:
        cls.naming_scheme = naming_scheme

    @classmethod
    def set_default_attributes(cls, attributes: dict[str, str] | None = None) -> None:
        if attributes is not None:
            for att in attributes:
                cls.default_attributes[att] = attributes[att]


def instrument(
    _func_or_class: T | None = None,
    /,
    *,
    span_name: str = "",
    record_exception: bool = True,
    attributes: dict[str, str] | None = None,
    existing_tracer: Tracer | None = None,
    ignore: bool = False,
) -> T:
    """
    A decorator to instrument a class or function with an OTEL tracing span.
    :param cls: internal, used to specify scope of instrumentation
    :param _func_or_class: The function or span to instrument, this is automatically assigned
    :param span_name: Specify the span name explicitly, rather than use the naming convention.
    This parameter has no effect for class decorators: str
    :param record_exception: Sets whether any exceptions occurring in the span and the stacktrace are recorded
    automatically: bool
    :param attributes:A dictionary of span attributes. These will be automatically added to the span. If defined on a
    class decorator, they will be added to every function span under the class.: dict
    :param existing_tracer: Use a specific tracer instead of creating one :Tracer
    :param ignore: Do not instrument this function, has no effect for class decorators:bool
    :return:The decorator function
    """

    def decorate_class(cls: T) -> T:
        for name, method in inspect.getmembers(cls, inspect.isfunction):
            # Ignore private functions, TODO: maybe make this a setting?
            if not name.startswith("_"):
                if isinstance(inspect.getattr_static(cls, name), staticmethod):
                    setattr(
                        cls,
                        name,
                        staticmethod(
                            instrument(
                                method,
                                record_exception=record_exception,
                                attributes=attributes,
                                existing_tracer=existing_tracer,
                            )
                        ),
                    )
                else:
                    setattr(
                        cls,
                        name,
                        instrument(
                            method,
                            record_exception=record_exception,
                            attributes=attributes,
                            existing_tracer=existing_tracer,
                        ),
                    )

        return cls

    def span_decorator(func_or_class: T) -> T:
        if ignore:
            return func_or_class
        elif inspect.isclass(func_or_class):
            return decorate_class(func_or_class)

        # Check if already decorated (happens if both class and function
        # decorated). If so, we keep the function decorator settings only
        undecorated_func = getattr(func_or_class, "__tracing_unwrapped__", None)
        if undecorated_func:
            # We have already decorated this function, override
            return func_or_class

        func_or_class.__tracing_unwrapped__ = func_or_class  # type: ignore

        tracer = existing_tracer or trace.get_tracer(func_or_class.__module__)

        def _set_semantic_attributes(span: Span, func: Callable) -> None:
            thread = threading.current_thread()
            span.set_attribute(SpanAttributes.CODE_NAMESPACE, func.__module__)
            span.set_attribute(SpanAttributes.CODE_FUNCTION, func.__qualname__)
            span.set_attribute(SpanAttributes.CODE_FILEPATH, func.__code__.co_filename)
            span.set_attribute(SpanAttributes.CODE_LINENO, func.__code__.co_firstlineno)
            span.set_attribute(SpanAttributes.THREAD_ID, thread.ident)
            span.set_attribute(SpanAttributes.THREAD_NAME, thread.name)

        def _set_attributes(
            span: Span, attributes_dict: dict[str, str] | None = None
        ) -> None:
            if attributes_dict is not None:
                for att in attributes_dict:
                    span.set_attribute(att, attributes_dict[att])

        @wraps(func_or_class)
        def wrap_with_span_sync(*args: Any, **kwargs: Any) -> Any:
            name = span_name or TracingDecoratorOptions.naming_scheme(func_or_class)
            with tracer.start_as_current_span(
                name, record_exception=record_exception
            ) as span:
                _set_semantic_attributes(span, func_or_class)
                _set_attributes(span, TracingDecoratorOptions.default_attributes)
                _set_attributes(span, attributes)
                return func_or_class(*args, **kwargs)

        @wraps(func_or_class)
        async def wrap_with_span_async(*args: Any, **kwargs: Any) -> Callable:
            name = span_name or TracingDecoratorOptions.naming_scheme(func_or_class)
            with tracer.start_as_current_span(
                name, record_exception=record_exception
            ) as span:
                _set_semantic_attributes(span, func_or_class)
                _set_attributes(span, TracingDecoratorOptions.default_attributes)
                _set_attributes(span, attributes)
                return await func_or_class(*args, **kwargs)

        span_wrapper = (
            wrap_with_span_async
            if asyncio.iscoroutinefunction(func_or_class)
            else wrap_with_span_sync
        )
        span_wrapper.__signature__ = inspect.signature(func_or_class)

        return span_wrapper  # type: ignore

    # decorator factory on a class or func
    # @instrument or @instrument(span_name="my_span", ...)
    if _func_or_class and inspect.isclass(_func_or_class):
        return decorate_class(_func_or_class)
    elif _func_or_class:
        return span_decorator(_func_or_class)
    else:
        return span_decorator  # type: ignore
