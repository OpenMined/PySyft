# stdlib
from collections.abc import Callable
import inspect
from inspect import Parameter
from inspect import Signature
from inspect import _ParameterKind
from inspect import _signature_fromstr
import re

# relative
from .deserialize import _deserialize
from .recursive import recursive_serde_register
from .serialize import _serialize

recursive_serde_register(
    _ParameterKind,
    canonical_name="inspect_ParameterKind",
    version=1,
)


recursive_serde_register(
    Parameter,
    serialize_attrs=["_annotation", "_name", "_kind", "_default"],
    canonical_name="inspect_Parameter",
    version=1,
)


# def serialize_parameter(obj: Parameter) -> bytes:
#     # 🟡 TODO 3: Solve issue of Signature Parameter types being converted to String depending
#                  on import path
#     # currently types arent always being sent correctly maybe due to the path?
#     # instead we can send the fqn and recover it in the type checker on the client side
#     annotation = obj.annotation
#     if not isinstance(annotation, str):
#         annotation = f"{annotation.__module__}.{annotation.__name__}"
#     obj_dict = {
#         "name": obj.name,
#         "kind": obj.kind,
#         "default": obj.default,
#         "annotation": annotation,
#     }
#     return _serialize(obj_dict, to_bytes=True)


# def deserialize_parameter(blob: bytes) -> Parameter:
#     obj_dict = _deserialize(blob, from_bytes=True)
#     return Parameter(**obj_dict)


def serialize_signature(obj: Signature) -> bytes:
    parameters = list(dict(obj.parameters).values())
    return_annotation = obj.return_annotation
    obj_dict = {"parameters": parameters, "return_annotation": return_annotation}
    return _serialize(obj_dict, to_bytes=True)


def deserialize_signature(blob: bytes) -> Signature:
    obj_dict = _deserialize(blob, from_bytes=True)
    return Signature(**obj_dict)


recursive_serde_register(
    Signature,
    serialize_signature,
    deserialize_signature,
    canonical_name="inspect_Signature",
    version=1,
)


def signature_remove_self(signature: Signature) -> Signature:
    params = dict(signature.parameters)
    params.pop("self", None)
    return Signature(
        list(params.values()), return_annotation=signature.return_annotation
    )


def signature_remove_context(signature: Signature) -> Signature:
    params = dict(signature.parameters)
    params.pop("context", None)
    return Signature(
        list(params.values()), return_annotation=signature.return_annotation
    )


def signature_remove(signature: Signature, args: list[str]) -> Signature:
    params = dict(signature.parameters)
    for arg in args:
        params.pop(arg, None)
    return Signature(
        list(params.values()), return_annotation=signature.return_annotation
    )


def get_str_signature_from_docstring(doc: str, callable_name: str) -> str | None:
    if not doc or callable_name not in doc:
        return None
    else:
        # gets part of the docstring that contains the signature
        doc = re.sub(r"\s", "", doc.split("\n\n")[0])
        # gets signature
        signature_search_result = re.search(rf"{callable_name}\((.+)\)", doc)
        if signature_search_result:
            signature_str = signature_search_result.group(1)
            # decomposing "[]" optional  params
            params = re.findall(r"\[(.+?)\]", signature_str)

            if params:
                # for instance in:
                # numpy.matmul(x1, x2, /, out=None, *, casting='same_kind',
                # order='K', dtype=None, subok=True[, signature, extobj, axes, axis])
                # we have [, signature, extobj, axes, axis] , which we replace as normal params
                for param in params[:-1]:
                    signature_str = signature_str.replace(f"[{param}]", param)

                if re.search(rf"(?<={params[-1]})\],", signature_str):
                    signature_str = signature_str.replace(
                        f"[{params[-1]}],", params[-1]
                    )
                else:
                    signature_str = signature_str.replace(
                        f"[{params[-1]}]",
                        f', {",".join([f"{param}=None" for param in params[-1].split(",") if param])}',
                    )

            # removes "/" and "*"
            signature_str = re.sub(r",(\/|\*)", "", signature_str)
            # removes dtypes
            signature_str = re.sub(r"dtype=(\w+),", "dtype=None,", signature_str)
            return f"{callable_name}({signature_str})"
        else:
            return None


def generate_signature(_callable: Callable) -> inspect.Signature:
    name, doc = _callable.__name__, _callable.__doc__
    # returning predefined signature if in signature registry
    # name_in_registry = name in function_signatures_registry.keys()
    text_signature = get_str_signature_from_docstring(doc, name)
    # TODO safe handling if function signature can not be generated
    return _signature_fromstr(inspect.Signature, _callable, text_signature, True)


def get_signature(_callable: Callable) -> inspect.Signature:
    try:
        res = inspect.signature(_callable)
        if res is None:
            raise ValueError("")
        else:
            return res
    except Exception:
        return generate_signature(_callable)
