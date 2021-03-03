# stdlib
from inspect import signature
from itertools import chain
from typing import Optional

# syft relative
from ...util import T
from .allowlist import templated_allowlist
from .templatable_iterator import TemplateableIterator

type_cache = dict()


def generate_attrs_and_allowlist(
    templated_type, target_underlying_type, skip_typechecking, new_type_qualname
):
    attrs = {}
    new_allowlist = {}

    def wrap_type_check(method):
        target_signature = signature(method)
        variable_checks = []

        for idx, (name, target_type) in enumerate(target_signature.parameters.items()):
            if target_type.annotation is T:
                variable_checks.append(idx)

        ret_type_check = target_signature.return_annotation is T

        def func(variable_checks):
            def _func(*args, **kwargs):
                arg_list = list(chain(args, kwargs.items()))

                if not skip_typechecking:
                    for idx in variable_checks:
                        assert isinstance(arg_list[idx], target_underlying_type)

                result = method(*args, **kwargs)

                if ret_type_check and not skip_typechecking:
                    assert isinstance(result, target_underlying_type)

                return result

            return _func

        return func(variable_checks)

    for method_name, type_path in templated_allowlist.items():
        method = getattr(templated_type, method_name)

        if not type_path:
            attrs[method_name] = wrap_type_check(method)
            new_allowlist[
                new_type_qualname + "." + method_name
            ] = target_underlying_type.__qualname__
        else:
            attrs[method_name] = getattr(TemplateableIterator, method_name)
            new_allowlist[new_type_qualname + "." + method_name] = type_path
    return attrs, new_allowlist


class IndexableTrait(type):
    def __getitem__(self, target_type: Optional[str] = None) -> str:
        if target_type is None:
            return Iterator.from_qualname("syft.lib.python.Any", skip_typechecking=True)

        return Iterator.from_qualname(target_type)


class Iterator(type, metaclass=IndexableTrait):
    PREFIX_PATH = "syft.lib.python.iterator."
    SUFFIX_PATH = "Iterator"

    @staticmethod
    def generate_name(targeted_underlying_path_name: str) -> (str, str):
        targeted_type = targeted_underlying_path_name.rsplit(".", 1)[-1]
        name = targeted_type + Iterator.SUFFIX_PATH
        qualname = Iterator.PREFIX_PATH + name
        return name, qualname

    @staticmethod
    def from_qualname(
        underlying_type_path: str, skip_typechecking: bool = False
    ) -> str:
        _, qualname = Iterator.generate_name(underlying_type_path)

        if qualname in type_cache:
            return qualname
        else:
            type_cache[qualname] = (underlying_type_path, skip_typechecking)
            return qualname

    def __new__(
        cls, targeted_underlying_type: type, skip_typechecking: bool = True
    ) -> (type, dict):
        targeted_underlying_type_name = targeted_underlying_type.__qualname__
        name, qualname = Iterator.generate_name(targeted_underlying_type_name)

        bases = (TemplateableIterator,)
        attrs, allowlist = generate_attrs_and_allowlist(
            TemplateableIterator, targeted_underlying_type, skip_typechecking, qualname
        )
        attrs["__qualname__"] = qualname
        attrs["__name__"] = name

        new_type = super().__new__(cls, name, bases, attrs)
        globals()[name] = new_type

        return new_type, allowlist
