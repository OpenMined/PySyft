# stdlib
from collections import defaultdict
import sys
from typing import Any as TypeAny
from typing import Callable
from typing import Dict
from typing import KeysView
from typing import List as TypeList
from typing import Set

# syft relative
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast import globals
from .union import lazy_pairing


def get_allowed_functions(
    lib_ast: globals.Globals, union_types: TypeList[str]
) -> Dict[str, bool]:
    """
    This function generates a set of functions that can go into a union type.

    A function has to meet the following requirements to be present on a union type:
        1. If it's present on all Class attributes associated with the union types
        on the ast, add it.
        2. If it's not present on all Class attributes associated with the union
        types, check if they exist on the original type functions list. If they
        do exist, drop it, if not, add it.

    Args:
        lib_ast (Globals): the AST on which we want to generate the union pointer.
        union_types (List[str]): the qualnames of the types on which we want a union.

    Returns:
        allowed_functions (dict): The keys of the dict are function names (str)
        and the values are Bool (if they are allowed or not).
    """

    allowed_functions: Dict[str, bool] = defaultdict(lambda: True)

    def solve_ast_type_functions(path: str) -> KeysView:
        root = lib_ast
        for path_element in path.split("."):
            root = getattr(root, path_element)

        return root.attrs.keys()

    def solve_real_type_functions(path: str) -> Set[str]:
        parts = path.split(".")
        klass_name = parts[-1]
        return set(dir(getattr(sys.modules[".".join(parts[:-1])], klass_name)))

    for union_type in union_types:
        real_type_function_set = solve_real_type_functions(union_type)
        ast_type_function_set = solve_ast_type_functions(union_type)

        rejected_function_set = real_type_function_set - ast_type_function_set

        for accepted_function in ast_type_function_set:
            allowed_functions[accepted_function] &= True

        for rejected_function in rejected_function_set:
            allowed_functions[rejected_function] = False

    return allowed_functions


def create_union_ast(lib_ast: globals.Globals) -> globals.Globals:
    ast = globals.Globals()

    modules = ["syft", "syft.lib", "syft.lib.misc", "syft.lib.misc.union"]

    classes = []
    methods = []

    for klass in lazy_pairing.keys():
        classes.append(
            (
                f"syft.lib.misc.union.{klass.__name__}",
                f"syft.lib.misc.union.{klass.__name__}",
                klass,
            )
        )
        union_types = lazy_pairing[klass]
        allowed_functions = get_allowed_functions(lib_ast, union_types)

        for target_method, allowed in allowed_functions.items():
            if not allowed:
                continue

            def generate_func(target_method: str) -> Callable:
                def func(self: TypeAny, *args: TypeAny, **kwargs: TypeAny) -> TypeAny:
                    func = getattr(self, target_method, None)
                    if func:
                        return func(*args, **kwargs)
                    else:
                        raise ValueError(
                            f"Can't call {target_method} on {klass} with the instance type of {type(self)}"
                        )

                return func

            setattr(klass, target_method, generate_func(target_method))

            methods.append(
                (
                    f"syft.lib.misc.union.{klass.__name__}.{target_method}",
                    "syft.lib.python.Any",
                )
            )

    add_modules(ast, modules)
    add_classes(ast, classes)
    add_methods(ast, methods)

    for ast_klass in ast.classes:
        ast_klass.create_pointer_class()
        ast_klass.create_send_method()
        ast_klass.create_serialization_methods()
        ast_klass.create_storable_object_attr_convenience_methods()

    return ast
