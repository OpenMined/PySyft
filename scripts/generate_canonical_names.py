# stdlib
import ast
import inspect
import os

# If variable is set, search for all serializable classes without canonical names
os.environ["SYFT_SEARCH_MISSING_CANONICAL_NAME"] = "true"

# syft absolute
# NOTE import has to happen after setting the environment variable

# relative
from ..serde.recursive import SYFT_CLASSES_MISSING_CANONICAL_NAME  # noqa: E402
from ..types.syft_object_registry import SyftObjectRegistry  # noqa: E402


class DecoratorFinder(ast.NodeVisitor):
    def __init__(self, class_name: str, decorator_name: str):
        self.class_name = class_name
        self.decorator_name = decorator_name
        self.decorator: ast.Call | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name == self.class_name:
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and getattr(decorator.func, "id", None) == self.decorator_name
                ):
                    self.decorator = decorator
        self.generic_visit(node)


def get_class_file_path(cls: type) -> str:
    return inspect.getfile(cls)


def get_decorator_with_lines(
    file_path: str, class_name: str, decorator_name: str
) -> tuple[ast.Call | None, int | None, int | None]:
    with open(file_path) as source:
        tree = ast.parse(source.read())

    finder = DecoratorFinder(class_name, decorator_name)
    finder.visit(tree)

    if finder.decorator:
        start_line = finder.decorator.lineno - 1
        end_line = (
            finder.decorator.end_lineno
            if hasattr(finder.decorator, "end_lineno")
            else finder.decorator.lineno
        )
        return finder.decorator, start_line, end_line
    return None, None, None


def add_canonical_name_version(decorator: ast.Call, class_name: str) -> ast.Call:
    new_decorator = decorator

    canonical_name_exists = any(
        kw.arg == "canonical_name" for kw in new_decorator.keywords
    )
    version_exists = any(kw.arg == "version" for kw in new_decorator.keywords)

    if not canonical_name_exists:
        new_decorator.keywords.append(
            ast.keyword(arg="canonical_name", value=ast.Constant(value=class_name))
        )
    if not version_exists:
        new_decorator.keywords.append(
            ast.keyword(arg="version", value=ast.Constant(value=1))
        )

    return ast.copy_location(new_decorator, decorator)


def update_decorator_for_cls(
    cls: type, existing_canonical_names: list[str]
) -> str | None:
    file_path = inspect.getfile(cls)
    class_name = cls.__name__

    decorator, start_line, end_line = get_decorator_with_lines(
        file_path, class_name, "serializable"
    )

    if decorator is None:
        print(
            f"{cls.__module__}: Could not find decorator for class {class_name}. Did not update canonical name."
        )
        return None
    if start_line is None or end_line is None:
        print(
            f"{cls.__module__}: No start/end lines for decorator in class {class_name}. Did not update canonical name."
        )
        return None

    if class_name in existing_canonical_names:
        print(
            f"{cls.__module__}: {class_name} is already a registered canonical name. Did not update canonical name."
        )
        return None

    new_decorator = add_canonical_name_version(decorator, class_name)
    new_decorator_code = ast.unparse(new_decorator).split("\n")
    new_decorator_code[0] = "@" + new_decorator_code[0]

    with open(file_path) as file:
        lines = file.readlines()

    lines[start_line:end_line] = [line + "\n" for line in new_decorator_code]

    with open(file_path, "w") as file:
        file.writelines(lines)

    print(f"Updated {cls.__module__}.{cls.__name__}")
    return class_name


def update_canonical_names():
    existing_cnames = list(SyftObjectRegistry.__object_serialization_registry__.keys())
    for cls in SYFT_CLASSES_MISSING_CANONICAL_NAME:
        new_name = update_decorator_for_cls(cls, existing_cnames)
        if new_name:
            existing_cnames.append(new_name)
