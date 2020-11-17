# syft relative
from ..ast.globals import Globals
from ..lib.python import create_python_ast
from ..lib.torch import create_torch_ast
from ..lib.torchvision import create_torchvision_ast
from .misc import create_union_ast

sympc_available = True

try:
    # syft relative
    from ..lib.sympc import create_sympc_ast
except ImportError:
    sympc_available = False


# now we need to load the relevant frameworks onto the node
def create_lib_ast() -> Globals:
    lib_ast = Globals()

    python_ast = create_python_ast()
    lib_ast.add_attr(attr_name="syft", attr=python_ast.attrs["syft"])

    torch_ast = create_torch_ast()
    lib_ast.add_attr(attr_name="torch", attr=torch_ast.attrs["torch"])

    torchvision_ast = create_torchvision_ast()
    lib_ast.add_attr(attr_name="torchvision", attr=torchvision_ast.attrs["torchvision"])
    # let the misc creation be always the last, as it needs the full ast solved
    # to properly generated unions
    misc_ast = getattr(getattr(create_union_ast(lib_ast), "syft"), "lib")
    misc_root = getattr(getattr(lib_ast, "syft"), "lib")
    # lib_ast.add_attr(attr_name="numpy", attr=numpy_ast.attrs["numpy"])

    if sympc_available:
        sympc_ast = create_sympc_ast()
        lib_ast.add_attr(attr_name="sympc", attr=sympc_ast.attrs["sympc"])

    misc_root.add_attr(attr_name="misc", attr=misc_ast.attrs["misc"])

    return lib_ast


# constructor: copyType = create_lib_ast
lib_ast = create_lib_ast()
lib_ast._copy = create_lib_ast
