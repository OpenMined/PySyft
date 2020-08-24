from ..ast.globals import Globals

# from syft.lib.numpy.__init__ import create_numpy_ast
from syft.lib.torch.__init__ import create_torch_ast
from syft.lib.builtins.__init__ import create_builtins_ast


# now we need to load the relevant frameworks onto the node
def create_lib_ast() -> Globals:

    torch_ast = create_torch_ast()
    # numpy_ast = create_numpy_ast()

    lib_ast = Globals()
    lib_ast.add_attr(attr_name="torch", attr=torch_ast.attrs["torch"])
    lib_ast.add_attr(attr_name="builtins", attr=torch_ast.attrs["builtins"])

    # lib_ast.add_attr(attr_name="numpy", attr=numpy_ast.attrs["numpy"])

    return lib_ast


# constructor: copyType = create_lib_ast
lib_ast = create_lib_ast()
lib_ast._copy = create_lib_ast
