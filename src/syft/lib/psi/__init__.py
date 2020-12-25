# third party
import openmined_psi

# syft relative
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals
from ..python import GenerateProtobufWrapper


def create_psi_ast() -> Globals:
    ast = Globals()

    modules = ["openmined_psi"]

    classes = [
        ("openmined_psi.client", "openmined_psi.client", openmined_psi.client),
        ("openmined_psi.server", "openmined_psi.server", openmined_psi.server),
        (
            "openmined_psi.ServerSetup",
            "openmined_psi.ServerSetup",
            openmined_psi.ServerSetup,
        ),
        (
            "openmined_psi.Request",
            "openmined_psi.Request",
            openmined_psi.Request,
        ),
        (
            "openmined_psi.Response",
            "openmined_psi.Response",
            openmined_psi.Response,
        ),
    ]

    GenerateProtobufWrapper(
        cls_pb=openmined_psi.ServerSetup,
        import_path="openmined_psi.ServerSetup",
    )
    GenerateProtobufWrapper(
        cls_pb=openmined_psi.Request,
        import_path="openmined_psi.Request",
    )
    GenerateProtobufWrapper(
        cls_pb=openmined_psi.Response,
        import_path="openmined_psi.Response",
    )

    methods = [
        ("openmined_psi.client.CreateWithNewKey", "openmined_psi.client"),
        ("openmined_psi.client.CreateRequest", "openmined_psi.Request"),
        ("openmined_psi.client.GetIntersection", "syft.lib.python.List"),
        ("openmined_psi.client.GetIntersectionSize", "syft.lib.python.Int"),
        ("openmined_psi.server.CreateWithNewKey", "openmined_psi.server"),
        (
            "openmined_psi.server.CreateSetupMessage",
            "openmined_psi.ServerSetup",
        ),
        ("openmined_psi.server.ProcessRequest", "openmined_psi.Response"),
    ]

    add_modules(ast, modules)
    add_classes(ast, classes)
    add_methods(ast, methods)

    for klass in ast.classes:
        klass.create_pointer_class()
        klass.create_send_method()
        klass.create_serialization_methods()
        klass.create_storable_object_attr_convenience_methods()

    return ast
