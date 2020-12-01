# third party
import openmined_psi

# syft relative
from ...ast import add_classes
from ...ast import add_methods
from ...ast import add_modules
from ...ast.globals import Globals


def create_psi_ast() -> Globals:
    ast = Globals()

    modules = ["openmined_psi"]
    classes = [
        ("openmined_psi.client", "openmined_psi.client", openmined_psi.client),
        ("openmined_psi.server", "openmined_psi.server", openmined_psi.server),
        (
            "openmined_psi.proto_server_setup",
            "openmined_psi.proto_server_setup",
            openmined_psi.proto_server_setup,
        ),
        (
            "openmined_psi.proto_response",
            "openmined_psi.proto_response",
            openmined_psi.proto_response,
        ),
    ]

    methods = [
        ("openmined_psi.client.CreateWithNewKey", "openmined_psi.client"),
        ("openmined_psi.client.CreateRequest", "openmined_psi.proto_request"),
        ("openmined_psi.client.GetIntersection", "syft.lib.python.List"),
        ("openmined_psi.client.GetIntersectionSize", "syft.lib.python.Int"),
        ("openmined_psi.server.CreateWithNewKey", "openmined_psi.server"),
        ("openmined_psi.server.CreateSetupMessage", "openmined_psi.proto_server_setup"),
        ("openmined_psi.server.ProcessRequest", "openmined_psi.proto_response"),
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
