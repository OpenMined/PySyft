# third party
from pymbolic.primitives import Product

# syft relative
from .... import deserialize
from .... import serialize
from ....generate_wrapper import GenerateWrapper
from ...util import full_name_with_name
from ....proto.lib.pymbolic.product_pb2 import Product as Product_PB


def object2proto(obj: object) -> Product_PB:
    return Product_PB(children=[serialize(child) for child in obj.children])


def proto2object(proto: Product_PB) -> Product:
    return Product(children=tuple([deserialize(child) for child in proto.children]))


GenerateWrapper(
    wrapped_type=Product,
    import_path=full_name_with_name(Product),
    protobuf_scheme=Product_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
