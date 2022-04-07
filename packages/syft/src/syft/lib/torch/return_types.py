# stdlib
import re
from typing import Any
from typing import Dict
from typing import List

# third party
from packaging import version
import torch

# relative
from ...core.common.serde.serializable import serializable
from ...proto.lib.torch.returntypes_pb2 import ReturnTypes as ReturnTypes_PB
from ..util import full_name_with_name
from .tensor_util import tensor_deserializer
from .tensor_util import tensor_serializer

# TODO: a better way. Loot at https://github.com/OpenMined/PySyft/issues/5249
module_type = type(torch)
torch.__dict__["return_types"] = module_type(name="return_types")
parent = torch.__dict__["return_types"]


def get_field_names(obj: Any) -> List[str]:
    return re.findall("\n(.*)=", str(obj))


def get_supported_types_fields() -> Dict[type, List]:
    supported_types = {}
    # A = torch.tensor([[1.0, 1, 1], [2, 3, 4], [3, 5, 2], [4, 2, 5], [5, 4, 3]])
    # B = torch.tensor([[-10.0, -3], [12, 14], [14, 12], [16, 16], [18, 16]])
    x = torch.Tensor([[1, 2], [1, 2]])
    s = torch.tensor(
        [[-0.1000, 0.1000, 0.2000], [0.2000, 0.3000, 0.4000], [0.0000, -0.3000, 0.5000]]
    )

    torch_version_ge_1d5d0 = version.parse(
        torch.__version__.split("+")[0]
    ) >= version.parse("1.5.0")

    if torch_version_ge_1d5d0:
        cummax = x.cummax(0)
        supported_types[type(cummax)] = get_field_names(cummax)

    if torch_version_ge_1d5d0:
        cummin = x.cummin(0)
        supported_types[type(cummin)] = get_field_names(cummin)

    # deprecated in torch==1.10.0
    # eig = x.eig(True)
    # supported_types[type(eig)] = get_field_names(eig)

    kthvalue = x.kthvalue(1)
    supported_types[type(kthvalue)] = get_field_names(kthvalue)

    # deprecated in torch==1.10.0
    # lstsq = A.lstsq(B)
    # supported_types[type(lstsq)] = get_field_names(lstsq)

    slogdet = x.slogdet()
    supported_types[type(slogdet)] = get_field_names(slogdet)

    # deprecated in torch==1.10.0
    # qr = x.qr()
    # supported_types[type(qr)] = get_field_names(qr)

    mode = x.mode()
    supported_types[type(mode)] = get_field_names(mode)

    # deprecated in torch==1.10.0
    # solve = s.solve(s)
    # supported_types[type(solve)] = get_field_names(solve)

    sort = s.sort()
    supported_types[type(sort)] = get_field_names(sort)

    # deprecated in torch==1.10.0
    # symeig = s.symeig()
    # supported_types[type(symeig)] = get_field_names(symeig)

    topk = s.topk(1)
    supported_types[type(topk)] = get_field_names(topk)

    # deprecated in torch==1.11.0
    # triangular_solve = s.triangular_solve(s)
    # supported_types[type(triangular_solve)] = get_field_names(triangular_solve)

    svd = s.svd()
    supported_types[type(svd)] = get_field_names(svd)

    geqrf = s.geqrf()
    supported_types[type(geqrf)] = get_field_names(geqrf)

    median = s.median(0)
    supported_types[type(median)] = get_field_names(median)

    max_t = s.max(0)
    supported_types[type(max_t)] = get_field_names(max_t)

    min_t = s.min(0)
    supported_types[type(min_t)] = get_field_names(min_t)

    return supported_types


def wrap_type(typ: type, fields: List[str]) -> None:
    def object2proto(obj: object) -> ReturnTypes_PB:
        proto = ReturnTypes_PB()

        obj_type = full_name_with_name(klass=obj._sy_serializable_wrapper_type)  # type: ignore
        proto.obj_type = obj_type

        values = [getattr(obj, field, None) for field in fields]
        proto.values.extend(list(map(lambda x: tensor_serializer(x), values)))

        return proto

    def proto2object(proto: ReturnTypes_PB) -> "typ":  # type: ignore
        values = [tensor_deserializer(x) for x in proto.values]
        return typ(values)

    serializable(generate_wrapper=True)(
        wrapped_type=typ,
        import_path=f"{typ.__module__}.{typ.__name__}",
        protobuf_scheme=ReturnTypes_PB,
        type_object2proto=object2proto,
        type_proto2object=proto2object,
    )

    # TODO: a better way. Loot at https://github.com/OpenMined/PySyft/issues/5249
    # add type to torch.return_types
    parent.__dict__[typ.__name__] = typ


types_fields = get_supported_types_fields()
for typ, fields in types_fields.items():
    wrap_type(typ, fields)
