# third party
import numpy as np
import pyarrow as pa

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...proto.lib.numpy.array_pb2 import NumpyProto


def object2proto(obj: np.ndarray) -> NumpyProto:
    apache_arrow = pa.Tensor.from_numpy(obj=obj)
    sink = pa.BufferOutputStream()
    pa.ipc.write_tensor(apache_arrow, sink)
    buf = sink.getvalue().to_pybytes()
    proto = NumpyProto(data=buf)
    return proto


def proto2object(proto: NumpyProto) -> np.ndarray:
    reader = pa.BufferReader(proto.data)
    buf = reader.read_buffer()
    result = pa.ipc.read_tensor(buf)
    return result.to_numpy()


GenerateWrapper(
    wrapped_type=np.ndarray,
    import_path="numpy.ndarray",
    protobuf_scheme=NumpyProto,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
