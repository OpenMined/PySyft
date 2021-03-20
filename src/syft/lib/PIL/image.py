# third party
import PIL
import numpy as np
import torch
import torchvision

# syft relative
from ...generate_wrapper import GenerateWrapper
from ...proto.lib.torch.tensor_pb2 import TensorData
from ...lib.torch.tensor_util import protobuf_tensor_serializer
from ...lib.torch.tensor_util import protobuf_tensor_deserializer


def object2proto(obj: PIL.Image.Image) -> TensorData:
    image_tensor = torch.Tensor(np.array(obj))
    tensor_proto = protobuf_tensor_serializer(image_tensor)

    return tensor_proto


def proto2object(proto: TensorData) -> PIL.Image.Image:
    image_tensor = protobuf_tensor_deserializer(proto)
    image_obj = torchvision.transforms.functional.to_pil_image(image_tensor)
    return image_obj


GenerateWrapper(
    wrapped_type=PIL.Image.Image,
    import_path="PIL.Image.Image",
    protobuf_scheme=TensorData,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
