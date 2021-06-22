# third party
import PIL
import numpy as np
import torch
import torchvision

# syft relative
from ... import deserialize
from ... import serialize
from ...generate_wrapper import GenerateWrapper
from ...proto.lib.PIL.image_pb2 import Image as Image_PB


def object2proto(obj: PIL.Image.Image) -> Image_PB:
    image_tensor = torch.Tensor(np.array(obj))
    tensor_proto = serialize(image_tensor, to_proto=True)
    return Image_PB(data=tensor_proto)


def proto2object(proto: Image_PB) -> PIL.Image.Image:
    image_tensor = deserialize(proto.data, from_proto=True)
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.permute(2, 0, 1)
    image_obj = torchvision.transforms.functional.to_pil_image(image_tensor)
    return image_obj


GenerateWrapper(
    wrapped_type=PIL.Image.Image,
    import_path="PIL.Image.Image",
    protobuf_scheme=Image_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
