# from __future__ import absolute_import
from typing import Union

from syft import dependency_check

framework_tensors = []
framework_shapes = []
hooks = []

if dependency_check.tensorflow_available:
    import tensorflow as tf
    from tensorflow.python.framework.ops import EagerTensor

    framework_tensors.append(EagerTensor)
    framework_shapes.append(tf.TensorShape)

if dependency_check.torch_available:
    import torch

    framework_tensors.append(torch.Tensor)
    framework_shapes.append(torch.Size)

FrameworkTensorType = None
for tensor_type in framework_tensors:
    if FrameworkTensorType is None:
        FrameworkTensorType = tensor_type
    else:
        FrameworkTensorType = Union[FrameworkTensorType, tensor_type]

FrameworkShapeType = None
for shape_type in framework_shapes:
    if FrameworkShapeType is None:
        FrameworkShapeType = shape_type
    else:
        FrameworkShapeType = Union[FrameworkShapeType, shape_type]

FrameworkTensor = tuple(tt for tt in framework_tensors)
FrameworkShape = tuple(sh for sh in framework_shapes)
