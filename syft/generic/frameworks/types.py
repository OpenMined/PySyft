from typing import Union

from syft import dependency_check

framework_packages = {}

framework_tensors = []
framework_shapes = []
framework_layer_modules = []

if dependency_check.tensorflow_available:
    import tensorflow as tf
    from tensorflow.python.framework.ops import EagerTensor
    from tensorflow.python.ops.resource_variable_ops import ResourceVariable

    framework_packages["tensorflow"] = tf

    framework_tensors.append(EagerTensor)
    framework_tensors.append(ResourceVariable)
    framework_shapes.append(tf.TensorShape)

    framework_layer_modules.append(tf.Module)

if dependency_check.torch_available:
    import torch

    framework_packages["torch"] = torch

    framework_tensors.append(torch.Tensor)
    framework_tensors.append(torch.nn.Parameter)
    framework_shapes.append(torch.Size)

    framework_layer_module = torch.nn.Module
    framework_layer_module.named_tensors = torch.nn.Module.named_parameters
    framework_layer_modules.append(framework_layer_module)


if dependency_check.crypten_available:
    import crypten

    framework_packages["crypten"] = crypten
    framework_tensors.append(crypten.mpc.MPCTensor)
    framework_tensors.append(crypten.nn.Module)


framework_tensors = tuple(framework_tensors)
FrameworkTensorType = Union[framework_tensors]
FrameworkTensor = framework_tensors

framework_shapes = tuple(framework_shapes)
FrameworkShapeType = Union[framework_shapes]
FrameworkShape = framework_shapes

framework_layer_modules = tuple(framework_layer_modules)
FrameworkLayerModuleType = Union[framework_layer_modules]
FrameworkLayerModule = framework_layer_modules
