import syft

if syft.dependency_check.tensorflow_available:
    from syft_tensorflow.hook import TensorFlowHook
    from syft_tensorflow.tensor import TensorFlowTensor

    setattr(syft, "TensorFlowHook", TensorFlowHook)
