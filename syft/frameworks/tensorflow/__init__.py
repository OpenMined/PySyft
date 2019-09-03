import syft

if syft.dependency_check.tensorflow_available:
    from syft_tensorflow.hook import TensorFlowHook
    from syft_tensorflow.serde import MAP_TF_SIMPLIFIERS_AND_DETAILERS
    from syft_tensorflow.tensor import TensorFlowTensor

    setattr(syft, "TensorFlowHook", TensorFlowHook)
