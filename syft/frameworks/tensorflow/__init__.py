import syft

if syft.dependency_check.tensorflow_available:
    from syft_tensorflow import bind_tensorflow
    from syft_tensorflow.serde import MAP_TF_SIMPLIFIERS_AND_DETAILERS
    from syft_tensorflow.tensor import TensorFlowTensor

    bind_tensorflow(syft)
