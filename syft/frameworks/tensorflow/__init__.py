import syft

if syft.dependency_check.tensorflow_available:
    from syft_tensorflow.hook import TensorFlowHook
    from syft_tensorflow.hook import hook_args  # noqa: F401
    from syft_tensorflow.syft_types import TensorFlowTensor  # noqa: F401

    setattr(syft, "TensorFlowHook", TensorFlowHook)
