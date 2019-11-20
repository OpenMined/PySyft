import syft

if syft.dependency_check.tensorflow_available:
    from syft_tensorflow.hook import TensorFlowHook
    from syft_tensorflow.hook import hook_args
    from syft_tensorflow.syft_types import TensorFlowTensor

    setattr(syft, "TensorFlowHook", TensorFlowHook)
