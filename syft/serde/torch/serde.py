import torch

# Torch dtypes to string (and back) mappers
TORCH_DTYPE_STR = {
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.complex32: "complex32",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
    torch.bool: "bool",
    torch.qint8: "qint8",
    torch.quint8: "quint8",
    torch.qint32: "qint32",
    torch.bfloat16: "bfloat16",
}
TORCH_STR_DTYPE = {name: cls for cls, name in TORCH_DTYPE_STR.items()}