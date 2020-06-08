import torch
import crypten
from crypten.nn import onnx_converter as _onnx_converter


PACK_OTHER = -1
PACK_TORCH_TENSOR = 0
PACK_CRYPTEN_MODEL = 1


def pack_values(values):
    """Pack return values to be passed into a queue then sent over the wire.
    The main goal here is to be able to return torch tensors.
    Args:
        values: returned values from a function, can be a single object or a tuple.
    Return:
        A list of packed values.
    """

    packed_values = []
    # single value
    if not isinstance(values, tuple):
        packed_values.append(_pack_value(values))
    # multiple values
    else:
        for value in values:
            packed_values.append(_pack_value(value))
    return packed_values


def _pack_value(value):
    if isinstance(value, torch.Tensor):
        return (PACK_TORCH_TENSOR, value.tolist())

    elif isinstance(value, crypten.nn.Module):
        if value.encrypted:
            raise TypeError("Cannot pack an encrypted crypten model.")
        params = []
        for p in value.parameters():
            params.append(p.tolist())

        return (PACK_CRYPTEN_MODEL, params)

    return (PACK_OTHER, value)


def unpack_values(values, model=None):
    """Unpack return values that are fetched from the queue.
    Args:
        values: list of packed values.
        model: a crypten model to unpack parameters to.
    Return:
        A list of unpacked values.
    """

    unpacked_values = []
    for value in values:
        unpacked_values.append(_unpack_value(value, model))
        # single value
    if len(unpacked_values) == 1:
        return unpacked_values[0]
    # multiple values
    else:
        return tuple(unpacked_values)


def _unpack_value(value, model=None):
    value_type = value[0]
    if value_type == PACK_OTHER:
        return value[1]
    elif value_type == PACK_TORCH_TENSOR:
        return torch.tensor(value[1])
    elif value_type == PACK_CRYPTEN_MODEL:
        if model is None:
            raise TypeError("model can't be None when value is a crypten model.")
        params = value[1]
        for p, p_val in zip(model.parameters(), params):
            # Can't set value for leaf variable that requires grad
            with torch.no_grad():
                p.set_(torch.tensor(p_val))

        return model


def pytorch_to_onnx(pytorch_model, dummy_input):
    """Export a pytorch model to onnx.

    Args:
        pytorch_model: torch.nn.Module to export.
        dummy_input: example input that can be forwarded with the pytorch_model.

    Returns:
        bytes containing the exported pytorch model.
    """
    f = _onnx_converter._from_pytorch_to_bytes(pytorch_model, dummy_input)
    onnx_bytes = f.read()
    f.close()
    return onnx_bytes


def onnx_to_crypten(onnx_bytes):
    """Build a crypten model from onnx bytes.

    Args:
        onnx_bytes: bytes containing an exported pytorch model.

    Returns:
        crypten model.
    """
    return _onnx_converter.from_onnx(onnx_bytes)
