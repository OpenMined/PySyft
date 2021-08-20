import torch as th

from syft.core.compression.compressed_tensor import CompressedTensor
from syft.core.compression.dgc_compressor import DgcCompressor

cmprsd_tensor = CompressedTensor(th.Tensor(list(range(10))))
comp_tensor = th.Tensor(list(range(10)))
ops_tensor = th.Tensor(list(range(11, 21)))


def test_compressed_tensor_operation() -> None:
    res = cmprsd_tensor + ops_tensor
    expected_res = th.Tensor(list(range(10))) + ops_tensor

    assert th.equal(expected_res, res)


def test_compressed_tensor_clone() -> None:
    res = cmprsd_tensor.clone()

    assert th.equal(res, cmprsd_tensor)


def test_compressed_tensor_serde() -> None:
    proto = cmprsd_tensor._object2proto()
    res_tensor = CompressedTensor._proto2object(proto)
    res_cmprsd = CompressedTensor._proto2object(proto, return_compressed=True)

    assert th.equal(res_tensor, cmprsd_tensor)
    assert th.equal(res_cmprsd, cmprsd_tensor)
    assert isinstance(res_tensor, th.Tensor)
    assert isinstance(res_cmprsd, th.Tensor)
    assert not isinstance(res_tensor, CompressedTensor)
    assert isinstance(res_cmprsd, CompressedTensor)


def test_compressed_tensor_compression() -> None:
    cmprsd_tensor.compress_more(DgcCompressor)
    res = cmprsd_tensor.decompress()

    assert th.equal(res, comp_tensor)