# stdlib
from typing import Dict
from typing import Union

# relative
from ..misc.union import UnionGenerator

allowlist: Dict[str, Union[str, Dict[str, str]]] = {}  # (path: str, return_type:type)
dynamic_allowlist: Dict[str, str] = {}
# --------------------------------------------------------------------------------------
# SECTION - Tensor methods which are intentionally disabled
# --------------------------------------------------------------------------------------

# SECTION - Tensor methods which are insecure
# allowlist["torch.Tensor.__array__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__array_priority__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__array_wrap__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__bool__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__class__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__contains__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__deepcopy__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__delattr__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__delitem__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__dict__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__dir__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__doc__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__format__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__getattribute__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__init__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__init_subclass__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__len__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST YET - talk to TRASK
# allowlist["torch.Tensor.__module__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__new__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__reduce__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__reduce_ex__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__repr__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__setattr__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__setstate__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__sizeof__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST YET - talk to TRASK
# allowlist["torch.Tensor.__str__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__subclasshook__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.__weakref__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor._backward_hooks"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor._base"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor._cdata"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor._coalesced_"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor._dimI"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor._dimV"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor._grad"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor._grad_fn"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.grad_fn"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor._indices"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor._is_view"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor._make_subclass"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor._nnz"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor._update_names"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor._values"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.dtype"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST YET - talk to TRASK
# allowlist["torch.Tensor.has_names"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.record_stream"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST#
# allowlist["torch.Tensor.register_hook"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.share_memory_"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.storage"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.storage_offset"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.storage_type"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.where"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST

# SECTION - Torch functions which are insecure
# allowlist["torch.where"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.storage"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST

# SECTION - Tensor methods which have serde issues
# allowlist["torch.Tensor.to_dense"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.to_mkldnn"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.to_sparse"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST

# --------------------------------------------------------------------------------------
# SECTION - Tensor methods which are tested
# --------------------------------------------------------------------------------------


# SECTION - The capital Tensor constructors
# allowlist["torch.__version__"] = "syft.lib.python.String"
# allowlist["torch.Tensor.retain_graph"] = "syft.lib.python.Bool"

allowlist["torch.Tensor"] = "torch.Tensor"
allowlist["torch.BFloat16Tensor"] = "torch.Tensor"
allowlist["torch.BoolTensor"] = "torch.Tensor"
allowlist["torch.ByteTensor"] = "torch.Tensor"
allowlist["torch.CharTensor"] = "torch.Tensor"
allowlist["torch.DoubleTensor"] = "torch.Tensor"
allowlist["torch.FloatTensor"] = "torch.Tensor"
allowlist["torch.HalfTensor"] = "torch.Tensor"
allowlist["torch.IntTensor"] = "torch.Tensor"
allowlist["torch.LongTensor"] = "torch.Tensor"
allowlist["torch.ShortTensor"] = "torch.Tensor"

# SECTION - Tensor methods
allowlist["torch.Tensor.__abs__"] = "torch.Tensor"
allowlist["torch.Tensor.__add__"] = "torch.Tensor"
allowlist["torch.Tensor.__and__"] = "torch.Tensor"
allowlist["torch.Tensor.__eq__"] = "torch.Tensor"
allowlist["torch.Tensor.__float__"] = "syft.lib.python.Float"
allowlist["torch.Tensor.__ge__"] = "torch.Tensor"
allowlist["torch.Tensor.__getitem__"] = "torch.Tensor"
allowlist["torch.Tensor.__gt__"] = "torch.Tensor"
allowlist["torch.Tensor.__hash__"] = "syft.lib.python.Int"
allowlist["torch.Tensor.__iadd__"] = "torch.Tensor"
allowlist["torch.Tensor.__iand__"] = "torch.Tensor"
allowlist["torch.Tensor.__idiv__"] = "torch.Tensor"
allowlist["torch.Tensor.__ilshift__"] = "torch.Tensor"
allowlist["torch.Tensor.__imul__"] = "torch.Tensor"
allowlist["torch.Tensor.__index__"] = "torch.Tensor"
allowlist["torch.Tensor.__int__"] = "syft.lib.python.Int"
allowlist["torch.Tensor.__invert__"] = "torch.Tensor"
allowlist["torch.Tensor.__ior__"] = "torch.Tensor"
allowlist["torch.Tensor.__ipow__"] = "torch.Tensor"  # none implemented in 1.5.1
allowlist["torch.Tensor.__irshift__"] = "torch.Tensor"
allowlist["torch.Tensor.__isub__"] = "torch.Tensor"
allowlist["torch.Tensor.__itruediv__"] = "torch.Tensor"
allowlist["torch.Tensor.__ixor__"] = "torch.Tensor"
allowlist["torch.Tensor.__le__"] = "torch.Tensor"
allowlist["torch.Tensor.__long__"] = "syft.lib.python.Int"
allowlist["torch.Tensor.__lshift__"] = "torch.Tensor"
allowlist["torch.Tensor.__lt__"] = "torch.Tensor"
allowlist["torch.Tensor.__matmul__"] = "torch.Tensor"
allowlist["torch.Tensor.__mod__"] = "torch.Tensor"
allowlist["torch.Tensor.__mul__"] = "torch.Tensor"
allowlist["torch.Tensor.__ne__"] = "torch.Tensor"
allowlist["torch.Tensor.__neg__"] = "torch.Tensor"
allowlist["torch.Tensor.__nonzero__"] = "torch.Tensor"
allowlist["torch.Tensor.__or__"] = "torch.Tensor"
allowlist["torch.Tensor.__pow__"] = "torch.Tensor"
allowlist["torch.Tensor.__radd__"] = "torch.Tensor"
allowlist["torch.Tensor.__rdiv__"] = "torch.Tensor"
allowlist["torch.Tensor.__reversed__"] = "torch.Tensor"
allowlist["torch.Tensor.__rmul__"] = "torch.Tensor"
allowlist["torch.Tensor.__rpow__"] = "torch.Tensor"
allowlist["torch.Tensor.__rshift__"] = "torch.Tensor"
allowlist["torch.Tensor.__rsub__"] = "torch.Tensor"
allowlist["torch.Tensor.__rtruediv__"] = "torch.Tensor"
allowlist["torch.Tensor.__setitem__"] = "syft.lib.python._SyNone"
allowlist["torch.Tensor.__sub__"] = "torch.Tensor"
allowlist["torch.Tensor.__truediv__"] = "torch.Tensor"
allowlist["torch.Tensor.__xor__"] = "torch.Tensor"
allowlist["torch.Tensor._version"] = "syft.lib.python.Int"
allowlist["torch.Tensor.abs_"] = "torch.Tensor"
allowlist["torch.Tensor.abs"] = "torch.Tensor"
allowlist["torch.Tensor.acos_"] = "torch.Tensor"
allowlist["torch.Tensor.acos"] = "torch.Tensor"
allowlist["torch.Tensor.add_"] = "torch.Tensor"
allowlist["torch.Tensor.add"] = "torch.Tensor"
allowlist["torch.Tensor.addbmm_"] = "torch.Tensor"
allowlist["torch.Tensor.addbmm"] = "torch.Tensor"
allowlist["torch.Tensor.addcdiv_"] = "torch.Tensor"
allowlist["torch.Tensor.addcdiv"] = "torch.Tensor"
allowlist["torch.Tensor.addcmul_"] = "torch.Tensor"
allowlist["torch.Tensor.addcmul"] = "torch.Tensor"
allowlist["torch.Tensor.addmm_"] = "torch.Tensor"
allowlist["torch.Tensor.addmm"] = "torch.Tensor"
allowlist["torch.Tensor.addmv_"] = "torch.Tensor"
allowlist["torch.Tensor.addmv"] = "torch.Tensor"
allowlist["torch.Tensor.addr_"] = "torch.Tensor"
allowlist["torch.Tensor.addr"] = "torch.Tensor"
allowlist["torch.Tensor.all"] = "torch.Tensor"
allowlist["torch.Tensor.allclose"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.angle"] = "torch.Tensor"
allowlist["torch.Tensor.any"] = "torch.Tensor"
allowlist["torch.Tensor.argmax"] = "torch.Tensor"
allowlist["torch.Tensor.argmin"] = "torch.Tensor"
allowlist["torch.Tensor.argsort"] = "torch.Tensor"
allowlist["torch.Tensor.as_strided_"] = "torch.Tensor"
allowlist["torch.Tensor.as_strided"] = "torch.Tensor"
allowlist["torch.Tensor.asin_"] = "torch.Tensor"
allowlist["torch.Tensor.asin"] = "torch.Tensor"
allowlist["torch.Tensor.atan_"] = "torch.Tensor"
allowlist["torch.Tensor.atan"] = "torch.Tensor"
allowlist["torch.Tensor.atan2_"] = "torch.Tensor"
allowlist["torch.Tensor.atan2"] = "torch.Tensor"
allowlist["torch.Tensor.backward"] = "syft.lib.python._SyNone"
allowlist["torch.Tensor.baddbmm_"] = "torch.Tensor"
allowlist["torch.Tensor.baddbmm"] = "torch.Tensor"
allowlist["torch.Tensor.bernoulli_"] = "torch.Tensor"
allowlist["torch.Tensor.bernoulli"] = "torch.Tensor"
allowlist["torch.Tensor.bfloat16"] = "torch.Tensor"
allowlist["torch.Tensor.bincount"] = "torch.Tensor"
allowlist["torch.Tensor.bitwise_not_"] = "torch.Tensor"
allowlist["torch.Tensor.bitwise_not"] = "torch.Tensor"
allowlist["torch.Tensor.bitwise_xor_"] = "torch.Tensor"
allowlist["torch.Tensor.bitwise_xor"] = "torch.Tensor"
allowlist["torch.Tensor.bmm"] = "torch.Tensor"
allowlist["torch.Tensor.bool"] = "torch.Tensor"
allowlist["torch.Tensor.byte"] = "torch.Tensor"
allowlist["torch.Tensor.cauchy_"] = "torch.Tensor"
allowlist["torch.Tensor.ceil_"] = "torch.Tensor"
allowlist["torch.Tensor.ceil"] = "torch.Tensor"
allowlist["torch.Tensor.char"] = "torch.Tensor"
allowlist["torch.Tensor.cholesky_inverse"] = "torch.Tensor"
allowlist["torch.Tensor.cholesky_solve"] = "torch.Tensor"
allowlist["torch.Tensor.cholesky"] = "torch.Tensor"
allowlist["torch.Tensor.chunk"] = "syft.lib.python.List"  # Tuple not List
allowlist["torch.Tensor.clamp_"] = "torch.Tensor"
allowlist["torch.Tensor.clamp_max_"] = "torch.Tensor"
allowlist["torch.Tensor.clamp_max"] = "torch.Tensor"
allowlist["torch.Tensor.clamp_min_"] = "torch.Tensor"
allowlist["torch.Tensor.clamp_min"] = "torch.Tensor"
allowlist["torch.Tensor.clamp"] = "torch.Tensor"
allowlist["torch.Tensor.clone"] = "torch.Tensor"
allowlist["torch.Tensor.coalesce"] = "torch.Tensor"
allowlist["torch.Tensor.conj"] = "torch.Tensor"
allowlist["torch.Tensor.contiguous"] = "torch.Tensor"
allowlist["torch.Tensor.copy_"] = "torch.Tensor"
allowlist["torch.Tensor.cos_"] = "torch.Tensor"
allowlist["torch.Tensor.cos"] = "torch.Tensor"
allowlist["torch.Tensor.cosh_"] = "torch.Tensor"
allowlist["torch.Tensor.cosh"] = "torch.Tensor"
allowlist["torch.Tensor.cpu"] = "torch.Tensor"
allowlist["torch.Tensor.cross"] = "torch.Tensor"
allowlist["torch.Tensor.cuda"] = "torch.Tensor"
allowlist["torch.return_types"] = "torch.return_types"
allowlist["torch.return_types.cummax"] = {
    "return_type": "torch.return_types.cummax",
    "min_version": "1.5.0",
}
allowlist["torch.return_types.cummin"] = {
    "return_type": "torch.return_types.cummin",
    "min_version": "1.5.0",
}
# allowlist["torch.return_types.eig"] = "torch.return_types.eig" # deprecated in torch==1.10.0
allowlist["torch.return_types.kthvalue"] = "torch.return_types.kthvalue"
# allowlist["torch.return_types.lstsq"] = "torch.return_types.lstsq" # deprecated in torch==1.10.0
allowlist["torch.return_types.slogdet"] = "torch.return_types.slogdet"
# allowlist["torch.return_types.qr"] = "torch.return_types.qr" # deprecated in torch==1.10.0
allowlist["torch.return_types.mode"] = "torch.return_types.mode"
# allowlist["torch.return_types.solve"] = "torch.return_types.solve" # deprecated in torch==1.10.0
allowlist["torch.return_types.sort"] = "torch.return_types.sort"
# allowlist["torch.return_types.symeig"] = "torch.return_types.symeig" # deprecated in torch==1.10.0
allowlist["torch.return_types.topk"] = "torch.return_types.topk"
# allowlist["torch.return_types.triangular_solve"] = "torch.return_types.triangular_solve" # deprecated in torch==1.11.0
allowlist["torch.return_types.svd"] = "torch.return_types.svd"
allowlist["torch.return_types.geqrf"] = "torch.return_types.geqrf"
allowlist["torch.return_types.median"] = "torch.return_types.median"
allowlist["torch.return_types.max"] = "torch.return_types.max"
allowlist["torch.return_types.min"] = "torch.return_types.min"
allowlist["torch.Tensor.cummax"] = {
    "return_type": "torch.return_types.cummax",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.cummin"] = {
    "return_type": "torch.return_types.cummin",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.cumprod"] = "torch.Tensor"
allowlist["torch.Tensor.cumsum"] = "torch.Tensor"
allowlist["torch.Tensor.data_ptr"] = "syft.lib.python.Int"
allowlist["torch.Tensor.data"] = "torch.Tensor"
allowlist["torch.Tensor.dense_dim"] = "torch.Tensor"
allowlist["torch.Tensor.dequantize"] = "torch.Tensor"
allowlist["torch.Tensor.det"] = "torch.Tensor"
allowlist["torch.Tensor.detach"] = "torch.Tensor"
allowlist["torch.Tensor.diag_embed"] = "torch.Tensor"
allowlist["torch.Tensor.diag"] = "torch.Tensor"
allowlist["torch.Tensor.diagflat"] = "torch.Tensor"
allowlist["torch.Tensor.diagonal"] = "torch.Tensor"
allowlist["torch.Tensor.digamma_"] = "torch.Tensor"
allowlist["torch.Tensor.digamma"] = "torch.Tensor"
allowlist["torch.Tensor.dim"] = "torch.Tensor"
allowlist["torch.Tensor.dist"] = "torch.Tensor"
allowlist["torch.Tensor.dot"] = "torch.Tensor"
allowlist["torch.Tensor.double"] = "torch.Tensor"
# allowlist["torch.Tensor.eig"] = "torch.return_types.eig" # deprecated in torch==1.10.0
allowlist["torch.Tensor.element_size"] = "syft.lib.python.Int"
allowlist["torch.Tensor.eq_"] = "torch.Tensor"
allowlist["torch.Tensor.eq"] = "torch.Tensor"
allowlist["torch.Tensor.equal"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.erf_"] = "torch.Tensor"
allowlist["torch.Tensor.erf"] = "torch.Tensor"
allowlist["torch.Tensor.erfc_"] = "torch.Tensor"
allowlist["torch.Tensor.erfc"] = "torch.Tensor"
allowlist["torch.Tensor.erfinv_"] = "torch.Tensor"
allowlist["torch.Tensor.erfinv"] = "torch.Tensor"
allowlist["torch.Tensor.exp_"] = "torch.Tensor"
allowlist["torch.Tensor.exp"] = "torch.Tensor"
allowlist["torch.Tensor.expand_as"] = "torch.Tensor"
allowlist["torch.Tensor.expand"] = "torch.Tensor"
allowlist["torch.Tensor.expm1_"] = "torch.Tensor"
allowlist["torch.Tensor.expm1"] = "torch.Tensor"
allowlist["torch.Tensor.exponential_"] = "torch.Tensor"
allowlist["torch.Tensor.fill_"] = "torch.Tensor"
allowlist["torch.Tensor.fill_diagonal_"] = "torch.Tensor"
allowlist["torch.Tensor.flatten"] = "torch.Tensor"
allowlist["torch.Tensor.flip"] = "torch.Tensor"
allowlist["torch.Tensor.float"] = "torch.Tensor"
allowlist["torch.Tensor.floor_"] = "torch.Tensor"
allowlist["torch.Tensor.floor"] = "torch.Tensor"
allowlist["torch.Tensor.fmod_"] = "torch.Tensor"
allowlist["torch.Tensor.fmod"] = "torch.Tensor"
allowlist["torch.Tensor.frac_"] = "torch.Tensor"
allowlist["torch.Tensor.frac"] = "torch.Tensor"
allowlist["torch.Tensor.gather"] = "torch.Tensor"
allowlist["torch.Tensor.ge_"] = "torch.Tensor"
allowlist["torch.Tensor.ge"] = "torch.Tensor"
allowlist["torch.Tensor.geometric_"] = "torch.Tensor"
allowlist["torch.Tensor.geqrf"] = "torch.return_types.geqrf"
allowlist["torch.Tensor.ger"] = "torch.Tensor"
allowlist["torch.Tensor.get_device"] = "syft.lib.python.Int"
allowlist["torch.Tensor.gt_"] = "torch.Tensor"
allowlist["torch.Tensor.gt"] = "torch.Tensor"
allowlist["torch.Tensor.half"] = "torch.Tensor"
allowlist["torch.Tensor.hardshrink"] = "torch.Tensor"
allowlist["torch.Tensor.histc"] = "torch.Tensor"
allowlist["torch.Tensor.index_add_"] = "torch.Tensor"
allowlist["torch.Tensor.index_add"] = "torch.Tensor"
allowlist["torch.Tensor.index_copy_"] = "torch.Tensor"
allowlist["torch.Tensor.index_copy"] = "torch.Tensor"
allowlist["torch.Tensor.index_fill_"] = "torch.Tensor"
allowlist["torch.Tensor.index_fill"] = "torch.Tensor"
allowlist["torch.Tensor.index_put_"] = "torch.Tensor"
allowlist["torch.Tensor.index_put"] = "torch.Tensor"
allowlist["torch.Tensor.index_select"] = "torch.Tensor"
allowlist["torch.Tensor.indices"] = "torch.Tensor"
allowlist["torch.Tensor.int_repr"] = "torch.Tensor"
allowlist["torch.Tensor.int"] = "torch.Tensor"
allowlist["torch.Tensor.inverse"] = "torch.Tensor"
allowlist["torch.Tensor.is_coalesced"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_complex"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_contiguous"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_cuda"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_distributed"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_floating_point"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_leaf"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_mkldnn"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_nonzero"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_pinned"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_quantized"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_same_size"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_set_to"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_shared"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_signed"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_sparse"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.isclose"] = "torch.Tensor"
allowlist["torch.Tensor.item"] = UnionGenerator[
    "syft.lib.python.Int", "syft.lib.python.Float", "syft.lib.python.Bool"
]
allowlist["torch.Tensor.kthvalue"] = "torch.return_types.kthvalue"
allowlist["torch.Tensor.le_"] = "torch.Tensor"
allowlist["torch.Tensor.le"] = "torch.Tensor"
allowlist["torch.Tensor.lerp_"] = "torch.Tensor"
allowlist["torch.Tensor.lerp"] = "torch.Tensor"
allowlist["torch.Tensor.lgamma_"] = "torch.Tensor"
allowlist["torch.Tensor.lgamma"] = "torch.Tensor"
allowlist["torch.Tensor.log_"] = "torch.Tensor"
allowlist["torch.Tensor.log_normal_"] = "torch.Tensor"
allowlist["torch.Tensor.log_softmax"] = "torch.Tensor"
allowlist["torch.Tensor.log"] = "torch.Tensor"
allowlist["torch.Tensor.log10_"] = "torch.Tensor"
allowlist["torch.Tensor.log10"] = "torch.Tensor"
allowlist["torch.Tensor.log1p_"] = "torch.Tensor"
allowlist["torch.Tensor.log1p"] = "torch.Tensor"
allowlist["torch.Tensor.log2_"] = "torch.Tensor"
allowlist["torch.Tensor.log2"] = "torch.Tensor"
allowlist["torch.Tensor.logdet"] = "torch.Tensor"
allowlist["torch.Tensor.logical_not_"] = "torch.Tensor"
allowlist["torch.Tensor.logical_not"] = "torch.Tensor"
allowlist["torch.Tensor.logical_xor_"] = "torch.Tensor"
allowlist["torch.Tensor.logical_xor"] = "torch.Tensor"
allowlist["torch.Tensor.logsumexp"] = "torch.Tensor"
allowlist["torch.Tensor.long"] = "torch.Tensor"
# allowlist["torch.Tensor.lstsq"] = "torch.return_types.lstsq" # deprecated in torch==1.10.0
allowlist["torch.Tensor.lt_"] = "torch.Tensor"
allowlist["torch.Tensor.lt"] = "torch.Tensor"
allowlist["torch.Tensor.lu_solve"] = "torch.Tensor"
allowlist["torch.Tensor.lu"] = "syft.lib.python.List"  # Tuple not List
allowlist["torch.Tensor.masked_fill_"] = "torch.Tensor"
allowlist["torch.Tensor.masked_fill"] = "torch.Tensor"
allowlist["torch.Tensor.masked_scatter_"] = "torch.Tensor"
allowlist["torch.Tensor.masked_scatter"] = "torch.Tensor"
allowlist["torch.Tensor.masked_select"] = "torch.Tensor"
allowlist["torch.Tensor.matmul"] = "torch.Tensor"
allowlist["torch.Tensor.matrix_power"] = "torch.Tensor"
allowlist["torch.Tensor.max"] = UnionGenerator[
    "syft.lib.python.Bool",
    "syft.lib.python.Float",
    "syft.lib.python.Int",
    "torch.return_types.max",
]
allowlist["torch.Tensor.mean"] = "torch.Tensor"
allowlist["torch.Tensor.median"] = UnionGenerator[
    "syft.lib.python.Bool",
    "syft.lib.python.Float",
    "syft.lib.python.Int",
    "torch.return_types.median",
]
allowlist["torch.Tensor.min"] = UnionGenerator[
    "syft.lib.python.Bool",
    "syft.lib.python.Float",
    "syft.lib.python.Int",
    "torch.return_types.min",
]
allowlist["torch.Tensor.mm"] = "torch.Tensor"
allowlist["torch.Tensor.mode"] = "torch.return_types.mode"
allowlist["torch.Tensor.mul_"] = "torch.Tensor"
allowlist["torch.Tensor.mul"] = "torch.Tensor"
allowlist["torch.Tensor.multinomial"] = "torch.Tensor"
allowlist["torch.Tensor.mv"] = "torch.Tensor"
allowlist["torch.Tensor.mvlgamma_"] = "torch.Tensor"
allowlist["torch.Tensor.mvlgamma"] = "torch.Tensor"
allowlist["torch.Tensor.narrow_copy"] = "torch.Tensor"
allowlist["torch.Tensor.narrow"] = "torch.Tensor"
allowlist["torch.Tensor.ndim"] = "syft.lib.python.Int"
allowlist["torch.Tensor.ndimension"] = "syft.lib.python.Int"
allowlist["torch.Tensor.ne_"] = "torch.Tensor"
allowlist["torch.Tensor.ne"] = "torch.Tensor"
allowlist["torch.Tensor.neg_"] = "torch.Tensor"
allowlist["torch.Tensor.neg"] = "torch.Tensor"
allowlist["torch.Tensor.nelement"] = "syft.lib.python.Int"  # is this INSECURE???
allowlist["torch.Tensor.new_empty"] = "torch.Tensor"
allowlist["torch.Tensor.new_full"] = "torch.Tensor"
allowlist["torch.Tensor.new_ones"] = "torch.Tensor"
allowlist["torch.Tensor.new_tensor"] = "torch.Tensor"
allowlist["torch.Tensor.new_zeros"] = "torch.Tensor"
allowlist["torch.Tensor.new"] = "torch.Tensor"
allowlist["torch.Tensor.nonzero"] = "torch.Tensor"
allowlist["torch.Tensor.norm"] = "torch.Tensor"
allowlist["torch.Tensor.normal_"] = "torch.Tensor"
allowlist["torch.Tensor.numel"] = "syft.lib.python.Int"  # is this INSECURE???
# allowlist["torch.Tensor.orgqr"] = "torch.Tensor"  # wrong return type in 1.9.1
# allowlist["torch.Tensor.ormqr"] = "torch.Tensor"  # wrong return type in 1.9.1
allowlist["torch.Tensor.output_nr"] = "syft.lib.python.Int"
allowlist["torch.Tensor.permute"] = "torch.Tensor"
allowlist["torch.Tensor.pin_memory"] = "torch.Tensor"
allowlist["torch.Tensor.pinverse"] = "torch.Tensor"
allowlist["torch.Tensor.polygamma_"] = "torch.Tensor"
allowlist["torch.Tensor.polygamma"] = "torch.Tensor"
allowlist["torch.Tensor.pow_"] = "torch.Tensor"
allowlist["torch.Tensor.pow"] = "torch.Tensor"
allowlist["torch.Tensor.prelu"] = "torch.Tensor"
allowlist["torch.Tensor.prod"] = "torch.Tensor"
allowlist["torch.Tensor.put_"] = "torch.Tensor"
allowlist["torch.Tensor.q_per_channel_axis"] = "syft.lib.python.Int"
allowlist["torch.Tensor.q_per_channel_scales"] = "torch.Tensor"
allowlist["torch.Tensor.q_per_channel_zero_points"] = "torch.Tensor"
allowlist["torch.Tensor.q_scale"] = "syft.lib.python.Float"
allowlist["torch.Tensor.q_zero_point"] = "syft.lib.python.Int"
# allowlist["torch.Tensor.qr"] = "torch.return_types.qr" # deprecated in torch==1.10.0
allowlist["torch.Tensor.random_"] = "torch.Tensor"
allowlist["torch.Tensor.reciprocal_"] = "torch.Tensor"
allowlist["torch.Tensor.reciprocal"] = "torch.Tensor"
allowlist["torch.Tensor.relu_"] = "torch.Tensor"
allowlist["torch.Tensor.relu"] = "torch.Tensor"
allowlist["torch.Tensor.renorm_"] = "torch.Tensor"
allowlist["torch.Tensor.renorm"] = "torch.Tensor"
allowlist["torch.Tensor.repeat_interleave"] = "torch.Tensor"
allowlist["torch.Tensor.repeat"] = "torch.Tensor"
allowlist["torch.Tensor.requires_grad_"] = "torch.Tensor"
allowlist["torch.Tensor.requires_grad"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.reshape_as"] = "torch.Tensor"
allowlist["torch.Tensor.reshape"] = "torch.Tensor"
allowlist["torch.Tensor.resize_"] = "torch.Tensor"
allowlist["torch.Tensor.resize_as_"] = "torch.Tensor"
allowlist["torch.Tensor.resize_as"] = "torch.Tensor"
allowlist["torch.Tensor.resize"] = "torch.Tensor"
allowlist["torch.Tensor.retain_grad"] = "syft.lib.python._SyNone"
allowlist["torch.Tensor.roll"] = "torch.Tensor"
allowlist["torch.Tensor.rot90"] = "torch.Tensor"
allowlist["torch.Tensor.round_"] = "torch.Tensor"
allowlist["torch.Tensor.round"] = "torch.Tensor"
allowlist["torch.Tensor.rsqrt_"] = "torch.Tensor"
allowlist["torch.Tensor.rsqrt"] = "torch.Tensor"
allowlist["torch.Tensor.scatter_"] = "torch.Tensor"
allowlist["torch.Tensor.scatter_add_"] = "torch.Tensor"
allowlist["torch.Tensor.scatter_add"] = "torch.Tensor"
allowlist["torch.Tensor.scatter"] = "torch.Tensor"
allowlist["torch.Tensor.select"] = "torch.Tensor"
allowlist["torch.Tensor.set_"] = "torch.Tensor"
allowlist["torch.Tensor.short"] = "torch.Tensor"
allowlist["torch.Tensor.sigmoid_"] = "torch.Tensor"
allowlist["torch.Tensor.sigmoid"] = "torch.Tensor"
allowlist["torch.Tensor.sign_"] = "torch.Tensor"
allowlist["torch.Tensor.sign"] = "torch.Tensor"
allowlist["torch.Tensor.sin_"] = "torch.Tensor"
allowlist["torch.Tensor.sin"] = "torch.Tensor"
allowlist["torch.Tensor.sinh_"] = "torch.Tensor"
allowlist["torch.Tensor.sinh"] = "torch.Tensor"
allowlist["torch.Tensor.slogdet"] = "torch.return_types.slogdet"
allowlist["torch.Tensor.softmax"] = "torch.Tensor"
# allowlist["torch.Tensor.solve"] = "torch.return_types.solve" # deprecated in torch==1.10.0
allowlist["torch.Tensor.sort"] = "torch.return_types.sort"
allowlist["torch.Tensor.split_with_sizes"] = "syft.lib.python.List"  # Tuple not List
allowlist["torch.Tensor.split"] = "syft.lib.python.List"  # Tuple not List
allowlist["torch.Tensor.sqrt_"] = "torch.Tensor"
allowlist["torch.Tensor.sqrt"] = "torch.Tensor"
allowlist["torch.Tensor.squeeze_"] = "torch.Tensor"
allowlist["torch.Tensor.squeeze"] = "torch.Tensor"
allowlist["torch.Tensor.std"] = "torch.Tensor"
allowlist["torch.Tensor.stft"] = "torch.Tensor"
allowlist["torch.Tensor.stride"] = UnionGenerator[  # Tuple not List
    "syft.lib.python.List", "syft.lib.python.Int"
]
allowlist["torch.Tensor.sub_"] = "torch.Tensor"
allowlist["torch.Tensor.sub"] = "torch.Tensor"
allowlist["torch.Tensor.sum_to_size"] = "torch.Tensor"
allowlist["torch.Tensor.sum"] = "torch.Tensor"
allowlist["torch.Tensor.svd"] = "torch.return_types.svd"
# allowlist["torch.Tensor.symeig"] = "torch.return_types.symeig" # deprecated in torch==1.10.0
allowlist["torch.Tensor.t_"] = "torch.Tensor"
allowlist["torch.Tensor.t"] = "torch.Tensor"
allowlist["torch.Tensor.t"] = "torch.Tensor"
allowlist["torch.Tensor.T"] = "torch.Tensor"
allowlist["torch.Tensor.take"] = "torch.Tensor"
allowlist["torch.Tensor.tan_"] = "torch.Tensor"
allowlist["torch.Tensor.tan"] = "torch.Tensor"
allowlist["torch.Tensor.tanh_"] = "torch.Tensor"
allowlist["torch.Tensor.tanh"] = "torch.Tensor"
allowlist["torch.Tensor.to"] = "torch.Tensor"
allowlist["torch.Tensor.tolist"] = "syft.lib.python.List"
allowlist["torch.Tensor.topk"] = "torch.return_types.topk"
allowlist["torch.Tensor.trace"] = "torch.Tensor"
allowlist["torch.Tensor.transpose_"] = "torch.Tensor"
allowlist["torch.Tensor.transpose"] = "torch.Tensor"
# allowlist["torch.Tensor.triangular_solve"] = "torch.return_types.triangular_solve" # deprecated in torch==1.11.0
allowlist["torch.Tensor.tril_"] = "torch.Tensor"
allowlist["torch.Tensor.tril"] = "torch.Tensor"
allowlist["torch.Tensor.triu_"] = "torch.Tensor"
allowlist["torch.Tensor.triu"] = "torch.Tensor"
allowlist["torch.Tensor.trunc_"] = "torch.Tensor"
allowlist["torch.Tensor.trunc"] = "torch.Tensor"
allowlist["torch.Tensor.type_as"] = "torch.Tensor"
allowlist["torch.Tensor.type"] = "syft.lib.python.String"
allowlist["torch.Tensor.unbind"] = "syft.lib.python.List"  # Tuple not List
allowlist["torch.Tensor.unfold"] = "torch.Tensor"
allowlist["torch.Tensor.uniform_"] = "torch.Tensor"
allowlist["torch.Tensor.unique_consecutive"] = "torch.Tensor"
allowlist["torch.Tensor.unique"] = "torch.Tensor"
allowlist["torch.Tensor.unsqueeze_"] = "torch.Tensor"
allowlist["torch.Tensor.unsqueeze"] = "torch.Tensor"
allowlist["torch.Tensor.var"] = "torch.Tensor"
allowlist["torch.Tensor.view_as"] = "torch.Tensor"
allowlist["torch.Tensor.view"] = "torch.Tensor"
allowlist["torch.Tensor.zero_"] = "torch.Tensor"


# --------------------------------------------------------------------------------------
# SECTION - Tensor methods with special version requirements
# --------------------------------------------------------------------------------------
# SECTION - Tensor methods since 1.5.0

allowlist[
    "torch.Tensor.__div__"
] = {  # exists in 1.4.0 but causes fatal exception on non floats
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist[
    "torch.Tensor.__floordiv__"
] = {  # exists in 1.4.0 but causes fatal exception on non floats
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist[
    "torch.Tensor.__rfloordiv__"
] = {  # exists in 1.4.0 but causes fatal exception on non floats
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}

allowlist["torch.Tensor.bitwise_and"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.bitwise_and_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.bitwise_or"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.bitwise_or_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist[
    "torch.Tensor.div"
] = {  # exists in 1.4.0 but causes fatal exception on non floats
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist[
    "torch.Tensor.div_"
] = {  # exists in 1.4.0 but causes fatal exception on non floats
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.floor_divide"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.floor_divide_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.logical_and"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.logical_and_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.logical_or"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.logical_or_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist[
    "torch.Tensor.remainder"
] = {  # exists in 1.4.0 but causes fatal exception on non floats
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist[
    "torch.Tensor.remainder_"
] = {  # exists in 1.4.0 but causes fatal exception on non floats
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.square"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.square_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.true_divide"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.true_divide_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}


# SECTION - Tensor methods since 1.5.1

allowlist["torch.Tensor.__ifloordiv__"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.1",
}

# SECTION - Tensor methods since 1.6.0

allowlist["torch.Tensor.is_meta"] = {
    "return_type": "syft.lib.python.Bool",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.absolute_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.absolute"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.acosh_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.acosh"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.asinh_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.asinh"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.atanh_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.atanh"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.deg2rad_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.deg2rad"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.fliplr"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.flipud"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.isfinite"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.isinf"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.isnan"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.logaddexp"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.logaddexp2"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.logcumsumexp"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.rad2deg_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.rad2deg"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.istft"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}

# SECTION - Tensor methods since 1.7.0

allowlist["torch.Tensor.__complex__"] = {
    "return_type": "syft.lib.python.Complex",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.amax"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.amin"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.arccos"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.arccos_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.arccosh"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.arccosh_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.arcsin"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.arcsin_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.arcsinh"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.arcsinh_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.arctan"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.arctan_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.arctanh"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.arctanh_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.clip"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.clip_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.count_nonzero"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.divide"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.divide_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.exp2"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.exp2_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.fix"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.fix_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.gcd"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.gcd_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.greater"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.greater_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.greater_equal"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.greater_equal_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.heaviside"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.heaviside_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.hypot"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.hypot_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.i0"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.i0_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.isneginf"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.isposinf"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.isreal"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.lcm"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.lcm_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.less"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.less_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.less_equal"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.less_equal_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.logit"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.logit_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.maximum"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.minimum"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.matrix_exp"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.multiply"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.multiply_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.nanquantile"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.nansum"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.negative"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.negative_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.nextafter"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.nextafter_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.outer"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.quantile"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.sgn"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.sgn_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.signbit"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.subtract"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.subtract_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.unsafe_chunk"] = {
    "return_type": "syft.lib.python.List",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.unsafe_split"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.vdot"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.movedim"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.Tensor.unsafe_split_with_sizes"] = {
    "return_type": "syft.lib.python.List",  # Tuple not List
    "min_version": "1.7.0",
}

# SECTION - Tensor methods since 1.8.0

# Deprecated
allowlist["torch.Tensor.fft"] = {
    "return_type": "torch.Tensor",
    "max_version": "1.7.1",
}

allowlist["torch.Tensor.ifft"] = {
    "return_type": "torch.Tensor",
    "max_version": "1.7.1",
}

allowlist["torch.Tensor.irfft"] = {
    "return_type": "torch.Tensor",
    "max_version": "1.7.1",
}

allowlist["torch.Tensor.rfft"] = {
    "return_type": "torch.Tensor",
    "max_version": "1.7.1",
}

# --------------------------------------------------------------------------------------
# SECTION - Tensor methods which are incomplete or untested but enabled
# --------------------------------------------------------------------------------------
allowlist["torch.Tensor.device"] = "torch.device"
allowlist["torch.Tensor.detach_"] = "torch.Tensor"
allowlist["torch.Tensor.grad"] = "torch.Tensor"  # MADHAVA: this needs fixing

# --------------------------------------------------------------------------------------
# SECTION - Tensor methods with specific issues or require a special test combination
# --------------------------------------------------------------------------------------
# allowlist["torch.layout"] = "torch.layout"  # requires protobuf serialization
# allowlist["torch.Tensor.layout"] = "torch.layout" # requires torch layout
allowlist["torch.Size"] = "torch.Size"  # requires protobuf serialization
allowlist["torch.Size.__len__"] = "syft.lib.python.Int"
allowlist["torch.Size.__iter__"] = "syft.lib.python.Iterator"
allowlist["torch.Size.__getitem__"] = "syft.lib.python.Int"
allowlist["torch.Tensor.size"] = UnionGenerator["torch.Size", "syft.lib.python.Int"]
allowlist["torch.Tensor.shape"] = "torch.Size"  # MADHAVA: this needs fixing
# allowlist["torch.Tensor.__iter__"] = "unknown"  # How to handle return iterator?
# allowlist["torch.Tensor.imag"] = "torch.Tensor"  # requires dtype complex
# allowlist["torch.Tensor.real"] = "torch.Tensor"  # requires dtype complex
# allowlist["torch.Tensor.qscheme"] = "unknown"  # requires quantized backend

# --------------------------------------------------------------------------------------
# SECTION - Tensor methods which require named tensors
# --------------------------------------------------------------------------------------
# allowlist["torch.Tensor.unflatten"] = "torch.Tensor" # named tensors
# allowlist["torch.Tensor.refine_names"] = "torch.Tensor" # named tensors
# allowlist["torch.Tensor.rename_"] = "torch.Tensor"  # named tensors
# allowlist["torch.Tensor.rename"] = "torch.Tensor"  # named tensors
# allowlist["torch.Tensor.align_as"] = "torch.Tensor" # named tensors
# allowlist["torch.Tensor.align_to"] = "torch.Tensor" # named tensors
# allowlist["torch.Tensor.name"] = "Optional[str]" # requires named tensors and Optional
# allowlist["torch.Tensor.names"] = "Tuple[str]" # requires named tensors and Tuple
# allowlist["torch.Tensor.__torch_function__"] = "unknown" # 1.7.0 # probably wont work

# --------------------------------------------------------------------------------------
# SECTION - Tensor methods which require classes or callables or external libs
# --------------------------------------------------------------------------------------
# allowlist["torch.Tensor.apply_"] = "torch.Tensor" # requires a callable
# allowlist["torch.Tensor.as_subclass"] = "torch.Tensor" # requires a subclass
# allowlist["torch.Tensor.map_"] = "unknown"  # requires a callable
# allowlist["torch.Tensor.map2_"] = "unknown"  # requires a callable
# allowlist["torch.Tensor.numpy"] = "numpy.ndarray"  # requires numpy.ndarray
# allowlist["torch.Tensor.reinforce"] = "unknown"  # requires reinforce

# --------------------------------------------------------------------------------------
# SECTION - Tensor methods which require sparse
# --------------------------------------------------------------------------------------
# allowlist["torch.Tensor.smm"] = "unknown"  # requires sparse tensors
# allowlist["torch.Tensor.sparse_dim"] = "unknown"  # requires sparse tensors
# allowlist["torch.Tensor.sparse_mask"] = "unknown"  # requires sparse tensors
# allowlist["torch.Tensor.sspaddmm"] = "torch.Tensor"  # requires sparse tensors
# allowlist["torch.Tensor.sparse_resize_"] = "unknown" # requires sparse tensors
# allowlist["torch.Tensor.sparse_resize_and_clear_"] = "unknown" # requires sparse
# allowlist["torch.Tensor.values"] = "unknown"  # requires sparse tensors

# SECTION - Module methods
allowlist["torch.set_grad_enabled"] = "syft.lib.python._SyNone"
allowlist["torch.zeros"] = "torch.Tensor"
allowlist["torch.randn"] = "torch.Tensor"
allowlist["torch.ones_like"] = "torch.Tensor"
allowlist["torch.Tensor.__len__"] = "syft.lib.python.Int"
allowlist["torch.arange"] = "torch.Tensor"

# --------------------------------------------------------------------------------------
# SECTION - Torch functions enabled as torch.Tensor methods above
# --------------------------------------------------------------------------------------

allowlist["torch.abs_"] = "torch.Tensor"
allowlist["torch.abs"] = "torch.Tensor"
allowlist["torch.acos_"] = "torch.Tensor"
allowlist["torch.acos"] = "torch.Tensor"
allowlist["torch.add"] = "torch.Tensor"
allowlist["torch.addbmm"] = "torch.Tensor"
allowlist["torch.addcdiv"] = "torch.Tensor"
allowlist["torch.addcmul"] = "torch.Tensor"
allowlist["torch.addmm"] = "torch.Tensor"
allowlist["torch.addmv_"] = "torch.Tensor"
allowlist["torch.addmv"] = "torch.Tensor"
allowlist["torch.addr"] = "torch.Tensor"
allowlist["torch.all"] = "torch.Tensor"
allowlist["torch.allclose"] = "syft.lib.python.Bool"
allowlist["torch.angle"] = "torch.Tensor"
allowlist["torch.any"] = "torch.Tensor"
allowlist["torch.argmax"] = "torch.Tensor"
allowlist["torch.argmin"] = "torch.Tensor"
allowlist["torch.argsort"] = "torch.Tensor"
allowlist["torch.as_strided_"] = "torch.Tensor"
allowlist["torch.as_strided"] = "torch.Tensor"
allowlist["torch.asin_"] = "torch.Tensor"
allowlist["torch.asin"] = "torch.Tensor"
allowlist["torch.atan_"] = "torch.Tensor"
allowlist["torch.atan"] = "torch.Tensor"
allowlist["torch.atan2"] = "torch.Tensor"
allowlist["torch.baddbmm"] = "torch.Tensor"
allowlist["torch.bernoulli"] = "torch.Tensor"
allowlist["torch.bitwise_and"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.bitwise_not"] = "torch.Tensor"
allowlist["torch.bitwise_or"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.bitwise_xor"] = "torch.Tensor"
allowlist["torch.bmm"] = "torch.Tensor"
allowlist["torch.cat"] = "torch.Tensor"
allowlist["torch.ceil_"] = "torch.Tensor"
allowlist["torch.ceil"] = "torch.Tensor"
allowlist["torch.cholesky_inverse"] = "torch.Tensor"
allowlist["torch.cholesky_solve"] = "torch.Tensor"
allowlist["torch.cholesky"] = "torch.Tensor"
allowlist["torch.chunk"] = "syft.lib.python.List"  # Tuple not List
allowlist["torch.clamp_"] = "torch.Tensor"
allowlist["torch.clamp_max_"] = "torch.Tensor"
allowlist["torch.clamp_max"] = "torch.Tensor"
allowlist["torch.clamp_min_"] = "torch.Tensor"
allowlist["torch.clamp_min"] = "torch.Tensor"
allowlist["torch.clamp"] = "torch.Tensor"
allowlist["torch.clone"] = "torch.Tensor"
allowlist["torch.conj"] = "torch.Tensor"
allowlist["torch.cos_"] = "torch.Tensor"
allowlist["torch.cos"] = "torch.Tensor"
allowlist["torch.cosh_"] = "torch.Tensor"
allowlist["torch.cosh"] = "torch.Tensor"
allowlist["torch.cross"] = "torch.Tensor"
allowlist["torch.cummax"] = {
    "return_type": "torch.return_types.cummax",
    "min_version": "1.5.0",
}
allowlist["torch.cummin"] = {
    "return_type": "torch.return_types.cummin",
    "min_version": "1.5.0",
}
allowlist["torch.cumprod"] = "torch.Tensor"
allowlist["torch.cumsum"] = "torch.Tensor"
allowlist["torch.dequantize"] = "torch.Tensor"
allowlist["torch.det"] = "torch.Tensor"
allowlist["torch.detach"] = "torch.Tensor"
allowlist["torch.diag_embed"] = "torch.Tensor"
allowlist["torch.diag"] = "torch.Tensor"
allowlist["torch.diagflat"] = "torch.Tensor"
allowlist["torch.diagonal"] = "torch.Tensor"
allowlist["torch.digamma"] = "torch.Tensor"
allowlist["torch.dist"] = "torch.Tensor"
allowlist["torch.div"] = "torch.Tensor"
allowlist["torch.dot"] = "torch.Tensor"
# allowlist["torch.eig"] = "torch.return_types.eig" # deprecated in torch==1.10.0
allowlist["torch.eq"] = "torch.Tensor"
allowlist["torch.equal"] = "syft.lib.python.Bool"
allowlist["torch.erf_"] = "torch.Tensor"
allowlist["torch.erf"] = "torch.Tensor"
allowlist["torch.erfc_"] = "torch.Tensor"
allowlist["torch.erfc"] = "torch.Tensor"
allowlist["torch.erfinv"] = "torch.Tensor"
allowlist["torch.exp_"] = "torch.Tensor"
allowlist["torch.exp"] = "torch.Tensor"
allowlist["torch.expm1_"] = "torch.Tensor"
allowlist["torch.expm1"] = "torch.Tensor"
allowlist["torch.fft"] = "torch.Tensor"
allowlist["torch.fill_"] = "torch.Tensor"
allowlist["torch.flatten"] = "torch.Tensor"
allowlist["torch.flip"] = "torch.Tensor"
allowlist["torch.floor_"] = "torch.Tensor"
allowlist["torch.floor_divide"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.floor"] = "torch.Tensor"
allowlist["torch.fmod"] = "torch.Tensor"
allowlist["torch.frac_"] = "torch.Tensor"
allowlist["torch.frac"] = "torch.Tensor"
allowlist["torch.from_numpy"] = "torch.Tensor"
allowlist["torch.gather"] = "torch.Tensor"
allowlist["torch.ge"] = "torch.Tensor"
allowlist["torch.geqrf"] = "torch.return_types.geqrf"
allowlist["torch.ger"] = "torch.Tensor"
allowlist["torch.get_device"] = "syft.lib.python.Int"
allowlist["torch.gt"] = "torch.Tensor"
allowlist["torch.hardshrink"] = "torch.Tensor"
allowlist["torch.histc"] = "torch.Tensor"
allowlist["torch.index_add"] = "torch.Tensor"
allowlist["torch.index_copy"] = "torch.Tensor"
allowlist["torch.index_fill"] = "torch.Tensor"
allowlist["torch.index_put_"] = "torch.Tensor"
allowlist["torch.index_put"] = "torch.Tensor"
allowlist["torch.index_select"] = "torch.Tensor"
allowlist["torch.int_repr"] = "torch.Tensor"
allowlist["torch.inverse"] = "torch.Tensor"
allowlist["torch.is_complex"] = "syft.lib.python.Bool"
allowlist["torch.is_distributed"] = "syft.lib.python.Bool"
allowlist["torch.is_floating_point"] = "syft.lib.python.Bool"
allowlist["torch.is_nonzero"] = "syft.lib.python.Bool"
allowlist["torch.is_same_size"] = "syft.lib.python.Bool"
allowlist["torch.is_signed"] = "syft.lib.python.Bool"
allowlist["torch.isclose"] = "torch.Tensor"
allowlist["torch.kthvalue"] = "torch.return_types.kthvalue"
allowlist["torch.le"] = "torch.Tensor"
allowlist["torch.lerp"] = "torch.Tensor"
allowlist["torch.lgamma"] = "torch.Tensor"
allowlist["torch.log_"] = "torch.Tensor"
allowlist["torch.log_softmax"] = "torch.Tensor"
allowlist["torch.log"] = "torch.Tensor"
allowlist["torch.log10_"] = "torch.Tensor"
allowlist["torch.log10"] = "torch.Tensor"
allowlist["torch.log1p_"] = "torch.Tensor"
allowlist["torch.log1p"] = "torch.Tensor"
allowlist["torch.log2_"] = "torch.Tensor"
allowlist["torch.log2"] = "torch.Tensor"
allowlist["torch.logdet"] = "torch.Tensor"
allowlist["torch.logical_and"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.logical_not"] = "torch.Tensor"
allowlist["torch.logical_or"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.logical_xor"] = "torch.Tensor"
allowlist["torch.logsumexp"] = "torch.Tensor"
# allowlist["torch.lstsq"] = "torch.return_types.lstsq" # deprecated in torch==1.10.0
allowlist["torch.lt"] = "torch.Tensor"
allowlist["torch.lu_solve"] = "torch.Tensor"
allowlist["torch.lu"] = "syft.lib.python.List"  # Tuple not List
allowlist["torch.masked_fill"] = "torch.Tensor"
allowlist["torch.masked_scatter"] = "torch.Tensor"
allowlist["torch.masked_select"] = "torch.Tensor"
allowlist["torch.matmul"] = "torch.Tensor"
allowlist["torch.matrix_power"] = "torch.Tensor"
allowlist["torch.mean"] = "torch.Tensor"
allowlist["torch.mm"] = "torch.Tensor"
allowlist["torch.mode"] = "torch.return_types.mode"
allowlist["torch.mul"] = "torch.Tensor"
allowlist["torch.multinomial"] = "torch.Tensor"
allowlist["torch.mv"] = "torch.Tensor"
allowlist["torch.mvlgamma"] = "torch.Tensor"
allowlist["torch.narrow"] = "torch.Tensor"
allowlist["torch.ne"] = "torch.Tensor"
allowlist["torch.neg_"] = "torch.Tensor"
allowlist["torch.neg"] = "torch.Tensor"
allowlist["torch.nonzero"] = "torch.Tensor"
allowlist["torch.norm"] = "torch.Tensor"
# allowlist["torch.orgqr"] = "torch.Tensor" # wrong return type in 1.9.1
# allowlist["torch.ormqr"] = "torch.Tensor" # wrong return type in 1.9.1
allowlist["torch.pinverse"] = "torch.Tensor"
allowlist["torch.polygamma"] = "torch.Tensor"
allowlist["torch.pow"] = "torch.Tensor"
allowlist["torch.prelu"] = "torch.Tensor"
allowlist["torch.q_per_channel_axis"] = "syft.lib.python.Int"
allowlist["torch.q_per_channel_scales"] = "torch.Tensor"
allowlist["torch.q_per_channel_zero_points"] = "torch.Tensor"
allowlist["torch.q_scale"] = "syft.lib.python.Float"
allowlist["torch.q_zero_point"] = "syft.lib.python.Int"
# allowlist["torch.qr"] = "torch.return_types.qr" # deprecated in torch==1.10.0
allowlist["torch.reciprocal_"] = "torch.Tensor"
allowlist["torch.reciprocal"] = "torch.Tensor"
allowlist["torch.relu_"] = "torch.Tensor"
allowlist["torch.relu"] = "torch.Tensor"
allowlist["torch.remainder"] = "torch.Tensor"
allowlist["torch.renorm"] = "torch.Tensor"
allowlist["torch.repeat_interleave"] = "torch.Tensor"
allowlist["torch.reshape"] = "torch.Tensor"
allowlist["torch.resize_as_"] = "torch.Tensor"
allowlist["torch.roll"] = "torch.Tensor"
allowlist["torch.rot90"] = "torch.Tensor"
allowlist["torch.round_"] = "torch.Tensor"
allowlist["torch.round"] = "torch.Tensor"
allowlist["torch.rsqrt_"] = "torch.Tensor"
allowlist["torch.rsqrt"] = "torch.Tensor"
allowlist["torch.scatter_add"] = "torch.Tensor"
allowlist["torch.scatter"] = "torch.Tensor"
allowlist["torch.select"] = "torch.Tensor"
allowlist["torch.sigmoid_"] = "torch.Tensor"
allowlist["torch.sigmoid"] = "torch.Tensor"
allowlist["torch.sign"] = "torch.Tensor"
allowlist["torch.sin_"] = "torch.Tensor"
allowlist["torch.sin"] = "torch.Tensor"
allowlist["torch.sinh_"] = "torch.Tensor"
allowlist["torch.sinh"] = "torch.Tensor"
allowlist["torch.slogdet"] = "torch.return_types.slogdet"
allowlist["torch.softmax"] = "torch.Tensor"
# allowlist["torch.solve"] = "torch.return_types.solve" # deprecated in torch==1.10.0
allowlist["torch.sort"] = "torch.return_types.sort"
allowlist["torch.split_with_sizes"] = "syft.lib.python.List"  # Tuple not List
allowlist["torch.split"] = "syft.lib.python.List"  # Tuple not List
allowlist["torch.sqrt_"] = "torch.Tensor"
allowlist["torch.sqrt"] = "torch.Tensor"
allowlist["torch.square_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.square"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.squeeze"] = "torch.Tensor"
allowlist["torch.stack"] = "torch.Tensor"
allowlist["torch.std"] = "torch.Tensor"
allowlist["torch.stft"] = "torch.Tensor"
allowlist["torch.sub"] = "torch.Tensor"
allowlist["torch.sum"] = "torch.Tensor"
allowlist["torch.svd"] = "torch.return_types.svd"
# allowlist["torch.symeig"] = "torch.return_types.symeig" # deprecated in torch==1.10.0
allowlist["torch.t"] = "torch.Tensor"
allowlist["torch.take"] = "torch.Tensor"
allowlist["torch.tan_"] = "torch.Tensor"
allowlist["torch.tan"] = "torch.Tensor"
allowlist["torch.tanh_"] = "torch.Tensor"
allowlist["torch.tanh"] = "torch.Tensor"
allowlist["torch.topk"] = "torch.return_types.topk"
allowlist["torch.trace"] = "torch.Tensor"
allowlist["torch.transpose"] = "torch.Tensor"
# allowlist["torch.triangular_solve"] = "torch.return_types.triangular_solve" # deprecated in torch==1.11.0
allowlist["torch.tril"] = "torch.Tensor"
allowlist["torch.triu"] = "torch.Tensor"
allowlist["torch.true_divide"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.trunc_"] = "torch.Tensor"
allowlist["torch.trunc"] = "torch.Tensor"
allowlist["torch.unique_consecutive"] = "torch.Tensor"
allowlist["torch.unique"] = "torch.Tensor"
allowlist["torch.unsqueeze"] = "torch.Tensor"
allowlist["torch.var"] = "torch.Tensor"
allowlist["torch.unsafe_chunk"] = "syft.lib.python.List"  # Tuple not List

allowlist["torch.ifft"] = {
    "return_type": "torch.Tensor",
    "max_version": "1.7.1",
}

allowlist["torch.irfft"] = {
    "return_type": "torch.Tensor",
    "max_version": "1.7.1",
}

allowlist["torch.rfft"] = {
    "return_type": "torch.Tensor",
    "max_version": "1.7.1",
}

# SECTION - Tensor functions since 1.6.0

allowlist["torch.absolute"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.acosh_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.acosh"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.asinh_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.asinh"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.atanh_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.atanh"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.deg2rad_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.deg2rad"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.fliplr"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.flipud"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.isfinite"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.isinf"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.isnan"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.logaddexp"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.logaddexp2"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.logcumsumexp"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.rad2deg_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.rad2deg"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}
allowlist["torch.istft"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}

# SECTION - Tensor functions since 1.7.0

allowlist["torch.amax"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.amin"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.arccos"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.arccos_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.arccosh"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.arccosh_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.arcsin"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.arcsin_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.arcsinh"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.arcsinh_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.arctan"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.arctan_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.arctanh"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.arctanh_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.clip"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.clip_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.count_nonzero"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.divide"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.exp2"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.exp2_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.fix"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.fix_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.gcd"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.gcd_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.greater"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.greater_equal"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.heaviside"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.hypot"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.i0"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.i0_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.isneginf"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.isposinf"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.isreal"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.lcm"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.lcm_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.less"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.less_equal"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.logit"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.logit_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.maximum"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.minimum"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.matrix_exp"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.multiply"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.nanquantile"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.nansum"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.negative"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.negative_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.nextafter"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.outer"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.quantile"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.sgn"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.signbit"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.subtract"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.unsafe_chunk"] = {
    "return_type": "syft.lib.python.List",  # Tuple not List
    "min_version": "1.7.0",
}
allowlist["torch.unsafe_split"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.vdot"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.movedim"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.7.0",
}
allowlist["torch.unsafe_split_with_sizes"] = {
    "return_type": "syft.lib.python.List",  # Tuple not List
    "min_version": "1.7.0",
}

# --------------------------------------------------------------------------------------
# SECTION - Torch functions not enabled yet
# --------------------------------------------------------------------------------------

# allowlist["torch.zero_"] = "torch.Tensor"
# allowlist["torch.detach_"] = "torch.Tensor"
# allowlist["torch.device"] = "torch.Tensor"
# allowlist["torch.imag"] = "torch.Tensor"
# allowlist["torch.layout"] = "torch.Tensor"
# allowlist["torch.max"] = "torch.Tensor"
# allowlist["torch.median"] = "torch.Tensor"
# allowlist["torch.min"] = "torch.Tensor"
# allowlist["torch.name"] = "torch.Tensor"
# allowlist["torch.not_equal"] = "torch.Tensor"
# allowlist["torch.qscheme"] = "torch.Tensor"
# allowlist["torch.real"] = "torch.Tensor"
# allowlist["torch.smm"] = "torch.Tensor"
# allowlist["torch.sspaddmm"] = "torch.Tensor"


# --------------------------------------------------------------------------------------
# SECTION - Torch functions used in the fast tests: $ pytest -m fast
# --------------------------------------------------------------------------------------

allowlist["torch.cuda.is_available"] = "syft.lib.python.Bool"
allowlist["torch.device"] = "torch.device"  # warning this must come before the attrs
allowlist["torch.device.index"] = "syft.lib.python.Int"
allowlist["torch.device.type"] = "syft.lib.python.String"
allowlist["torch.random.initial_seed"] = "syft.lib.python.Int"
allowlist["torch.zeros_like"] = "torch.Tensor"

# --------------------------------------------------------------------------------------
# SECTION - Torch functions which are enabled but supported above on torch.Tensor
# --------------------------------------------------------------------------------------

# SECTION - Parameter methods
# torch.nn.Parameter is a subclass of torch.Tensor
# However, we still need the constructor Class to be listed here. Everything else is
# automatically added in create_torch_ast function by doing:
# method = method.replace("torch.Tensor.", "torch.nn.Parameter.")
# allowlist["torch.nn.Parameter"] = "torch.nn.Parameter"

# Misc
allowlist["torch.manual_seed"] = "torch.Generator"
allowlist["torch.Generator"] = "torch.Generator"
allowlist["torch.Generator.manual_seed"] = "torch.Generator"
allowlist["torch.Generator.get_state"] = "torch.Tensor"
allowlist["torch.Generator.set_state"] = "torch.Generator"
allowlist["torch.exp"] = "torch.Tensor"

# Modules
allowlist["torch.nn.Module"] = "torch.nn.Module"
allowlist["torch.nn.Module.__call__"] = "torch.Tensor"
allowlist["torch.nn.Module.forward"] = "torch.Tensor"
allowlist["torch.nn.Module.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Module.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Module.train"] = "torch.nn.Module"
allowlist["torch.nn.Module.cuda"] = "torch.nn.Module"
allowlist["torch.nn.Module.cpu"] = "torch.nn.Module"
allowlist["torch.nn.Module.add_module"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Module.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist[
    "torch.nn.Module.load_state_dict"
] = "syft.lib.python._SyNone"  # torch.nn.modules.module._IncompatibleKeys
allowlist["torch.nn.Module.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Conv2d"] = "torch.nn.Conv2d"
allowlist["torch.nn.Conv2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.Conv2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Conv2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Conv2d.train"] = "torch.nn.Conv2d"
allowlist["torch.nn.Conv2d.cuda"] = "torch.nn.Conv2d"
allowlist["torch.nn.Conv2d.cpu"] = "torch.nn.Conv2d"
allowlist["torch.nn.Conv2d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist[
    "torch.nn.Conv2d.load_state_dict"
] = "syft.lib.python._SyNone"  # torch.nn.modules.module._IncompatibleKeys
allowlist["torch.nn.Conv2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Dropout2d"] = "torch.nn.Dropout2d"
allowlist["torch.nn.Dropout2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.Dropout2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Dropout2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Dropout2d.train"] = "torch.nn.Dropout2d"
allowlist["torch.nn.Dropout2d.cuda"] = "torch.nn.Dropout2d"
allowlist["torch.nn.Dropout2d.cpu"] = "torch.nn.Dropout2d"
allowlist["torch.nn.Dropout2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Linear"] = "torch.nn.Linear"
allowlist["torch.nn.Linear.__call__"] = "torch.Tensor"
allowlist["torch.nn.Linear.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Linear.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Linear.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Linear.train"] = "torch.nn.Linear"
allowlist["torch.nn.Linear.cuda"] = "torch.nn.Linear"
allowlist["torch.nn.Linear.cpu"] = "torch.nn.Linear"
allowlist["torch.nn.Linear.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Linear.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Linear.extra_repr"] = "syft.lib.python.String"


# DataLoader
allowlist["torch.utils.data.DataLoader"] = "torch.utils.data.DataLoader"
allowlist["torch.utils.data.DataLoader.__iter__"] = "syft.lib.python.Iterator"
allowlist["torch.utils.data.DataLoader.__len__"] = "syft.lib.python.Int"

allowlist[
    "torch.utils.data.dataloader._SingleProcessDataLoaderIter"
] = "torch.utils.data.dataloader._SingleProcessDataLoaderIter"
allowlist[
    "torch.utils.data.dataloader._SingleProcessDataLoaderIter.__iter__"
] = "syft.lib.python.Iterator"
allowlist[
    "torch.utils.data.dataloader._SingleProcessDataLoaderIter.__len__"
] = "syft.lib.python.Int"


# Functional
allowlist["torch.nn.functional.relu"] = "torch.Tensor"
allowlist["torch.nn.functional.leaky_relu"] = "torch.Tensor"
allowlist["torch.nn.functional.gelu"] = "torch.Tensor"
allowlist["torch.nn.functional.max_pool2d"] = "torch.Tensor"
allowlist["torch.nn.functional.log_softmax"] = "torch.Tensor"
allowlist["torch.flatten"] = "torch.Tensor"

# Optimizers
allowlist["torch.optim.ASGD"] = "torch.optim.ASGD"
allowlist["torch.optim.ASGD.zero_grad"] = "syft.lib.python._SyNone"
allowlist["torch.optim.ASGD.step"] = "syft.lib.python._SyNone"
allowlist["torch.optim.ASGD.state_dict"] = "syft.lib.python.collections.OrderedDict"

allowlist["torch.optim.Adadelta"] = "torch.optim.Adadelta"
allowlist["torch.optim.Adadelta.zero_grad"] = "syft.lib.python._SyNone"
allowlist["torch.optim.Adadelta.step"] = "syft.lib.python._SyNone"
allowlist["torch.optim.Adadelta.state_dict"] = "syft.lib.python.collections.OrderedDict"

allowlist["torch.optim.Adagrad"] = "torch.optim.Adagrad"
allowlist["torch.optim.Adagrad.zero_grad"] = "syft.lib.python._SyNone"
allowlist["torch.optim.Adagrad.step"] = "syft.lib.python._SyNone"
allowlist["torch.optim.Adagrad.state_dict"] = "syft.lib.python.collections.OrderedDict"

allowlist["torch.optim.Adam"] = "torch.optim.Adam"
allowlist["torch.optim.Adam.zero_grad"] = "syft.lib.python._SyNone"
allowlist["torch.optim.Adam.step"] = "syft.lib.python._SyNone"
allowlist["torch.optim.Adam.state_dict"] = "syft.lib.python.collections.OrderedDict"

allowlist["torch.optim.AdamW"] = "torch.optim.AdamW"
allowlist["torch.optim.AdamW.zero_grad"] = "syft.lib.python._SyNone"
allowlist["torch.optim.AdamW.step"] = "syft.lib.python._SyNone"
allowlist["torch.optim.AdamW.state_dict"] = "syft.lib.python.collections.OrderedDict"

allowlist["torch.optim.Adamax"] = "torch.optim.Adamax"
allowlist["torch.optim.Adamax.zero_grad"] = "syft.lib.python._SyNone"
allowlist["torch.optim.Adamax.step"] = "syft.lib.python._SyNone"
allowlist["torch.optim.Adamax.state_dict"] = "syft.lib.python.collections.OrderedDict"

allowlist["torch.optim.LBFGS"] = "torch.optim.LBFGS"
allowlist["torch.optim.LBFGS.zero_grad"] = "syft.lib.python._SyNone"
allowlist["torch.optim.LBFGS.step"] = "syft.lib.python._SyNone"
allowlist["torch.optim.LBFGS.state_dict"] = "syft.lib.python.collections.OrderedDict"

allowlist["torch.optim.Optimizer"] = "torch.optim.Optimizer"
allowlist["torch.optim.Optimizer.zero_grad"] = "syft.lib.python._SyNone"
allowlist["torch.optim.Optimizer.step"] = "syft.lib.python._SyNone"
allowlist[
    "torch.optim.Optimizer.state_dict"
] = "syft.lib.python.collections.OrderedDict"

allowlist["torch.optim.RMSprop"] = "torch.optim.RMSprop"
allowlist["torch.optim.RMSprop.zero_grad"] = "syft.lib.python._SyNone"
allowlist["torch.optim.RMSprop.step"] = "syft.lib.python._SyNone"
allowlist["torch.optim.RMSprop.state_dict"] = "syft.lib.python.collections.OrderedDict"

allowlist["torch.optim.Rprop"] = "torch.optim.Rprop"
allowlist["torch.optim.Rprop.zero_grad"] = "syft.lib.python._SyNone"
allowlist["torch.optim.Rprop.step"] = "syft.lib.python._SyNone"
allowlist["torch.optim.Rprop.state_dict"] = "syft.lib.python.collections.OrderedDict"

allowlist["torch.optim.SGD"] = "torch.optim.SGD"
allowlist["torch.optim.SGD.zero_grad"] = "syft.lib.python._SyNone"
allowlist["torch.optim.SGD.step"] = "syft.lib.python._SyNone"
allowlist["torch.optim.SGD.state_dict"] = "syft.lib.python.collections.OrderedDict"

allowlist["torch.optim.SparseAdam"] = "torch.optim.SparseAdam"
allowlist["torch.optim.SparseAdam.zero_grad"] = "syft.lib.python._SyNone"
allowlist["torch.optim.SparseAdam.step"] = "syft.lib.python._SyNone"
allowlist[
    "torch.optim.SparseAdam.state_dict"
] = "syft.lib.python.collections.OrderedDict"

# Scheduler
allowlist["torch.optim.lr_scheduler.StepLR"] = "torch.optim.lr_scheduler.StepLR"
allowlist["torch.optim.lr_scheduler.StepLR.step"] = "syft.lib.python._SyNone"
allowlist[
    "torch.optim.lr_scheduler.StepLR.state_dict"
] = "syft.lib.python.collections.OrderedDict"

# Autograd
allowlist["torch.no_grad"] = "torch.autograd.grad_mode.no_grad"
allowlist["torch.autograd.grad_mode.no_grad"] = "torch.autograd.grad_mode.no_grad"
allowlist["torch.autograd.grad_mode.no_grad.__enter__"] = "syft.lib.python._SyNone"
allowlist["torch.autograd.grad_mode.no_grad.__exit__"] = "syft.lib.python._SyNone"

allowlist["torch.nn.Sequential"] = "torch.nn.Sequential"
allowlist["torch.nn.Sequential.cpu"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Sequential.cuda"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Sequential.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Sequential.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Sequential.train"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Sequential.eval"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Sequential.__call__"] = "torch.Tensor"

# Loss Functions
allowlist["torch.nn.functional.cosine_embedding_loss"] = "torch.Tensor"
allowlist["torch.nn.functional.ctc_loss"] = "torch.Tensor"
allowlist["torch.nn.functional.hinge_embedding_loss"] = "torch.Tensor"
allowlist["torch.nn.functional.l1_loss"] = "torch.Tensor"
allowlist["torch.nn.functional.margin_ranking_loss"] = "torch.Tensor"
allowlist["torch.nn.functional.mse_loss"] = "torch.Tensor"
allowlist["torch.nn.functional.multi_margin_loss"] = "torch.Tensor"
allowlist["torch.nn.functional.multilabel_margin_loss"] = "torch.Tensor"
allowlist["torch.nn.functional.multilabel_soft_margin_loss"] = "torch.Tensor"
allowlist["torch.nn.functional.nll_loss"] = "torch.Tensor"
allowlist["torch.nn.functional.cross_entropy"] = "torch.Tensor"
allowlist["torch.nn.functional.poisson_nll_loss"] = "torch.Tensor"
allowlist["torch.nn.functional.smooth_l1_loss"] = "torch.Tensor"
allowlist["torch.nn.functional.soft_margin_loss"] = "torch.Tensor"
allowlist["torch.nn.functional.triplet_margin_loss"] = "torch.Tensor"

allowlist["torch.nn.AdaptiveLogSoftmaxWithLoss"] = "torch.nn.AdaptiveLogSoftmaxWithLoss"
allowlist["torch.nn.AdaptiveLogSoftmaxWithLoss.__call__"] = "torch.Tensor"
allowlist["torch.nn.BCELoss"] = "torch.nn.BCELoss"
allowlist["torch.nn.BCELoss.__call__"] = "torch.Tensor"
allowlist["torch.nn.BCEWithLogitsLoss"] = "torch.nn.BCEWithLogitsLoss"
allowlist["torch.nn.BCEWithLogitsLoss.__call__"] = "torch.Tensor"
allowlist["torch.nn.CTCLoss"] = "torch.nn.CTCLoss"
allowlist["torch.nn.CTCLoss.__call__"] = "torch.Tensor"
allowlist["torch.nn.CrossEntropyLoss"] = "torch.nn.CrossEntropyLoss"
allowlist["torch.nn.CrossEntropyLoss.__call__"] = "torch.Tensor"
allowlist["torch.nn.CosineEmbeddingLoss"] = "torch.nn.CosineEmbeddingLoss"
allowlist["torch.nn.CosineEmbeddingLoss.__call__"] = "torch.Tensor"
allowlist["torch.nn.HingeEmbeddingLoss"] = "torch.nn.HingeEmbeddingLoss"
allowlist["torch.nn.HingeEmbeddingLoss.__call__"] = "torch.Tensor"
allowlist["torch.nn.KLDivLoss"] = "torch.nn.KLDivLoss"
allowlist["torch.nn.KLDivLoss.__call__"] = "torch.Tensor"
allowlist["torch.nn.L1Loss"] = "torch.nn.L1Loss"
allowlist["torch.nn.L1Loss.__call__"] = "torch.Tensor"
allowlist["torch.nn.MSELoss"] = "torch.nn.MSELoss"
allowlist["torch.nn.MSELoss.__call__"] = "torch.Tensor"
allowlist["torch.nn.MarginRankingLoss"] = "torch.nn.MarginRankingLoss"
allowlist["torch.nn.MarginRankingLoss.__call__"] = "torch.Tensor"
allowlist["torch.nn.MultiLabelMarginLoss"] = "torch.nn.MultiLabelMarginLoss"
allowlist["torch.nn.MultiLabelMarginLoss.__call__"] = "torch.Tensor"
allowlist["torch.nn.MultiLabelSoftMarginLoss"] = "torch.nn.MultiLabelSoftMarginLoss"
allowlist["torch.nn.MultiLabelSoftMarginLoss.__call__"] = "torch.Tensor"
allowlist["torch.nn.MultiMarginLoss"] = "torch.nn.MultiMarginLoss"
allowlist["torch.nn.MultiMarginLoss.__call__"] = "torch.Tensor"
allowlist["torch.nn.NLLLoss"] = "torch.nn.NLLLoss"
allowlist["torch.nn.NLLLoss.__call__"] = "torch.Tensor"
allowlist["torch.nn.NLLLoss2d"] = "torch.nn.NLLLoss2d"
allowlist["torch.nn.NLLLoss2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.PoissonNLLLoss"] = "torch.nn.PoissonNLLLoss"
allowlist["torch.nn.PoissonNLLLoss.__call__"] = "torch.Tensor"
allowlist["torch.nn.SmoothL1Loss"] = "torch.nn.SmoothL1Loss"
allowlist["torch.nn.SmoothL1Loss.__call__"] = "torch.Tensor"
allowlist["torch.nn.SoftMarginLoss"] = "torch.nn.SoftMarginLoss"
allowlist["torch.nn.SoftMarginLoss.__call__"] = "torch.Tensor"
allowlist["torch.nn.TripletMarginLoss"] = "torch.nn.TripletMarginLoss"
allowlist["torch.nn.TripletMarginLoss.__call__"] = "torch.Tensor"

# Layer Classes

allowlist["torch.nn.AdaptiveAvgPool1d"] = "torch.nn.AdaptiveAvgPool1d"
allowlist["torch.nn.AdaptiveAvgPool1d.__call__"] = "torch.Tensor"
allowlist["torch.nn.AdaptiveAvgPool1d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.AdaptiveAvgPool1d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AdaptiveAvgPool1d.train"] = "torch.nn.AdaptiveAvgPool1d"
allowlist["torch.nn.AdaptiveAvgPool1d.cuda"] = "torch.nn.AdaptiveAvgPool1d"
allowlist["torch.nn.AdaptiveAvgPool1d.cpu"] = "torch.nn.AdaptiveAvgPool1d"
allowlist[
    "torch.nn.AdaptiveAvgPool1d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.AdaptiveAvgPool1d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AdaptiveAvgPool1d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.AdaptiveAvgPool2d"] = "torch.nn.AdaptiveAvgPool2d"
allowlist["torch.nn.AdaptiveAvgPool2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.AdaptiveAvgPool2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.AdaptiveAvgPool2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AdaptiveAvgPool2d.train"] = "torch.nn.AdaptiveAvgPool2d"
allowlist["torch.nn.AdaptiveAvgPool2d.cuda"] = "torch.nn.AdaptiveAvgPool2d"
allowlist["torch.nn.AdaptiveAvgPool2d.cpu"] = "torch.nn.AdaptiveAvgPool2d"
allowlist[
    "torch.nn.AdaptiveAvgPool2d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.AdaptiveAvgPool2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AdaptiveAvgPool2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.AdaptiveAvgPool3d"] = "torch.nn.AdaptiveAvgPool3d"
allowlist["torch.nn.AdaptiveAvgPool3d.__call__"] = "torch.Tensor"
allowlist["torch.nn.AdaptiveAvgPool3d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.AdaptiveAvgPool3d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AdaptiveAvgPool3d.train"] = "torch.nn.AdaptiveAvgPool3d"
allowlist["torch.nn.AdaptiveAvgPool3d.cuda"] = "torch.nn.AdaptiveAvgPool3d"
allowlist["torch.nn.AdaptiveAvgPool3d.cpu"] = "torch.nn.AdaptiveAvgPool3d"
allowlist[
    "torch.nn.AdaptiveAvgPool3d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.AdaptiveAvgPool3d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AdaptiveAvgPool3d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.AdaptiveMaxPool1d"] = "torch.nn.AdaptiveMaxPool1d"
allowlist["torch.nn.AdaptiveMaxPool1d.__call__"] = "torch.Tensor"
allowlist["torch.nn.AdaptiveMaxPool1d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.AdaptiveMaxPool1d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AdaptiveMaxPool1d.train"] = "torch.nn.AdaptiveMaxPool1d"
allowlist["torch.nn.AdaptiveMaxPool1d.cuda"] = "torch.nn.AdaptiveMaxPool1d"
allowlist["torch.nn.AdaptiveMaxPool1d.cpu"] = "torch.nn.AdaptiveMaxPool1d"
allowlist[
    "torch.nn.AdaptiveMaxPool1d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.AdaptiveMaxPool1d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AdaptiveMaxPool1d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.AdaptiveMaxPool2d"] = "torch.nn.AdaptiveMaxPool2d"
allowlist["torch.nn.AdaptiveMaxPool2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.AdaptiveMaxPool2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.AdaptiveMaxPool2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AdaptiveMaxPool2d.train"] = "torch.nn.AdaptiveMaxPool2d"
allowlist["torch.nn.AdaptiveMaxPool2d.cuda"] = "torch.nn.AdaptiveMaxPool2d"
allowlist["torch.nn.AdaptiveMaxPool2d.cpu"] = "torch.nn.AdaptiveMaxPool2d"
allowlist[
    "torch.nn.AdaptiveMaxPool2d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.AdaptiveMaxPool2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AdaptiveMaxPool2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.AdaptiveMaxPool3d"] = "torch.nn.AdaptiveMaxPool3d"
allowlist["torch.nn.AdaptiveMaxPool3d.__call__"] = "torch.Tensor"
allowlist["torch.nn.AdaptiveMaxPool3d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.AdaptiveMaxPool3d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AdaptiveMaxPool3d.train"] = "torch.nn.AdaptiveMaxPool3d"
allowlist["torch.nn.AdaptiveMaxPool3d.cuda"] = "torch.nn.AdaptiveMaxPool3d"
allowlist["torch.nn.AdaptiveMaxPool3d.cpu"] = "torch.nn.AdaptiveMaxPool3d"
allowlist[
    "torch.nn.AdaptiveMaxPool3d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.AdaptiveMaxPool3d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AdaptiveMaxPool3d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.AlphaDropout"] = "torch.nn.AlphaDropout"
allowlist["torch.nn.AlphaDropout.__call__"] = "torch.Tensor"
allowlist["torch.nn.AlphaDropout.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.AlphaDropout.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AlphaDropout.train"] = "torch.nn.AlphaDropout"
allowlist["torch.nn.AlphaDropout.cuda"] = "torch.nn.AlphaDropout"
allowlist["torch.nn.AlphaDropout.cpu"] = "torch.nn.AlphaDropout"
allowlist[
    "torch.nn.AlphaDropout.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.AlphaDropout.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AlphaDropout.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.AvgPool1d"] = "torch.nn.AvgPool1d"
allowlist["torch.nn.AvgPool1d.__call__"] = "torch.Tensor"
allowlist["torch.nn.AvgPool1d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.AvgPool1d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AvgPool1d.train"] = "torch.nn.AvgPool1d"
allowlist["torch.nn.AvgPool1d.cuda"] = "torch.nn.AvgPool1d"
allowlist["torch.nn.AvgPool1d.cpu"] = "torch.nn.AvgPool1d"
allowlist["torch.nn.AvgPool1d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.AvgPool1d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AvgPool1d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.AvgPool2d"] = "torch.nn.AvgPool2d"
allowlist["torch.nn.AvgPool2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.AvgPool2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.AvgPool2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AvgPool2d.train"] = "torch.nn.AvgPool2d"
allowlist["torch.nn.AvgPool2d.cuda"] = "torch.nn.AvgPool2d"
allowlist["torch.nn.AvgPool2d.cpu"] = "torch.nn.AvgPool2d"
allowlist["torch.nn.AvgPool2d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.AvgPool2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AvgPool2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.AvgPool3d"] = "torch.nn.AvgPool3d"
allowlist["torch.nn.AvgPool3d.__call__"] = "torch.Tensor"
allowlist["torch.nn.AvgPool3d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.AvgPool3d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AvgPool3d.train"] = "torch.nn.AvgPool3d"
allowlist["torch.nn.AvgPool3d.cuda"] = "torch.nn.AvgPool3d"
allowlist["torch.nn.AvgPool3d.cpu"] = "torch.nn.AvgPool3d"
allowlist["torch.nn.AvgPool3d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.AvgPool3d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.AvgPool3d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.BatchNorm1d"] = "torch.nn.BatchNorm1d"
allowlist["torch.nn.BatchNorm1d.__call__"] = "torch.Tensor"
allowlist["torch.nn.BatchNorm1d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.BatchNorm1d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.BatchNorm1d.train"] = "torch.nn.BatchNorm1d"
allowlist["torch.nn.BatchNorm1d.cuda"] = "torch.nn.BatchNorm1d"
allowlist["torch.nn.BatchNorm1d.cpu"] = "torch.nn.BatchNorm1d"
allowlist["torch.nn.BatchNorm1d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.BatchNorm1d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.BatchNorm1d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.BatchNorm2d"] = "torch.nn.BatchNorm2d"
allowlist["torch.nn.BatchNorm2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.BatchNorm2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.BatchNorm2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.BatchNorm2d.train"] = "torch.nn.BatchNorm2d"
allowlist["torch.nn.BatchNorm2d.cuda"] = "torch.nn.BatchNorm2d"
allowlist["torch.nn.BatchNorm2d.cpu"] = "torch.nn.BatchNorm2d"
allowlist["torch.nn.BatchNorm2d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.BatchNorm2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.BatchNorm2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.BatchNorm3d"] = "torch.nn.BatchNorm3d"
allowlist["torch.nn.BatchNorm3d.__call__"] = "torch.Tensor"
allowlist["torch.nn.BatchNorm3d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.BatchNorm3d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.BatchNorm3d.train"] = "torch.nn.BatchNorm3d"
allowlist["torch.nn.BatchNorm3d.cuda"] = "torch.nn.BatchNorm3d"
allowlist["torch.nn.BatchNorm3d.cpu"] = "torch.nn.BatchNorm3d"
allowlist["torch.nn.BatchNorm3d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.BatchNorm3d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.BatchNorm3d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Bilinear"] = "torch.nn.Bilinear"
allowlist["torch.nn.Bilinear.__call__"] = "torch.Tensor"
allowlist["torch.nn.Bilinear.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Bilinear.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Bilinear.train"] = "torch.nn.Bilinear"
allowlist["torch.nn.Bilinear.cuda"] = "torch.nn.Bilinear"
allowlist["torch.nn.Bilinear.cpu"] = "torch.nn.Bilinear"
allowlist["torch.nn.Bilinear.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Bilinear.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Bilinear.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.CELU"] = "torch.nn.CELU"
allowlist["torch.nn.CELU.__call__"] = "torch.Tensor"
allowlist["torch.nn.CELU.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.CELU.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.CELU.train"] = "torch.nn.CELU"
allowlist["torch.nn.CELU.cuda"] = "torch.nn.CELU"
allowlist["torch.nn.CELU.cpu"] = "torch.nn.CELU"
allowlist["torch.nn.CELU.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.CELU.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.CELU.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.ConstantPad1d"] = "torch.nn.ConstantPad1d"
allowlist["torch.nn.ConstantPad1d.__call__"] = "torch.Tensor"
allowlist["torch.nn.ConstantPad1d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.ConstantPad1d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ConstantPad1d.train"] = "torch.nn.ConstantPad1d"
allowlist["torch.nn.ConstantPad1d.cuda"] = "torch.nn.ConstantPad1d"
allowlist["torch.nn.ConstantPad1d.cpu"] = "torch.nn.ConstantPad1d"
allowlist[
    "torch.nn.ConstantPad1d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.ConstantPad1d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ConstantPad1d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.ConstantPad2d"] = "torch.nn.ConstantPad2d"
allowlist["torch.nn.ConstantPad2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.ConstantPad2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.ConstantPad2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ConstantPad2d.train"] = "torch.nn.ConstantPad2d"
allowlist["torch.nn.ConstantPad2d.cuda"] = "torch.nn.ConstantPad2d"
allowlist["torch.nn.ConstantPad2d.cpu"] = "torch.nn.ConstantPad2d"
allowlist[
    "torch.nn.ConstantPad2d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.ConstantPad2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ConstantPad2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.ConstantPad3d"] = "torch.nn.ConstantPad3d"
allowlist["torch.nn.ConstantPad3d.__call__"] = "torch.Tensor"
allowlist["torch.nn.ConstantPad3d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.ConstantPad3d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ConstantPad3d.train"] = "torch.nn.ConstantPad3d"
allowlist["torch.nn.ConstantPad3d.cuda"] = "torch.nn.ConstantPad3d"
allowlist["torch.nn.ConstantPad3d.cpu"] = "torch.nn.ConstantPad3d"
allowlist[
    "torch.nn.ConstantPad3d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.ConstantPad3d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ConstantPad3d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Container"] = "torch.nn.Container"
allowlist["torch.nn.Container.__call__"] = "torch.Tensor"
allowlist["torch.nn.Container.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Container.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Container.train"] = "torch.nn.Container"
allowlist["torch.nn.Container.cuda"] = "torch.nn.Container"
allowlist["torch.nn.Container.cpu"] = "torch.nn.Container"
allowlist["torch.nn.Container.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Container.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Container.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Conv1d"] = "torch.nn.Conv1d"
allowlist["torch.nn.Conv1d.__call__"] = "torch.Tensor"
allowlist["torch.nn.Conv1d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Conv1d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Conv1d.train"] = "torch.nn.Conv1d"
allowlist["torch.nn.Conv1d.cuda"] = "torch.nn.Conv1d"
allowlist["torch.nn.Conv1d.cpu"] = "torch.nn.Conv1d"
allowlist["torch.nn.Conv1d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Conv1d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Conv1d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Conv2d"] = "torch.nn.Conv2d"
allowlist["torch.nn.Conv2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.Conv2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Conv2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Conv2d.train"] = "torch.nn.Conv2d"
allowlist["torch.nn.Conv2d.cuda"] = "torch.nn.Conv2d"
allowlist["torch.nn.Conv2d.cpu"] = "torch.nn.Conv2d"
allowlist["torch.nn.Conv2d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Conv2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Conv2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Conv3d"] = "torch.nn.Conv3d"
allowlist["torch.nn.Conv3d.__call__"] = "torch.Tensor"
allowlist["torch.nn.Conv3d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Conv3d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Conv3d.train"] = "torch.nn.Conv3d"
allowlist["torch.nn.Conv3d.cuda"] = "torch.nn.Conv3d"
allowlist["torch.nn.Conv3d.cpu"] = "torch.nn.Conv3d"
allowlist["torch.nn.Conv3d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Conv3d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Conv3d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.ConvTranspose1d"] = "torch.nn.ConvTranspose1d"
allowlist["torch.nn.ConvTranspose1d.__call__"] = "torch.Tensor"
allowlist["torch.nn.ConvTranspose1d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.ConvTranspose1d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ConvTranspose1d.train"] = "torch.nn.ConvTranspose1d"
allowlist["torch.nn.ConvTranspose1d.cuda"] = "torch.nn.ConvTranspose1d"
allowlist["torch.nn.ConvTranspose1d.cpu"] = "torch.nn.ConvTranspose1d"
allowlist[
    "torch.nn.ConvTranspose1d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.ConvTranspose1d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ConvTranspose1d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.ConvTranspose2d"] = "torch.nn.ConvTranspose2d"
allowlist["torch.nn.ConvTranspose2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.ConvTranspose2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.ConvTranspose2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ConvTranspose2d.train"] = "torch.nn.ConvTranspose2d"
allowlist["torch.nn.ConvTranspose2d.cuda"] = "torch.nn.ConvTranspose2d"
allowlist["torch.nn.ConvTranspose2d.cpu"] = "torch.nn.ConvTranspose2d"
allowlist[
    "torch.nn.ConvTranspose2d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.ConvTranspose2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ConvTranspose2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.ConvTranspose3d"] = "torch.nn.ConvTranspose3d"
allowlist["torch.nn.ConvTranspose3d.__call__"] = "torch.Tensor"
allowlist["torch.nn.ConvTranspose3d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.ConvTranspose3d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ConvTranspose3d.train"] = "torch.nn.ConvTranspose3d"
allowlist["torch.nn.ConvTranspose3d.cuda"] = "torch.nn.ConvTranspose3d"
allowlist["torch.nn.ConvTranspose3d.cpu"] = "torch.nn.ConvTranspose3d"
allowlist[
    "torch.nn.ConvTranspose3d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.ConvTranspose3d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ConvTranspose3d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.CosineSimilarity"] = "torch.nn.CosineSimilarity"
allowlist["torch.nn.CosineSimilarity.__call__"] = "torch.Tensor"
allowlist["torch.nn.CosineSimilarity.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.CosineSimilarity.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.CosineSimilarity.train"] = "torch.nn.CosineSimilarity"
allowlist["torch.nn.CosineSimilarity.cuda"] = "torch.nn.CosineSimilarity"
allowlist["torch.nn.CosineSimilarity.cpu"] = "torch.nn.CosineSimilarity"
allowlist[
    "torch.nn.CosineSimilarity.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.CosineSimilarity.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.CosineSimilarity.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.CrossMapLRN2d"] = "torch.nn.CrossMapLRN2d"
allowlist["torch.nn.CrossMapLRN2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.CrossMapLRN2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.CrossMapLRN2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.CrossMapLRN2d.train"] = "torch.nn.CrossMapLRN2d"
allowlist["torch.nn.CrossMapLRN2d.cuda"] = "torch.nn.CrossMapLRN2d"
allowlist["torch.nn.CrossMapLRN2d.cpu"] = "torch.nn.CrossMapLRN2d"
allowlist[
    "torch.nn.CrossMapLRN2d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.CrossMapLRN2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.CrossMapLRN2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.DataParallel"] = "torch.nn.DataParallel"
allowlist["torch.nn.DataParallel.__call__"] = "torch.Tensor"
allowlist["torch.nn.DataParallel.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.DataParallel.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.DataParallel.train"] = "torch.nn.DataParallel"
allowlist["torch.nn.DataParallel.cuda"] = "torch.nn.DataParallel"
allowlist["torch.nn.DataParallel.cpu"] = "torch.nn.DataParallel"
allowlist[
    "torch.nn.DataParallel.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.DataParallel.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.DataParallel.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Dropout"] = "torch.nn.Dropout"
allowlist["torch.nn.Dropout.__call__"] = "torch.Tensor"
allowlist["torch.nn.Dropout.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Dropout.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Dropout.train"] = "torch.nn.Dropout"
allowlist["torch.nn.Dropout.cuda"] = "torch.nn.Dropout"
allowlist["torch.nn.Dropout.cpu"] = "torch.nn.Dropout"
allowlist["torch.nn.Dropout.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Dropout.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Dropout.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Dropout2d"] = "torch.nn.Dropout2d"
allowlist["torch.nn.Dropout2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.Dropout2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Dropout2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Dropout2d.train"] = "torch.nn.Dropout2d"
allowlist["torch.nn.Dropout2d.cuda"] = "torch.nn.Dropout2d"
allowlist["torch.nn.Dropout2d.cpu"] = "torch.nn.Dropout2d"
allowlist["torch.nn.Dropout2d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Dropout2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Dropout2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Dropout3d"] = "torch.nn.Dropout3d"
allowlist["torch.nn.Dropout3d.__call__"] = "torch.Tensor"
allowlist["torch.nn.Dropout3d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Dropout3d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Dropout3d.train"] = "torch.nn.Dropout3d"
allowlist["torch.nn.Dropout3d.cuda"] = "torch.nn.Dropout3d"
allowlist["torch.nn.Dropout3d.cpu"] = "torch.nn.Dropout3d"
allowlist["torch.nn.Dropout3d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Dropout3d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Dropout3d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.ELU"] = "torch.nn.ELU"
allowlist["torch.nn.ELU.__call__"] = "torch.Tensor"
allowlist["torch.nn.ELU.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.ELU.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ELU.train"] = "torch.nn.ELU"
allowlist["torch.nn.ELU.cuda"] = "torch.nn.ELU"
allowlist["torch.nn.ELU.cpu"] = "torch.nn.ELU"
allowlist["torch.nn.ELU.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.ELU.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ELU.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Embedding"] = "torch.nn.Embedding"
allowlist["torch.nn.Embedding.__call__"] = "torch.Tensor"
allowlist["torch.nn.Embedding.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Embedding.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Embedding.train"] = "torch.nn.Embedding"
allowlist["torch.nn.Embedding.cuda"] = "torch.nn.Embedding"
allowlist["torch.nn.Embedding.cpu"] = "torch.nn.Embedding"
allowlist["torch.nn.Embedding.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Embedding.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Embedding.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.EmbeddingBag"] = "torch.nn.EmbeddingBag"
allowlist["torch.nn.EmbeddingBag.__call__"] = "torch.Tensor"
allowlist["torch.nn.EmbeddingBag.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.EmbeddingBag.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.EmbeddingBag.train"] = "torch.nn.EmbeddingBag"
allowlist["torch.nn.EmbeddingBag.cuda"] = "torch.nn.EmbeddingBag"
allowlist["torch.nn.EmbeddingBag.cpu"] = "torch.nn.EmbeddingBag"
allowlist[
    "torch.nn.EmbeddingBag.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.EmbeddingBag.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.EmbeddingBag.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.FeatureAlphaDropout"] = "torch.nn.FeatureAlphaDropout"
allowlist["torch.nn.FeatureAlphaDropout.__call__"] = "torch.Tensor"
allowlist["torch.nn.FeatureAlphaDropout.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.FeatureAlphaDropout.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.FeatureAlphaDropout.train"] = "torch.nn.FeatureAlphaDropout"
allowlist["torch.nn.FeatureAlphaDropout.cuda"] = "torch.nn.FeatureAlphaDropout"
allowlist["torch.nn.FeatureAlphaDropout.cpu"] = "torch.nn.FeatureAlphaDropout"
allowlist[
    "torch.nn.FeatureAlphaDropout.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.FeatureAlphaDropout.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.FeatureAlphaDropout.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Flatten"] = "torch.nn.Flatten"
allowlist["torch.nn.Flatten.__call__"] = "torch.Tensor"
allowlist["torch.nn.Flatten.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Flatten.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Flatten.train"] = "torch.nn.Flatten"
allowlist["torch.nn.Flatten.cuda"] = "torch.nn.Flatten"
allowlist["torch.nn.Flatten.cpu"] = "torch.nn.Flatten"
allowlist["torch.nn.Flatten.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Flatten.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Flatten.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Fold"] = "torch.nn.Fold"
allowlist["torch.nn.Fold.__call__"] = "torch.Tensor"
allowlist["torch.nn.Fold.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Fold.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Fold.train"] = "torch.nn.Fold"
allowlist["torch.nn.Fold.cuda"] = "torch.nn.Fold"
allowlist["torch.nn.Fold.cpu"] = "torch.nn.Fold"
allowlist["torch.nn.Fold.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Fold.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Fold.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.FractionalMaxPool2d"] = "torch.nn.FractionalMaxPool2d"
allowlist["torch.nn.FractionalMaxPool2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.FractionalMaxPool2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.FractionalMaxPool2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.FractionalMaxPool2d.train"] = "torch.nn.FractionalMaxPool2d"
allowlist["torch.nn.FractionalMaxPool2d.cuda"] = "torch.nn.FractionalMaxPool2d"
allowlist["torch.nn.FractionalMaxPool2d.cpu"] = "torch.nn.FractionalMaxPool2d"
allowlist[
    "torch.nn.FractionalMaxPool2d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.FractionalMaxPool2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.FractionalMaxPool2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.FractionalMaxPool3d"] = "torch.nn.FractionalMaxPool3d"
allowlist["torch.nn.FractionalMaxPool3d.__call__"] = "torch.Tensor"
allowlist["torch.nn.FractionalMaxPool3d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.FractionalMaxPool3d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.FractionalMaxPool3d.train"] = "torch.nn.FractionalMaxPool3d"
allowlist["torch.nn.FractionalMaxPool3d.cuda"] = "torch.nn.FractionalMaxPool3d"
allowlist["torch.nn.FractionalMaxPool3d.cpu"] = "torch.nn.FractionalMaxPool3d"
allowlist[
    "torch.nn.FractionalMaxPool3d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.FractionalMaxPool3d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.FractionalMaxPool3d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.GELU"] = "torch.nn.GELU"
allowlist["torch.nn.GELU.__call__"] = "torch.Tensor"
allowlist["torch.nn.GELU.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.GELU.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.GELU.train"] = "torch.nn.GELU"
allowlist["torch.nn.GELU.cuda"] = "torch.nn.GELU"
allowlist["torch.nn.GELU.cpu"] = "torch.nn.GELU"
allowlist["torch.nn.GELU.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.GELU.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.GELU.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.GLU"] = "torch.nn.GLU"
allowlist["torch.nn.GLU.__call__"] = "torch.Tensor"
allowlist["torch.nn.GLU.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.GLU.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.GLU.train"] = "torch.nn.GLU"
allowlist["torch.nn.GLU.cuda"] = "torch.nn.GLU"
allowlist["torch.nn.GLU.cpu"] = "torch.nn.GLU"
allowlist["torch.nn.GLU.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.GLU.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.GLU.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.GRU"] = "torch.nn.GRU"
allowlist["torch.nn.GRU.__call__"] = "torch.Tensor"
allowlist["torch.nn.GRU.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.GRU.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.GRU.train"] = "torch.nn.GRU"
allowlist["torch.nn.GRU.cuda"] = "torch.nn.GRU"
allowlist["torch.nn.GRU.cpu"] = "torch.nn.GRU"
allowlist["torch.nn.GRU.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.GRU.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.GRU.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.GRUCell"] = "torch.nn.GRUCell"
allowlist["torch.nn.GRUCell.__call__"] = "torch.Tensor"
allowlist["torch.nn.GRUCell.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.GRUCell.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.GRUCell.train"] = "torch.nn.GRUCell"
allowlist["torch.nn.GRUCell.cuda"] = "torch.nn.GRUCell"
allowlist["torch.nn.GRUCell.cpu"] = "torch.nn.GRUCell"
allowlist["torch.nn.GRUCell.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.GRUCell.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.GRUCell.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.GroupNorm"] = "torch.nn.GroupNorm"
allowlist["torch.nn.GroupNorm.__call__"] = "torch.Tensor"
allowlist["torch.nn.GroupNorm.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.GroupNorm.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.GroupNorm.train"] = "torch.nn.GroupNorm"
allowlist["torch.nn.GroupNorm.cuda"] = "torch.nn.GroupNorm"
allowlist["torch.nn.GroupNorm.cpu"] = "torch.nn.GroupNorm"
allowlist["torch.nn.GroupNorm.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.GroupNorm.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.GroupNorm.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Hardshrink"] = "torch.nn.Hardshrink"
allowlist["torch.nn.Hardshrink.__call__"] = "torch.Tensor"
allowlist["torch.nn.Hardshrink.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Hardshrink.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Hardshrink.train"] = "torch.nn.Hardshrink"
allowlist["torch.nn.Hardshrink.cuda"] = "torch.nn.Hardshrink"
allowlist["torch.nn.Hardshrink.cpu"] = "torch.nn.Hardshrink"
allowlist["torch.nn.Hardshrink.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Hardshrink.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Hardshrink.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Hardsigmoid"] = {
    "return_type": "torch.nn.Hardsigmoid",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Hardsigmoid.__call__"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Hardsigmoid.parameters"] = {
    "return_type": "syft.lib.python.List",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Hardsigmoid.train"] = {
    "return_type": "torch.nn.Hardsigmoid",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Hardsigmoid.cuda"] = {
    "return_type": "torch.nn.Hardsigmoid",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Hardsigmoid.cpu"] = {
    "return_type": "torch.nn.Hardsigmoid",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Hardsigmoid.state_dict"] = {
    "return_type": "syft.lib.python.collections.OrderedDict",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Hardsigmoid.load_state_dict"] = {
    "return_type": "syft.lib.python._SyNone",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Hardsigmoid.extra_repr"] = {
    "return_type": "syft.lib.python.String",
    "min_version": "1.5.0",
}


allowlist["torch.nn.Hardswish"] = {  # exists in # 1.6.0 +
    "return_type": "torch.nn.Hardswish",
    "min_version": "1.6.0",
}

allowlist["torch.nn.Hardswish.__call__"] = {  # exists in # 1.6.0 +
    "return_type": "torch.Tensor",
    "min_version": "1.6.0",
}

allowlist["torch.nn.Hardswish.parameters"] = {  # exists in # 1.6.0 +
    "return_type": "syft.lib.python.List",
    "min_version": "1.6.0",
}

allowlist["torch.nn.Hardswish.train"] = {  # exists in # 1.6.0 +
    "return_type": "torch.nn.Hardswish",
    "min_version": "1.6.0",
}
allowlist["torch.nn.Hardswish.cuda"] = {  # exists in # 1.6.0 +
    "return_type": "torch.nn.Hardswish",
    "min_version": "1.6.0",
}
allowlist["torch.nn.Hardswish.cpu"] = {  # exists in # 1.6.0 +
    "return_type": "torch.nn.Hardswish",
    "min_version": "1.6.0",
}
allowlist["torch.nn.Hardswish.state_dict"] = {  # exists in # 1.6.0 +
    "return_type": "syft.lib.python.collections.OrderedDict",
    "min_version": "1.6.0",
}
allowlist["torch.nn.Hardswish.load_state_dict"] = {  # exists in # 1.6.0 +
    "return_type": "syft.lib.python._SyNone",
    "min_version": "1.6.0",
}
allowlist["torch.nn.Hardswish.extra_repr"] = {  # exists in # 1.6.0 +
    "return_type": "syft.lib.python.String",
    "min_version": "1.6.0",
}

allowlist["torch.nn.Hardtanh"] = "torch.nn.Hardtanh"
allowlist["torch.nn.Hardtanh.__call__"] = "torch.Tensor"
allowlist["torch.nn.Hardtanh.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Hardtanh.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Hardtanh.train"] = "torch.nn.Hardtanh"
allowlist["torch.nn.Hardtanh.cuda"] = "torch.nn.Hardtanh"
allowlist["torch.nn.Hardtanh.cpu"] = "torch.nn.Hardtanh"
allowlist["torch.nn.Hardtanh.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Hardtanh.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Hardtanh.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Identity"] = "torch.nn.Identity"
allowlist["torch.nn.Identity.__call__"] = "torch.Tensor"
allowlist["torch.nn.Identity.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Identity.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Identity.train"] = "torch.nn.Identity"
allowlist["torch.nn.Identity.cuda"] = "torch.nn.Identity"
allowlist["torch.nn.Identity.cpu"] = "torch.nn.Identity"
allowlist["torch.nn.Identity.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Identity.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Identity.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.InstanceNorm1d"] = "torch.nn.InstanceNorm1d"
allowlist["torch.nn.InstanceNorm1d.__call__"] = "torch.Tensor"
allowlist["torch.nn.InstanceNorm1d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.InstanceNorm1d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.InstanceNorm1d.train"] = "torch.nn.InstanceNorm1d"
allowlist["torch.nn.InstanceNorm1d.cuda"] = "torch.nn.InstanceNorm1d"
allowlist["torch.nn.InstanceNorm1d.cpu"] = "torch.nn.InstanceNorm1d"
allowlist[
    "torch.nn.InstanceNorm1d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.InstanceNorm1d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.InstanceNorm1d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.InstanceNorm2d"] = "torch.nn.InstanceNorm2d"
allowlist["torch.nn.InstanceNorm2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.InstanceNorm2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.InstanceNorm2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.InstanceNorm2d.train"] = "torch.nn.InstanceNorm2d"
allowlist["torch.nn.InstanceNorm2d.cuda"] = "torch.nn.InstanceNorm2d"
allowlist["torch.nn.InstanceNorm2d.cpu"] = "torch.nn.InstanceNorm2d"
allowlist[
    "torch.nn.InstanceNorm2d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.InstanceNorm2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.InstanceNorm2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.InstanceNorm3d"] = "torch.nn.InstanceNorm3d"
allowlist["torch.nn.InstanceNorm3d.__call__"] = "torch.Tensor"
allowlist["torch.nn.InstanceNorm3d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.InstanceNorm3d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.InstanceNorm3d.train"] = "torch.nn.InstanceNorm3d"
allowlist["torch.nn.InstanceNorm3d.cuda"] = "torch.nn.InstanceNorm3d"
allowlist["torch.nn.InstanceNorm3d.cpu"] = "torch.nn.InstanceNorm3d"
allowlist[
    "torch.nn.InstanceNorm3d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.InstanceNorm3d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.InstanceNorm3d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.LPPool1d"] = "torch.nn.LPPool1d"
allowlist["torch.nn.LPPool1d.__call__"] = "torch.Tensor"
allowlist["torch.nn.LPPool1d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.LPPool1d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.LPPool1d.train"] = "torch.nn.LPPool1d"
allowlist["torch.nn.LPPool1d.cuda"] = "torch.nn.LPPool1d"
allowlist["torch.nn.LPPool1d.cpu"] = "torch.nn.LPPool1d"
allowlist["torch.nn.LPPool1d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.LPPool1d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.LPPool1d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.LPPool2d"] = "torch.nn.LPPool2d"
allowlist["torch.nn.LPPool2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.LPPool2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.LPPool2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.LPPool2d.train"] = "torch.nn.LPPool2d"
allowlist["torch.nn.LPPool2d.cuda"] = "torch.nn.LPPool2d"
allowlist["torch.nn.LPPool2d.cpu"] = "torch.nn.LPPool2d"
allowlist["torch.nn.LPPool2d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.LPPool2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.LPPool2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.LSTM"] = "torch.nn.LSTM"
allowlist["torch.nn.LSTM.__call__"] = "torch.Tensor"
allowlist["torch.nn.LSTM.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.LSTM.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.LSTM.train"] = "torch.nn.LSTM"
allowlist["torch.nn.LSTM.cuda"] = "torch.nn.LSTM"
allowlist["torch.nn.LSTM.cpu"] = "torch.nn.LSTM"
allowlist["torch.nn.LSTM.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.LSTM.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.LSTM.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.LSTMCell"] = "torch.nn.LSTMCell"
allowlist["torch.nn.LSTMCell.__call__"] = "torch.Tensor"
allowlist["torch.nn.LSTMCell.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.LSTMCell.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.LSTMCell.train"] = "torch.nn.LSTMCell"
allowlist["torch.nn.LSTMCell.cuda"] = "torch.nn.LSTMCell"
allowlist["torch.nn.LSTMCell.cpu"] = "torch.nn.LSTMCell"
allowlist["torch.nn.LSTMCell.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.LSTMCell.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.LSTMCell.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.LayerNorm"] = "torch.nn.LayerNorm"
allowlist["torch.nn.LayerNorm.__call__"] = "torch.Tensor"
allowlist["torch.nn.LayerNorm.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.LayerNorm.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.LayerNorm.train"] = "torch.nn.LayerNorm"
allowlist["torch.nn.LayerNorm.cuda"] = "torch.nn.LayerNorm"
allowlist["torch.nn.LayerNorm.cpu"] = "torch.nn.LayerNorm"
allowlist["torch.nn.LayerNorm.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.LayerNorm.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.LayerNorm.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.LeakyReLU"] = "torch.nn.LeakyReLU"
allowlist["torch.nn.LeakyReLU.__call__"] = "torch.Tensor"
allowlist["torch.nn.LeakyReLU.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.LeakyReLU.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.LeakyReLU.train"] = "torch.nn.LeakyReLU"
allowlist["torch.nn.LeakyReLU.cuda"] = "torch.nn.LeakyReLU"
allowlist["torch.nn.LeakyReLU.cpu"] = "torch.nn.LeakyReLU"
allowlist["torch.nn.LeakyReLU.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.LeakyReLU.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.LeakyReLU.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Linear"] = "torch.nn.Linear"
allowlist["torch.nn.Linear.__call__"] = "torch.Tensor"
allowlist["torch.nn.Linear.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Linear.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Linear.train"] = "torch.nn.Linear"
allowlist["torch.nn.Linear.cuda"] = "torch.nn.Linear"
allowlist["torch.nn.Linear.cpu"] = "torch.nn.Linear"
allowlist["torch.nn.Linear.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Linear.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Linear.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.LocalResponseNorm"] = "torch.nn.LocalResponseNorm"
allowlist["torch.nn.LocalResponseNorm.__call__"] = "torch.Tensor"
allowlist["torch.nn.LocalResponseNorm.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.LocalResponseNorm.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.LocalResponseNorm.train"] = "torch.nn.LocalResponseNorm"
allowlist["torch.nn.LocalResponseNorm.cuda"] = "torch.nn.LocalResponseNorm"
allowlist["torch.nn.LocalResponseNorm.cpu"] = "torch.nn.LocalResponseNorm"
allowlist[
    "torch.nn.LocalResponseNorm.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.LocalResponseNorm.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.LocalResponseNorm.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.LogSigmoid"] = "torch.nn.LogSigmoid"
allowlist["torch.nn.LogSigmoid.__call__"] = "torch.Tensor"
allowlist["torch.nn.LogSigmoid.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.LogSigmoid.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.LogSigmoid.train"] = "torch.nn.LogSigmoid"
allowlist["torch.nn.LogSigmoid.cuda"] = "torch.nn.LogSigmoid"
allowlist["torch.nn.LogSigmoid.cpu"] = "torch.nn.LogSigmoid"
allowlist["torch.nn.LogSigmoid.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.LogSigmoid.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.LogSigmoid.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.LogSoftmax"] = "torch.nn.LogSoftmax"
allowlist["torch.nn.LogSoftmax.__call__"] = "torch.Tensor"
allowlist["torch.nn.LogSoftmax.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.LogSoftmax.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.LogSoftmax.train"] = "torch.nn.LogSoftmax"
allowlist["torch.nn.LogSoftmax.cuda"] = "torch.nn.LogSoftmax"
allowlist["torch.nn.LogSoftmax.cpu"] = "torch.nn.LogSoftmax"
allowlist["torch.nn.LogSoftmax.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.LogSoftmax.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.LogSoftmax.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.MaxPool1d"] = "torch.nn.MaxPool1d"
allowlist["torch.nn.MaxPool1d.__call__"] = "torch.Tensor"
allowlist["torch.nn.MaxPool1d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.MaxPool1d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.MaxPool1d.train"] = "torch.nn.MaxPool1d"
allowlist["torch.nn.MaxPool1d.cuda"] = "torch.nn.MaxPool1d"
allowlist["torch.nn.MaxPool1d.cpu"] = "torch.nn.MaxPool1d"
allowlist["torch.nn.MaxPool1d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.MaxPool1d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.MaxPool1d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.MaxPool2d"] = "torch.nn.MaxPool2d"
allowlist["torch.nn.MaxPool2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.MaxPool2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.MaxPool2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.MaxPool2d.train"] = "torch.nn.MaxPool2d"
allowlist["torch.nn.MaxPool2d.cuda"] = "torch.nn.MaxPool2d"
allowlist["torch.nn.MaxPool2d.cpu"] = "torch.nn.MaxPool2d"
allowlist["torch.nn.MaxPool2d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.MaxPool2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.MaxPool2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.MaxPool3d"] = "torch.nn.MaxPool3d"
allowlist["torch.nn.MaxPool3d.__call__"] = "torch.Tensor"
allowlist["torch.nn.MaxPool3d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.MaxPool3d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.MaxPool3d.train"] = "torch.nn.MaxPool3d"
allowlist["torch.nn.MaxPool3d.cuda"] = "torch.nn.MaxPool3d"
allowlist["torch.nn.MaxPool3d.cpu"] = "torch.nn.MaxPool3d"
allowlist["torch.nn.MaxPool3d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.MaxPool3d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.MaxPool3d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.MaxUnpool1d"] = "torch.nn.MaxUnpool1d"
allowlist["torch.nn.MaxUnpool1d.__call__"] = "torch.Tensor"
allowlist["torch.nn.MaxUnpool1d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.MaxUnpool1d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.MaxUnpool1d.train"] = "torch.nn.MaxUnpool1d"
allowlist["torch.nn.MaxUnpool1d.cuda"] = "torch.nn.MaxUnpool1d"
allowlist["torch.nn.MaxUnpool1d.cpu"] = "torch.nn.MaxUnpool1d"
allowlist["torch.nn.MaxUnpool1d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.MaxUnpool1d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.MaxUnpool1d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.MaxUnpool2d"] = "torch.nn.MaxUnpool2d"
allowlist["torch.nn.MaxUnpool2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.MaxUnpool2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.MaxUnpool2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.MaxUnpool2d.train"] = "torch.nn.MaxUnpool2d"
allowlist["torch.nn.MaxUnpool2d.cuda"] = "torch.nn.MaxUnpool2d"
allowlist["torch.nn.MaxUnpool2d.cpu"] = "torch.nn.MaxUnpool2d"
allowlist["torch.nn.MaxUnpool2d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.MaxUnpool2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.MaxUnpool2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.MaxUnpool3d"] = "torch.nn.MaxUnpool3d"
allowlist["torch.nn.MaxUnpool3d.__call__"] = "torch.Tensor"
allowlist["torch.nn.MaxUnpool3d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.MaxUnpool3d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.MaxUnpool3d.train"] = "torch.nn.MaxUnpool3d"
allowlist["torch.nn.MaxUnpool3d.cuda"] = "torch.nn.MaxUnpool3d"
allowlist["torch.nn.MaxUnpool3d.cpu"] = "torch.nn.MaxUnpool3d"
allowlist["torch.nn.MaxUnpool3d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.MaxUnpool3d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.MaxUnpool3d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Module"] = "torch.nn.Module"
allowlist["torch.nn.Module.__call__"] = "torch.Tensor"
allowlist["torch.nn.Module.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Module.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Module.train"] = "torch.nn.Module"
allowlist["torch.nn.Module.cuda"] = "torch.nn.Module"
allowlist["torch.nn.Module.cpu"] = "torch.nn.Module"
allowlist["torch.nn.Module.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Module.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Module.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.ModuleDict"] = "torch.nn.ModuleDict"
allowlist["torch.nn.ModuleDict.__call__"] = "torch.Tensor"
allowlist["torch.nn.ModuleDict.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.ModuleDict.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ModuleDict.train"] = "torch.nn.ModuleDict"
allowlist["torch.nn.ModuleDict.cuda"] = "torch.nn.ModuleDict"
allowlist["torch.nn.ModuleDict.cpu"] = "torch.nn.ModuleDict"
allowlist["torch.nn.ModuleDict.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.ModuleDict.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ModuleDict.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.ModuleList"] = "torch.nn.ModuleList"
allowlist["torch.nn.ModuleList.__call__"] = "torch.Tensor"
allowlist["torch.nn.ModuleList.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.ModuleList.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ModuleList.train"] = "torch.nn.ModuleList"
allowlist["torch.nn.ModuleList.cuda"] = "torch.nn.ModuleList"
allowlist["torch.nn.ModuleList.cpu"] = "torch.nn.ModuleList"
allowlist["torch.nn.ModuleList.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.ModuleList.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ModuleList.extra_repr"] = "syft.lib.python.String"
allowlist["torch.nn.ModuleList.__len__"] = "syft.lib.python.Int"
allowlist["torch.nn.ModuleList.__getitem__"] = "torch.nn.Module"

allowlist["torch.nn.MultiheadAttention"] = "torch.nn.MultiheadAttention"
allowlist["torch.nn.MultiheadAttention.__call__"] = "torch.Tensor"
allowlist["torch.nn.MultiheadAttention.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.MultiheadAttention.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.MultiheadAttention.train"] = "torch.nn.MultiheadAttention"
allowlist["torch.nn.MultiheadAttention.cuda"] = "torch.nn.MultiheadAttention"
allowlist["torch.nn.MultiheadAttention.cpu"] = "torch.nn.MultiheadAttention"
allowlist[
    "torch.nn.MultiheadAttention.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.MultiheadAttention.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.MultiheadAttention.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.PReLU"] = "torch.nn.PReLU"
allowlist["torch.nn.PReLU.__call__"] = "torch.Tensor"
allowlist["torch.nn.PReLU.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.PReLU.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.PReLU.train"] = "torch.nn.PReLU"
allowlist["torch.nn.PReLU.cuda"] = "torch.nn.PReLU"
allowlist["torch.nn.PReLU.cpu"] = "torch.nn.PReLU"
allowlist["torch.nn.PReLU.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.PReLU.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.PReLU.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.PairwiseDistance"] = "torch.nn.PairwiseDistance"
allowlist["torch.nn.PairwiseDistance.__call__"] = "torch.Tensor"
allowlist["torch.nn.PairwiseDistance.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.PairwiseDistance.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.PairwiseDistance.train"] = "torch.nn.PairwiseDistance"
allowlist["torch.nn.PairwiseDistance.cuda"] = "torch.nn.PairwiseDistance"
allowlist["torch.nn.PairwiseDistance.cpu"] = "torch.nn.PairwiseDistance"
allowlist[
    "torch.nn.PairwiseDistance.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.PairwiseDistance.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.PairwiseDistance.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.PixelShuffle"] = "torch.nn.PixelShuffle"
allowlist["torch.nn.PixelShuffle.__call__"] = "torch.Tensor"
allowlist["torch.nn.PixelShuffle.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.PixelShuffle.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.PixelShuffle.train"] = "torch.nn.PixelShuffle"
allowlist["torch.nn.PixelShuffle.cuda"] = "torch.nn.PixelShuffle"
allowlist["torch.nn.PixelShuffle.cpu"] = "torch.nn.PixelShuffle"
allowlist[
    "torch.nn.PixelShuffle.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.PixelShuffle.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.PixelShuffle.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.RNN"] = "torch.nn.RNN"
allowlist["torch.nn.RNN.__call__"] = "torch.Tensor"
allowlist["torch.nn.RNN.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.RNN.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.RNN.train"] = "torch.nn.RNN"
allowlist["torch.nn.RNN.cuda"] = "torch.nn.RNN"
allowlist["torch.nn.RNN.cpu"] = "torch.nn.RNN"
allowlist["torch.nn.RNN.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.RNN.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.RNN.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.RNNBase"] = "torch.nn.RNNBase"
allowlist["torch.nn.RNNBase.__call__"] = "torch.Tensor"
allowlist["torch.nn.RNNBase.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.RNNBase.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.RNNBase.train"] = "torch.nn.RNNBase"
allowlist["torch.nn.RNNBase.cuda"] = "torch.nn.RNNBase"
allowlist["torch.nn.RNNBase.cpu"] = "torch.nn.RNNBase"
allowlist["torch.nn.RNNBase.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.RNNBase.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.RNNBase.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.RNNCell"] = "torch.nn.RNNCell"
allowlist["torch.nn.RNNCell.__call__"] = "torch.Tensor"
allowlist["torch.nn.RNNCell.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.RNNCell.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.RNNCell.train"] = "torch.nn.RNNCell"
allowlist["torch.nn.RNNCell.cuda"] = "torch.nn.RNNCell"
allowlist["torch.nn.RNNCell.cpu"] = "torch.nn.RNNCell"
allowlist["torch.nn.RNNCell.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.RNNCell.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.RNNCell.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.RNNCellBase"] = "torch.nn.RNNCellBase"
allowlist["torch.nn.RNNCellBase.__call__"] = "torch.Tensor"
allowlist["torch.nn.RNNCellBase.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.RNNCellBase.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.RNNCellBase.train"] = "torch.nn.RNNCellBase"
allowlist["torch.nn.RNNCellBase.cuda"] = "torch.nn.RNNCellBase"
allowlist["torch.nn.RNNCellBase.cpu"] = "torch.nn.RNNCellBase"
allowlist["torch.nn.RNNCellBase.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.RNNCellBase.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.RNNCellBase.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.RReLU"] = "torch.nn.RReLU"
allowlist["torch.nn.RReLU.__call__"] = "torch.Tensor"
allowlist["torch.nn.RReLU.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.RReLU.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.RReLU.train"] = "torch.nn.RReLU"
allowlist["torch.nn.RReLU.cuda"] = "torch.nn.RReLU"
allowlist["torch.nn.RReLU.cpu"] = "torch.nn.RReLU"
allowlist["torch.nn.RReLU.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.RReLU.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.RReLU.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.ReLU"] = "torch.nn.ReLU"
allowlist["torch.nn.ReLU.__call__"] = "torch.Tensor"
allowlist["torch.nn.ReLU.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.ReLU.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ReLU.train"] = "torch.nn.ReLU"
allowlist["torch.nn.ReLU.cuda"] = "torch.nn.ReLU"
allowlist["torch.nn.ReLU.cpu"] = "torch.nn.ReLU"
allowlist["torch.nn.ReLU.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.ReLU.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ReLU.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.ReLU6"] = "torch.nn.ReLU6"
allowlist["torch.nn.ReLU6.__call__"] = "torch.Tensor"
allowlist["torch.nn.ReLU6.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.ReLU6.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ReLU6.train"] = "torch.nn.ReLU6"
allowlist["torch.nn.ReLU6.cuda"] = "torch.nn.ReLU6"
allowlist["torch.nn.ReLU6.cpu"] = "torch.nn.ReLU6"
allowlist["torch.nn.ReLU6.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.ReLU6.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ReLU6.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.ReflectionPad1d"] = "torch.nn.ReflectionPad1d"
allowlist["torch.nn.ReflectionPad1d.__call__"] = "torch.Tensor"
allowlist["torch.nn.ReflectionPad1d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.ReflectionPad1d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ReflectionPad1d.train"] = "torch.nn.ReflectionPad1d"
allowlist["torch.nn.ReflectionPad1d.cuda"] = "torch.nn.ReflectionPad1d"
allowlist["torch.nn.ReflectionPad1d.cpu"] = "torch.nn.ReflectionPad1d"
allowlist[
    "torch.nn.ReflectionPad1d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.ReflectionPad1d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ReflectionPad1d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.ReflectionPad2d"] = "torch.nn.ReflectionPad2d"
allowlist["torch.nn.ReflectionPad2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.ReflectionPad2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.ReflectionPad2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ReflectionPad2d.train"] = "torch.nn.ReflectionPad2d"
allowlist["torch.nn.ReflectionPad2d.cuda"] = "torch.nn.ReflectionPad2d"
allowlist["torch.nn.ReflectionPad2d.cpu"] = "torch.nn.ReflectionPad2d"
allowlist[
    "torch.nn.ReflectionPad2d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.ReflectionPad2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ReflectionPad2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.ReplicationPad1d"] = "torch.nn.ReplicationPad1d"
allowlist["torch.nn.ReplicationPad1d.__call__"] = "torch.Tensor"
allowlist["torch.nn.ReplicationPad1d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.ReplicationPad1d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ReplicationPad1d.train"] = "torch.nn.ReplicationPad1d"
allowlist["torch.nn.ReplicationPad1d.cuda"] = "torch.nn.ReplicationPad1d"
allowlist["torch.nn.ReplicationPad1d.cpu"] = "torch.nn.ReplicationPad1d"
allowlist[
    "torch.nn.ReplicationPad1d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.ReplicationPad1d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ReplicationPad1d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.ReplicationPad2d"] = "torch.nn.ReplicationPad2d"
allowlist["torch.nn.ReplicationPad2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.ReplicationPad2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.ReplicationPad2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ReplicationPad2d.train"] = "torch.nn.ReplicationPad2d"
allowlist["torch.nn.ReplicationPad2d.cuda"] = "torch.nn.ReplicationPad2d"
allowlist["torch.nn.ReplicationPad2d.cpu"] = "torch.nn.ReplicationPad2d"
allowlist[
    "torch.nn.ReplicationPad2d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.ReplicationPad2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ReplicationPad2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.ReplicationPad3d"] = "torch.nn.ReplicationPad3d"
allowlist["torch.nn.ReplicationPad3d.__call__"] = "torch.Tensor"
allowlist["torch.nn.ReplicationPad3d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.ReplicationPad3d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ReplicationPad3d.train"] = "torch.nn.ReplicationPad3d"
allowlist["torch.nn.ReplicationPad3d.cuda"] = "torch.nn.ReplicationPad3d"
allowlist["torch.nn.ReplicationPad3d.cpu"] = "torch.nn.ReplicationPad3d"
allowlist[
    "torch.nn.ReplicationPad3d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.ReplicationPad3d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ReplicationPad3d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.SELU"] = "torch.nn.SELU"
allowlist["torch.nn.SELU.__call__"] = "torch.Tensor"
allowlist["torch.nn.SELU.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.SELU.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.SELU.train"] = "torch.nn.SELU"
allowlist["torch.nn.SELU.cuda"] = "torch.nn.SELU"
allowlist["torch.nn.SELU.cpu"] = "torch.nn.SELU"
allowlist["torch.nn.SELU.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.SELU.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.SELU.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Sequential"] = "torch.nn.Sequential"
allowlist["torch.nn.Sequential.__call__"] = "torch.Tensor"
allowlist["torch.nn.Sequential.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Sequential.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Sequential.train"] = "torch.nn.Sequential"
allowlist["torch.nn.Sequential.cuda"] = "torch.nn.Sequential"
allowlist["torch.nn.Sequential.cpu"] = "torch.nn.Sequential"
allowlist["torch.nn.Sequential.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Sequential.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Sequential.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Sigmoid"] = "torch.nn.Sigmoid"
allowlist["torch.nn.Sigmoid.__call__"] = "torch.Tensor"
allowlist["torch.nn.Sigmoid.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Sigmoid.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Sigmoid.train"] = "torch.nn.Sigmoid"
allowlist["torch.nn.Sigmoid.cuda"] = "torch.nn.Sigmoid"
allowlist["torch.nn.Sigmoid.cpu"] = "torch.nn.Sigmoid"
allowlist["torch.nn.Sigmoid.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Sigmoid.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Sigmoid.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Softmax"] = "torch.nn.Softmax"
allowlist["torch.nn.Softmax.__call__"] = "torch.Tensor"
allowlist["torch.nn.Softmax.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Softmax.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Softmax.train"] = "torch.nn.Softmax"
allowlist["torch.nn.Softmax.cuda"] = "torch.nn.Softmax"
allowlist["torch.nn.Softmax.cpu"] = "torch.nn.Softmax"
allowlist["torch.nn.Softmax.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Softmax.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Softmax.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Softmax2d"] = "torch.nn.Softmax2d"
allowlist["torch.nn.Softmax2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.Softmax2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Softmax2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Softmax2d.train"] = "torch.nn.Softmax2d"
allowlist["torch.nn.Softmax2d.cuda"] = "torch.nn.Softmax2d"
allowlist["torch.nn.Softmax2d.cpu"] = "torch.nn.Softmax2d"
allowlist["torch.nn.Softmax2d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Softmax2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Softmax2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Softmin"] = "torch.nn.Softmin"
allowlist["torch.nn.Softmin.__call__"] = "torch.Tensor"
allowlist["torch.nn.Softmin.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Softmin.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Softmin.train"] = "torch.nn.Softmin"
allowlist["torch.nn.Softmin.cuda"] = "torch.nn.Softmin"
allowlist["torch.nn.Softmin.cpu"] = "torch.nn.Softmin"
allowlist["torch.nn.Softmin.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Softmin.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Softmin.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Softplus"] = "torch.nn.Softplus"
allowlist["torch.nn.Softplus.__call__"] = "torch.Tensor"
allowlist["torch.nn.Softplus.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Softplus.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Softplus.train"] = "torch.nn.Softplus"
allowlist["torch.nn.Softplus.cuda"] = "torch.nn.Softplus"
allowlist["torch.nn.Softplus.cpu"] = "torch.nn.Softplus"
allowlist["torch.nn.Softplus.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Softplus.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Softplus.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Softshrink"] = "torch.nn.Softshrink"
allowlist["torch.nn.Softshrink.__call__"] = "torch.Tensor"
allowlist["torch.nn.Softshrink.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Softshrink.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Softshrink.train"] = "torch.nn.Softshrink"
allowlist["torch.nn.Softshrink.cuda"] = "torch.nn.Softshrink"
allowlist["torch.nn.Softshrink.cpu"] = "torch.nn.Softshrink"
allowlist["torch.nn.Softshrink.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Softshrink.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Softshrink.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Softsign"] = "torch.nn.Softsign"
allowlist["torch.nn.Softsign.__call__"] = "torch.Tensor"
allowlist["torch.nn.Softsign.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Softsign.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Softsign.train"] = "torch.nn.Softsign"
allowlist["torch.nn.Softsign.cuda"] = "torch.nn.Softsign"
allowlist["torch.nn.Softsign.cpu"] = "torch.nn.Softsign"
allowlist["torch.nn.Softsign.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Softsign.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Softsign.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.SyncBatchNorm"] = "torch.nn.SyncBatchNorm"
allowlist["torch.nn.SyncBatchNorm.__call__"] = "torch.Tensor"
allowlist["torch.nn.SyncBatchNorm.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.SyncBatchNorm.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.SyncBatchNorm.train"] = "torch.nn.SyncBatchNorm"
allowlist["torch.nn.SyncBatchNorm.cuda"] = "torch.nn.SyncBatchNorm"
allowlist["torch.nn.SyncBatchNorm.cpu"] = "torch.nn.SyncBatchNorm"
allowlist[
    "torch.nn.SyncBatchNorm.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.SyncBatchNorm.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.SyncBatchNorm.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Tanh"] = "torch.nn.Tanh"
allowlist["torch.nn.Tanh.__call__"] = "torch.Tensor"
allowlist["torch.nn.Tanh.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Tanh.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Tanh.train"] = "torch.nn.Tanh"
allowlist["torch.nn.Tanh.cuda"] = "torch.nn.Tanh"
allowlist["torch.nn.Tanh.cpu"] = "torch.nn.Tanh"
allowlist["torch.nn.Tanh.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Tanh.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Tanh.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Tanhshrink"] = "torch.nn.Tanhshrink"
allowlist["torch.nn.Tanhshrink.__call__"] = "torch.Tensor"
allowlist["torch.nn.Tanhshrink.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Tanhshrink.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Tanhshrink.train"] = "torch.nn.Tanhshrink"
allowlist["torch.nn.Tanhshrink.cuda"] = "torch.nn.Tanhshrink"
allowlist["torch.nn.Tanhshrink.cpu"] = "torch.nn.Tanhshrink"
allowlist["torch.nn.Tanhshrink.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Tanhshrink.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Tanhshrink.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Threshold"] = "torch.nn.Threshold"
allowlist["torch.nn.Threshold.__call__"] = "torch.Tensor"
allowlist["torch.nn.Threshold.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Threshold.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Threshold.train"] = "torch.nn.Threshold"
allowlist["torch.nn.Threshold.cuda"] = "torch.nn.Threshold"
allowlist["torch.nn.Threshold.cpu"] = "torch.nn.Threshold"
allowlist["torch.nn.Threshold.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Threshold.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Threshold.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Transformer"] = "torch.nn.Transformer"
allowlist["torch.nn.Transformer.__call__"] = "torch.Tensor"
allowlist["torch.nn.Transformer.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Transformer.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Transformer.train"] = "torch.nn.Transformer"
allowlist["torch.nn.Transformer.cuda"] = "torch.nn.Transformer"
allowlist["torch.nn.Transformer.cpu"] = "torch.nn.Transformer"
allowlist["torch.nn.Transformer.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Transformer.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Transformer.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.TransformerDecoder"] = "torch.nn.TransformerDecoder"
allowlist["torch.nn.TransformerDecoder.__call__"] = "torch.Tensor"
allowlist["torch.nn.TransformerDecoder.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.TransformerDecoder.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.TransformerDecoder.train"] = "torch.nn.TransformerDecoder"
allowlist["torch.nn.TransformerDecoder.cuda"] = "torch.nn.TransformerDecoder"
allowlist["torch.nn.TransformerDecoder.cpu"] = "torch.nn.TransformerDecoder"
allowlist[
    "torch.nn.TransformerDecoder.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.TransformerDecoder.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.TransformerDecoder.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.TransformerDecoderLayer"] = "torch.nn.TransformerDecoderLayer"
allowlist["torch.nn.TransformerDecoderLayer.__call__"] = "torch.Tensor"
allowlist["torch.nn.TransformerDecoderLayer.parameters"] = "syft.lib.python.List"
allowlist[
    "torch.nn.TransformerDecoderLayer.register_parameter"
] = "syft.lib.python._SyNone"
allowlist["torch.nn.TransformerDecoderLayer.train"] = "torch.nn.TransformerDecoderLayer"
allowlist["torch.nn.TransformerDecoderLayer.cuda"] = "torch.nn.TransformerDecoderLayer"
allowlist["torch.nn.TransformerDecoderLayer.cpu"] = "torch.nn.TransformerDecoderLayer"
allowlist[
    "torch.nn.TransformerDecoderLayer.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist[
    "torch.nn.TransformerDecoderLayer.load_state_dict"
] = "syft.lib.python._SyNone"
allowlist["torch.nn.TransformerDecoderLayer.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.TransformerEncoder"] = "torch.nn.TransformerEncoder"
allowlist["torch.nn.TransformerEncoder.__call__"] = "torch.Tensor"
allowlist["torch.nn.TransformerEncoder.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.TransformerEncoder.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.TransformerEncoder.train"] = "torch.nn.TransformerEncoder"
allowlist["torch.nn.TransformerEncoder.cuda"] = "torch.nn.TransformerEncoder"
allowlist["torch.nn.TransformerEncoder.cpu"] = "torch.nn.TransformerEncoder"
allowlist[
    "torch.nn.TransformerEncoder.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.TransformerEncoder.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.TransformerEncoder.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.TransformerEncoderLayer"] = "torch.nn.TransformerEncoderLayer"
allowlist["torch.nn.TransformerEncoderLayer.__call__"] = "torch.Tensor"
allowlist["torch.nn.TransformerEncoderLayer.parameters"] = "syft.lib.python.List"
allowlist[
    "torch.nn.TransformerEncoderLayer.register_parameter"
] = "syft.lib.python._SyNone"
allowlist["torch.nn.TransformerEncoderLayer.train"] = "torch.nn.TransformerEncoderLayer"
allowlist["torch.nn.TransformerEncoderLayer.cuda"] = "torch.nn.TransformerEncoderLayer"
allowlist["torch.nn.TransformerEncoderLayer.cpu"] = "torch.nn.TransformerEncoderLayer"
allowlist[
    "torch.nn.TransformerEncoderLayer.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist[
    "torch.nn.TransformerEncoderLayer.load_state_dict"
] = "syft.lib.python._SyNone"
allowlist["torch.nn.TransformerEncoderLayer.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Unfold"] = "torch.nn.Unfold"
allowlist["torch.nn.Unfold.__call__"] = "torch.Tensor"
allowlist["torch.nn.Unfold.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Unfold.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Unfold.train"] = "torch.nn.Unfold"
allowlist["torch.nn.Unfold.cuda"] = "torch.nn.Unfold"
allowlist["torch.nn.Unfold.cpu"] = "torch.nn.Unfold"
allowlist["torch.nn.Unfold.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Unfold.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Unfold.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.Upsample"] = "torch.nn.Upsample"
allowlist["torch.nn.Upsample.__call__"] = "torch.Tensor"
allowlist["torch.nn.Upsample.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Upsample.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Upsample.train"] = "torch.nn.Upsample"
allowlist["torch.nn.Upsample.cuda"] = "torch.nn.Upsample"
allowlist["torch.nn.Upsample.cpu"] = "torch.nn.Upsample"
allowlist["torch.nn.Upsample.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.Upsample.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Upsample.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.UpsamplingBilinear2d"] = "torch.nn.UpsamplingBilinear2d"
allowlist["torch.nn.UpsamplingBilinear2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.UpsamplingBilinear2d.parameters"] = "syft.lib.python.List"
allowlist[
    "torch.nn.UpsamplingBilinear2d.register_parameter"
] = "syft.lib.python._SyNone"
allowlist["torch.nn.UpsamplingBilinear2d.train"] = "torch.nn.UpsamplingBilinear2d"
allowlist["torch.nn.UpsamplingBilinear2d.cuda"] = "torch.nn.UpsamplingBilinear2d"
allowlist["torch.nn.UpsamplingBilinear2d.cpu"] = "torch.nn.UpsamplingBilinear2d"
allowlist[
    "torch.nn.UpsamplingBilinear2d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.UpsamplingBilinear2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.UpsamplingBilinear2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.UpsamplingNearest2d"] = "torch.nn.UpsamplingNearest2d"
allowlist["torch.nn.UpsamplingNearest2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.UpsamplingNearest2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.UpsamplingNearest2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.UpsamplingNearest2d.train"] = "torch.nn.UpsamplingNearest2d"
allowlist["torch.nn.UpsamplingNearest2d.cuda"] = "torch.nn.UpsamplingNearest2d"
allowlist["torch.nn.UpsamplingNearest2d.cpu"] = "torch.nn.UpsamplingNearest2d"
allowlist[
    "torch.nn.UpsamplingNearest2d.state_dict"
] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.UpsamplingNearest2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.UpsamplingNearest2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.nn.ZeroPad2d"] = "torch.nn.ZeroPad2d"
allowlist["torch.nn.ZeroPad2d.__call__"] = "torch.Tensor"
allowlist["torch.nn.ZeroPad2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.ZeroPad2d.register_parameter"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ZeroPad2d.train"] = "torch.nn.ZeroPad2d"
allowlist["torch.nn.ZeroPad2d.cuda"] = "torch.nn.ZeroPad2d"
allowlist["torch.nn.ZeroPad2d.cpu"] = "torch.nn.ZeroPad2d"
allowlist["torch.nn.ZeroPad2d.state_dict"] = "syft.lib.python.collections.OrderedDict"
allowlist["torch.nn.ZeroPad2d.load_state_dict"] = "syft.lib.python._SyNone"
allowlist["torch.nn.ZeroPad2d.extra_repr"] = "syft.lib.python.String"

allowlist["torch.distributions.Categorical"] = "torch.distributions.Categorical"
allowlist["torch.distributions.Categorical.sample"] = "torch.Tensor"
allowlist["torch.distributions.Categorical.log_prob"] = "torch.Tensor"

allowlist["torch.Tensor.xpu"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.tile"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.fmax"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.ldexp_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.sinc"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.kron"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.nan_to_num"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.msort"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.row_stack"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.new_empty_strided"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.ravel"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.swapdims_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.moveaxis"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.swapaxes"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.tensor_split"] = {
    "return_type": "syft.lib.python.List",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.tensor_split"] = {
    "return_type": "syft.lib.python.List",
    "min_version": "1.8.0",
}
allowlist["torch.tile"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.float_power_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.broadcast_to"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.fmin"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.ldexp"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
# allowlist["torch.broadcast_shapes"] = {
#     "return_type": "torch.Size",
#     "min_version": "1.8.0",
# }
allowlist["torch.Tensor.swapdims"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.igamma"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.nan_to_num_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.copysign"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.swapaxes_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.cumprod_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.ldexp"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.igamma_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.float_power"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.igammac_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.xlogy"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.copysign"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.nanmedian"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.igammac"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.cumsum_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.diff"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.igamma"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.sinc"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.igammac"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.kron"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.column_stack"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.msort"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.pixel_unshuffle"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.fmin"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.xlogy"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.moveaxis"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.swapaxes"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.nan_to_num"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.inner"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.fmax"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.float_power"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.nanmedian"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.sinc_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.copysign_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.ravel"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.broadcast_to"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.diff"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.xlogy_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.swapdims"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}
allowlist["torch.Tensor.inner"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.8.0",
}

dynamic_allowlist["torch.nn.Linear.weight"] = "torch.nn.Parameter"
dynamic_allowlist["torch.nn.Linear.bias"] = "torch.nn.Parameter"
