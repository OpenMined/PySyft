# stdlib
from typing import Dict
from typing import Union

allowlist: Dict[str, Union[str, Dict[str, str]]] = {}  # (path: str, return_type:type)

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
# allowlist["torch.Tensor._indices"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor._is_view"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor._make_subclass"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor._nnz"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor._update_names"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor._values"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.dtype"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST YET - talk to TRASK
# allowlist["torch.Tensor.has_names"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.hex"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.record_stream"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST#
# allowlist["torch.Tensor.register_hook"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.share_memory_"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.storage"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.storage_offset"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.storage_type"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.where"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST

# SECTION - Tensor methods which have serde issues
# allowlist["torch.Tensor.json"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.to_binary"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.to_dense"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.to_hex"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.to_json"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.to_mkldnn"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.to_proto"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.to_sparse"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST

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
# SECTION - Tensor methods which are tested
# --------------------------------------------------------------------------------------

# SECTION - The capital Tensor constructor
allowlist["torch.Tensor"] = "torch.Tensor"

# SECTION - Tensor methods with special version requirements
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

allowlist["torch.Tensor.__ifloordiv__"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.1",
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
allowlist["torch.Tensor.is_meta"] = {
    "return_type": "syft.lib.python.Bool",
    "min_version": "1.6.0",
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

# SECTION - Tensor methods
allowlist["torch.Tensor.__abs__"] = "torch.Tensor"
allowlist["torch.Tensor.__add__"] = "torch.Tensor"
allowlist["torch.Tensor.__and__"] = "torch.Tensor"
allowlist["torch.Tensor.__eq__"] = "torch.Tensor"
allowlist["torch.Tensor.__float__"] = "syft.lib.python.Float"
allowlist["torch.Tensor.__ge__"] = "torch.Tensor"
allowlist["torch.Tensor.__gt__"] = "torch.Tensor"
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
allowlist["torch.Tensor.__sub__"] = "torch.Tensor"
allowlist["torch.Tensor.__truediv__"] = "torch.Tensor"
allowlist["torch.Tensor.__xor__"] = "torch.Tensor"
allowlist["torch.Tensor.abs_"] = "torch.Tensor"
allowlist["torch.Tensor.abs"] = "torch.Tensor"
allowlist["torch.Tensor.acos_"] = "torch.Tensor"
allowlist["torch.Tensor.acos"] = "torch.Tensor"
allowlist["torch.Tensor.add_"] = "torch.Tensor"
allowlist["torch.Tensor.add"] = "torch.Tensor"
allowlist["torch.Tensor.argmax"] = "torch.Tensor"
allowlist["torch.Tensor.asin_"] = "torch.Tensor"
allowlist["torch.Tensor.asin"] = "torch.Tensor"
allowlist["torch.Tensor.atan_"] = "torch.Tensor"
allowlist["torch.Tensor.atan"] = "torch.Tensor"
allowlist["torch.Tensor.atan2_"] = "torch.Tensor"
allowlist["torch.Tensor.atan2"] = "torch.Tensor"
allowlist["torch.Tensor.bitwise_not_"] = "torch.Tensor"
allowlist["torch.Tensor.bitwise_not"] = "torch.Tensor"
allowlist["torch.Tensor.bitwise_xor_"] = "torch.Tensor"
allowlist["torch.Tensor.bitwise_xor"] = "torch.Tensor"
allowlist["torch.Tensor.ceil_"] = "torch.Tensor"
allowlist["torch.Tensor.ceil"] = "torch.Tensor"
allowlist["torch.Tensor.char"] = "torch.Tensor"
allowlist["torch.Tensor.clone"] = "torch.Tensor"
allowlist["torch.Tensor.copy_"] = "torch.Tensor"
allowlist["torch.Tensor.cos_"] = "torch.Tensor"
allowlist["torch.Tensor.cos"] = "torch.Tensor"
allowlist["torch.Tensor.cosh_"] = "torch.Tensor"
allowlist["torch.Tensor.cosh"] = "torch.Tensor"
allowlist["torch.Tensor.cpu"] = "torch.Tensor"
allowlist["torch.Tensor.data"] = "torch.Tensor"
allowlist["torch.Tensor.diag"] = "torch.Tensor"
allowlist["torch.Tensor.diagonal"] = "torch.Tensor"
allowlist["torch.Tensor.dot"] = "torch.Tensor"
allowlist["torch.Tensor.double"] = "torch.Tensor"
allowlist["torch.Tensor.eq_"] = "torch.Tensor"
allowlist["torch.Tensor.eq"] = "torch.Tensor"
allowlist["torch.Tensor.erf_"] = "torch.Tensor"
allowlist["torch.Tensor.erf"] = "torch.Tensor"
allowlist["torch.Tensor.erfc_"] = "torch.Tensor"
allowlist["torch.Tensor.erfc"] = "torch.Tensor"
allowlist["torch.Tensor.erfinv_"] = "torch.Tensor"
allowlist["torch.Tensor.erfinv"] = "torch.Tensor"
allowlist["torch.Tensor.exp_"] = "torch.Tensor"
allowlist["torch.Tensor.exp"] = "torch.Tensor"
allowlist["torch.Tensor.expm1_"] = "torch.Tensor"
allowlist["torch.Tensor.expm1"] = "torch.Tensor"
allowlist["torch.Tensor.flatten"] = "torch.Tensor"
allowlist["torch.Tensor.float"] = "torch.Tensor"
allowlist["torch.Tensor.floor_"] = "torch.Tensor"
allowlist["torch.Tensor.floor"] = "torch.Tensor"
allowlist["torch.Tensor.frac_"] = "torch.Tensor"
allowlist["torch.Tensor.frac"] = "torch.Tensor"
allowlist["torch.Tensor.ge_"] = "torch.Tensor"
allowlist["torch.Tensor.ge"] = "torch.Tensor"
allowlist["torch.Tensor.ger"] = "torch.Tensor"
allowlist["torch.Tensor.gt_"] = "torch.Tensor"
allowlist["torch.Tensor.gt"] = "torch.Tensor"
allowlist["torch.Tensor.half"] = "torch.Tensor"
allowlist["torch.Tensor.int"] = "torch.Tensor"
allowlist["torch.Tensor.is_cuda"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_floating_point"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_leaf"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_mkldnn"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_quantized"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_sparse"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.le_"] = "torch.Tensor"
allowlist["torch.Tensor.le"] = "torch.Tensor"
allowlist["torch.Tensor.lgamma_"] = "torch.Tensor"
allowlist["torch.Tensor.lgamma"] = "torch.Tensor"
allowlist["torch.Tensor.log_"] = "torch.Tensor"
allowlist["torch.Tensor.log"] = "torch.Tensor"
allowlist["torch.Tensor.log10_"] = "torch.Tensor"
allowlist["torch.Tensor.log10"] = "torch.Tensor"
allowlist["torch.Tensor.log1p_"] = "torch.Tensor"
allowlist["torch.Tensor.log1p"] = "torch.Tensor"
allowlist["torch.Tensor.log2_"] = "torch.Tensor"
allowlist["torch.Tensor.log2"] = "torch.Tensor"
allowlist["torch.Tensor.logical_not_"] = "torch.Tensor"
allowlist["torch.Tensor.logical_not"] = "torch.Tensor"
allowlist["torch.Tensor.logical_xor_"] = "torch.Tensor"
allowlist["torch.Tensor.logical_xor"] = "torch.Tensor"
allowlist["torch.Tensor.long"] = "torch.Tensor"
allowlist["torch.Tensor.lt_"] = "torch.Tensor"
allowlist["torch.Tensor.lt"] = "torch.Tensor"
allowlist["torch.Tensor.matmul"] = "torch.Tensor"
allowlist["torch.Tensor.mm"] = "torch.Tensor"
allowlist["torch.Tensor.mul_"] = "torch.Tensor"
allowlist["torch.Tensor.mul"] = "torch.Tensor"
allowlist["torch.Tensor.ndim"] = "syft.lib.python.Int"
allowlist["torch.Tensor.ne_"] = "torch.Tensor"
allowlist["torch.Tensor.ne"] = "torch.Tensor"
allowlist["torch.Tensor.t"] = "torch.Tensor"
allowlist["torch.Tensor.T"] = "torch.Tensor"
allowlist["torch.Tensor.neg_"] = "torch.Tensor"
allowlist["torch.Tensor.neg"] = "torch.Tensor"
allowlist["torch.Tensor.new_tensor"] = "torch.Tensor"
allowlist["torch.Tensor.nonzero"] = "torch.Tensor"
allowlist["torch.Tensor.norm"] = "torch.Tensor"
allowlist["torch.Tensor.orgqr"] = "torch.Tensor"
allowlist["torch.Tensor.output_nr"] = "syft.lib.python.Int"
allowlist["torch.Tensor.pinverse"] = "torch.Tensor"
allowlist["torch.Tensor.pow_"] = "torch.Tensor"
allowlist["torch.Tensor.pow"] = "torch.Tensor"
allowlist["torch.Tensor.prod"] = "torch.Tensor"
allowlist["torch.Tensor.reciprocal_"] = "torch.Tensor"
allowlist["torch.Tensor.reciprocal"] = "torch.Tensor"
allowlist["torch.Tensor.relu_"] = "torch.Tensor"
allowlist["torch.Tensor.relu"] = "torch.Tensor"
allowlist["torch.Tensor.requires_grad_"] = "torch.Tensor"
allowlist["torch.Tensor.requires_grad"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.reshape_as"] = "torch.Tensor"
allowlist["torch.Tensor.resize_as_"] = "torch.Tensor"
allowlist["torch.Tensor.rot90"] = "torch.Tensor"
allowlist["torch.Tensor.round_"] = "torch.Tensor"
allowlist["torch.Tensor.round"] = "torch.Tensor"
allowlist["torch.Tensor.rsqrt_"] = "torch.Tensor"
allowlist["torch.Tensor.rsqrt"] = "torch.Tensor"
allowlist["torch.Tensor.short"] = "torch.Tensor"
allowlist["torch.Tensor.sigmoid_"] = "torch.Tensor"
allowlist["torch.Tensor.sigmoid"] = "torch.Tensor"
allowlist["torch.Tensor.sign_"] = "torch.Tensor"
allowlist["torch.Tensor.sign"] = "torch.Tensor"
allowlist["torch.Tensor.sin_"] = "torch.Tensor"
allowlist["torch.Tensor.sin"] = "torch.Tensor"
allowlist["torch.Tensor.sinh_"] = "torch.Tensor"
allowlist["torch.Tensor.sinh"] = "torch.Tensor"
allowlist["torch.Tensor.sqrt_"] = "torch.Tensor"
allowlist["torch.Tensor.sqrt"] = "torch.Tensor"
allowlist["torch.Tensor.squeeze_"] = "torch.Tensor"
allowlist["torch.Tensor.squeeze"] = "torch.Tensor"
allowlist["torch.Tensor.std"] = "torch.Tensor"
allowlist["torch.Tensor.sub_"] = "torch.Tensor"
allowlist["torch.Tensor.sub"] = "torch.Tensor"
allowlist["torch.Tensor.sum"] = "torch.Tensor"
allowlist["torch.Tensor.t_"] = "torch.Tensor"
allowlist["torch.Tensor.t"] = "torch.Tensor"
allowlist["torch.Tensor.tan_"] = "torch.Tensor"
allowlist["torch.Tensor.tan"] = "torch.Tensor"
allowlist["torch.Tensor.tanh_"] = "torch.Tensor"
allowlist["torch.Tensor.tanh"] = "torch.Tensor"
allowlist["torch.Tensor.to"] = "torch.Tensor"
allowlist["torch.Tensor.trace"] = "torch.Tensor"
allowlist["torch.Tensor.tril_"] = "torch.Tensor"
allowlist["torch.Tensor.tril"] = "torch.Tensor"
allowlist["torch.Tensor.triu_"] = "torch.Tensor"
allowlist["torch.Tensor.triu"] = "torch.Tensor"
allowlist["torch.Tensor.trunc_"] = "torch.Tensor"
allowlist["torch.Tensor.trunc"] = "torch.Tensor"
allowlist["torch.Tensor.unique"] = "torch.Tensor"
allowlist["torch.Tensor.unsqueeze_"] = "torch.Tensor"
allowlist["torch.Tensor.unsqueeze"] = "torch.Tensor"
allowlist["torch.Tensor.var"] = "torch.Tensor"
allowlist["torch.Tensor.view_as"] = "torch.Tensor"
allowlist["torch.Tensor.view"] = "torch.Tensor"
allowlist["torch.Tensor.zero_"] = "torch.Tensor"

# --------------------------------------------------------------------------------------
# SECTION - Tensor methods which are untested
# --------------------------------------------------------------------------------------

# allowlist["torch.Tensor.real"] = "torch.Tensor"  # requires complex or 1.6.0?
# allowlist["torch.Tensor.imag"] = "torch.Tensor"  # requires complex or 1.6.0?
# allowlist["torch.Tensor._version"] = "unknown"
# allowlist["torch.Tensor.addbmm_"] = "unknown"
# allowlist["torch.Tensor.addbmm"] = "unknown"
# allowlist["torch.Tensor.addcdiv_"] = "unknown"
# allowlist["torch.Tensor.addcdiv"] = "unknown"
# allowlist["torch.Tensor.addcmul_"] = "unknown"
# allowlist["torch.Tensor.addcmul"] = "unknown"
# allowlist["torch.Tensor.addmm_"] = "unknown"
# allowlist["torch.Tensor.addmm"] = "unknown"
# allowlist["torch.Tensor.addmv_"] = "unknown"
# allowlist["torch.Tensor.addmv"] = "unknown"
# allowlist["torch.Tensor.addr_"] = "unknown"
# allowlist["torch.Tensor.addr"] = "unknown"
# allowlist["torch.Tensor.align_as"] = "unknown"
# allowlist["torch.Tensor.align_to"] = "unknown"
# allowlist["torch.Tensor.all"] = "unknown"
# allowlist["torch.Tensor.allclose"] = "unknown"
# allowlist["torch.Tensor.angle"] = "unknown"
# allowlist["torch.Tensor.any"] = "unknown"
# allowlist["torch.Tensor.apply_"] = "unknown"
# allowlist["torch.Tensor.argmin"] = "unknown"
# allowlist["torch.Tensor.argsort"] = "unknown"
# allowlist["torch.Tensor.as_strided_"] = "unknown"
# allowlist["torch.Tensor.as_strided"] = "unknown"
# allowlist["torch.Tensor.baddbmm_"] = "unknown"
# allowlist["torch.Tensor.baddbmm"] = "unknown"
# allowlist["torch.Tensor.bernoulli_"] = "unknown"
# allowlist["torch.Tensor.bernoulli"] = "unknown"
# allowlist["torch.Tensor.bfloat16"] = "unknown"
# allowlist["torch.Tensor.binary"] = "unknown"
# allowlist["torch.Tensor.bincount"] = "unknown"
# allowlist["torch.Tensor.bmm"] = "unknown"
# allowlist["torch.Tensor.bool"] = "unknown"
# allowlist["torch.Tensor.byte"] = "unknown"
# allowlist["torch.Tensor.cauchy_"] = "unknown"
# allowlist["torch.Tensor.cholesky_inverse"] = "unknown"
# allowlist["torch.Tensor.cholesky"] = "unknown"
# allowlist["torch.Tensor.chunk"] = "unknown"
# allowlist["torch.Tensor.coalesce"] = "unknown"
# allowlist["torch.Tensor.conj"] = "unknown"
# allowlist["torch.Tensor.contiguous"] = "unknown"
# allowlist["torch.Tensor.cross"] = "unknown"
# allowlist["torch.Tensor.cuda"] = "unknown"
# allowlist["torch.Tensor.cummax"] = "unknown"
# allowlist["torch.Tensor.cummin"] = "unknown"
# allowlist["torch.Tensor.cumprod"] = "unknown"
# allowlist["torch.Tensor.cumsum"] = "unknown"
# allowlist["torch.Tensor.data_ptr"] = "unknown"
# allowlist["torch.Tensor.dense_dim"] = "unknown"
# allowlist["torch.Tensor.dequantize"] = "unknown"
# allowlist["torch.Tensor.describe"] = "unknown"
# allowlist["torch.Tensor.det"] = "unknown"
# allowlist["torch.Tensor.detach_"] = "unknown"
# allowlist["torch.Tensor.detach"] = "unknown"
# allowlist["torch.Tensor.device"] = "unknown"
# allowlist["torch.Tensor.diag_embed"] = "unknown"
# allowlist["torch.Tensor.diagflat"] = "unknown"
# allowlist["torch.Tensor.digamma_"] = "unknown"
# allowlist["torch.Tensor.digamma"] = "unknown"
# allowlist["torch.Tensor.dim"] = "unknown"
# allowlist["torch.Tensor.dist"] = "unknown"
# allowlist["torch.Tensor.eig"] = "unknown"
# allowlist["torch.Tensor.element_size"] = "unknown"
# allowlist["torch.Tensor.equal"] = "unknown"
# allowlist["torch.Tensor.expand_as"] = "unknown"
# allowlist["torch.Tensor.expand"] = "unknown"
# allowlist["torch.Tensor.exponential_"] = "unknown"
# allowlist["torch.Tensor.fft"] = "unknown"
# allowlist["torch.Tensor.fill_"] = "unknown"
# allowlist["torch.Tensor.fill_diagonal_"] = "unknown"
# allowlist["torch.Tensor.flip"] = "unknown"
# allowlist["torch.Tensor.fmod_"] = "unknown"
# allowlist["torch.Tensor.fmod"] = "unknown"
# allowlist["torch.Tensor.gather"] = "unknown"
# allowlist["torch.Tensor.geometric_"] = "unknown"
# allowlist["torch.Tensor.geqrf"] = "unknown"
# allowlist["torch.Tensor.get_device"] = "unknown"
# allowlist["torch.Tensor.grad_fn"] = "unknown"
# allowlist["torch.Tensor.grad"] = "unknown"
# allowlist["torch.Tensor.hardshrink"] = "unknown"
# allowlist["torch.Tensor.histc"] = "unknown"
# allowlist["torch.Tensor.id"] = "unknown"
# allowlist["torch.Tensor.ifft"] = "unknown"
# allowlist["torch.Tensor.index_add_"] = "unknown"
# allowlist["torch.Tensor.index_add"] = "unknown"
# allowlist["torch.Tensor.index_copy_"] = "unknown"
# allowlist["torch.Tensor.index_copy"] = "unknown"
# allowlist["torch.Tensor.index_fill_"] = "unknown"
# allowlist["torch.Tensor.index_fill"] = "unknown"
# allowlist["torch.Tensor.index_put_"] = "unknown"
# allowlist["torch.Tensor.index_put"] = "unknown"
# allowlist["torch.Tensor.indices"] = "unknown"
# allowlist["torch.Tensor.int_repr"] = "unknown"
# allowlist["torch.Tensor.inverse"] = "unknown"
# allowlist["torch.Tensor.irfft"] = "unknown"
# allowlist["torch.Tensor.is_coalesced"] = "unknown"
# allowlist["torch.Tensor.is_complex"] = "unknown"
# allowlist["torch.Tensor.is_contiguous"] = "unknown"
# allowlist["torch.Tensor.is_distributed"] = "unknown"
# allowlist["torch.Tensor.is_nonzero"] = "unknown"
# allowlist["torch.Tensor.is_pinned"] = "unknown"
# allowlist["torch.Tensor.is_same_size"] = "unknown"
# allowlist["torch.Tensor.is_set_to"] = "unknown"
# allowlist["torch.Tensor.is_shared"] = "unknown"
# allowlist["torch.Tensor.is_signed"] = "unknown"
# allowlist["torch.Tensor.isclose"] = "unknown"
# allowlist["torch.Tensor.kthvalue"] = "unknown"
# allowlist["torch.Tensor.lerp_"] = "unknown"
# allowlist["torch.Tensor.lerp"] = "unknown"
# allowlist["torch.Tensor.log_normal_"] = "unknown"
# allowlist["torch.Tensor.log_softmax"] = "unknown"
# allowlist["torch.Tensor.logdet"] = "unknown"
# allowlist["torch.Tensor.logsumexp"] = "unknown"
# allowlist["torch.Tensor.lstsq"] = "unknown"
# allowlist["torch.Tensor.lu_solve"] = "unknown"
# allowlist["torch.Tensor.lu"] = "unknown"
# allowlist["torch.Tensor.map_"] = "unknown"
# allowlist["torch.Tensor.map2_"] = "unknown"
# allowlist["torch.Tensor.masked_fill_"] = "unknown"
# allowlist["torch.Tensor.masked_fill"] = "unknown"
# allowlist["torch.Tensor.masked_scatter_"] = "unknown"
# allowlist["torch.Tensor.masked_scatter"] = "unknown"
# allowlist["torch.Tensor.masked_select"] = "unknown"
# allowlist["torch.Tensor.matrix_power"] = "unknown"
# allowlist["torch.Tensor.median"] = "unknown"  # requires torch.return_types.median
# allowlist["torch.Tensor.mode"] = "unknown"
# allowlist["torch.Tensor.multinomial"] = "unknown"
# allowlist["torch.Tensor.mv"] = "unknown"
# allowlist["torch.Tensor.mvlgamma_"] = "unknown"
# allowlist["torch.Tensor.mvlgamma"] = "unknown"
# allowlist["torch.Tensor.narrow_copy"] = "unknown"
# allowlist["torch.Tensor.narrow"] = "unknown"
# allowlist["torch.Tensor.ndimension"] = "unknown"
# allowlist["torch.Tensor.nelement"] = "unknown"
# allowlist["torch.Tensor.new_empty"] = "unknown"
# allowlist["torch.Tensor.new_full"] = "unknown"
# allowlist["torch.Tensor.new_ones"] = "unknown"
# allowlist["torch.Tensor.new_zeros"] = "unknown"
# allowlist["torch.Tensor.normal_"] = "unknown"
# allowlist["torch.Tensor.numel"] = "unknown"
# allowlist["torch.Tensor.numpy"] = "unknown"
# allowlist["torch.Tensor.ormqr"] = "unknown"  # requires two tensors as arguments
# allowlist["torch.Tensor.permute"] = "unknown"
# allowlist["torch.Tensor.pin_memory"] = "unknown"
# allowlist["torch.Tensor.polygamma_"] = "unknown"
# allowlist["torch.Tensor.polygamma"] = "unknown"
# allowlist["torch.Tensor.prelu"] = "unknown"
# allowlist["torch.Tensor.proto"] = "unknown"
# allowlist["torch.Tensor.put_"] = "unknown"
# allowlist["torch.Tensor.q_per_channel_axis"] = "unknown"
# allowlist["torch.Tensor.q_per_channel_scales"] = "unknown"
# allowlist["torch.Tensor.q_per_channel_zero_points"] = "unknown"
# allowlist["torch.Tensor.q_scale"] = "unknown"
# allowlist["torch.Tensor.q_zero_point"] = "unknown"
# allowlist["torch.Tensor.qr"] = "unknown"
# allowlist["torch.Tensor.qscheme"] = "unknown"
# allowlist["torch.Tensor.random_"] = "unknown"
# allowlist["torch.Tensor.refine_names"] = "unknown"
# allowlist["torch.Tensor.reinforce"] = "unknown"
# allowlist["torch.Tensor.rename_"] = "unknown"
# allowlist["torch.Tensor.rename"] = "unknown"
# allowlist["torch.Tensor.renorm_"] = "unknown"
# allowlist["torch.Tensor.renorm"] = "unknown"
# allowlist["torch.Tensor.repeat_interleave"] = "unknown"
# allowlist["torch.Tensor.repeat"] = "unknown"
# allowlist["torch.Tensor.reshape"] = "unknown"
# allowlist["torch.Tensor.resize_"] = "unknown"
# allowlist["torch.Tensor.resize"] = "unknown"
# allowlist["torch.Tensor.retain_grad"] = "unknown"
# allowlist["torch.Tensor.rfft"] = "unknown"
# allowlist["torch.Tensor.roll"] = "unknown"
# allowlist["torch.Tensor.scatter_"] = "unknown"
# allowlist["torch.Tensor.scatter_add_"] = "unknown"
# allowlist["torch.Tensor.scatter_add"] = "unknown"
# allowlist["torch.Tensor.scatter"] = "unknown"
# allowlist["torch.Tensor.select"] = "unknown"
# allowlist["torch.Tensor.send"] = "unknown"
# allowlist["torch.Tensor.serializable_wrapper_type"] = "unknown"
# allowlist["torch.Tensor.serialize"] = "unknown"
# allowlist["torch.Tensor.set_"] = "unknown"
# allowlist["torch.Tensor.size"] = "unknown"
# allowlist["torch.Tensor.slogdet"] = "unknown"
# allowlist["torch.Tensor.smm"] = "unknown"
# allowlist["torch.Tensor.softmax"] = "unknown"
# allowlist["torch.Tensor.solve"] = "unknown"
# allowlist["torch.Tensor.sort"] = "unknown"
# allowlist["torch.Tensor.sparse_dim"] = "unknown"
# allowlist["torch.Tensor.sparse_mask"] = "unknown"
# allowlist["torch.Tensor.sparse_resize_"] = "unknown"
# allowlist["torch.Tensor.sparse_resize_and_clear_"] = "unknown"
# allowlist["torch.Tensor.split_with_sizes"] = "unknown"
# allowlist["torch.Tensor.split"] = "unknown"
# allowlist["torch.Tensor.sspaddmm"] = "unknown"
# allowlist["torch.Tensor.stft"] = "unknown"
# allowlist["torch.Tensor.stride"] = "unknown"
# allowlist["torch.Tensor.sum_to_size"] = "unknown"
# allowlist["torch.Tensor.svd"] = "unknown"
# allowlist["torch.Tensor.symeig"] = "unknown"
# allowlist["torch.Tensor.tag"] = "unknown"
# allowlist["torch.Tensor.tolist"] = "unknown"
# allowlist["torch.Tensor.topk"] = "unknown"
# allowlist["torch.Tensor.triangular_solve"] = "unknown"
# allowlist["torch.Tensor.type_as"] = "unknown"
# allowlist["torch.Tensor.type"] = "unknown"
# allowlist["torch.Tensor.unbind"] = "unknown"
# allowlist["torch.Tensor.unflatten"] = "unknown"
# allowlist["torch.Tensor.values"] = "unknown"

# --------------------------------------------------------------------------------------
# SECTION - Tensor methods with specific issues or require a special test combination
# --------------------------------------------------------------------------------------
# hashes are not matching
# allowlist["torch.Tensor.__hash__"] = "syft.lib.python.Int"
# allowlist["torch.Tensor.__getitem__"] = "torch.Tensor"
# allowlist["torch.Tensor.__setitem__"] = "torch.Tensor"
# allowlist["torch.Tensor.__iter__"] = "unknown"  # How to handle return iterator?
# allowlist["torch.Tensor.backward"] = "syft.lib.python.SyNone"
# allowlist["torch.Tensor.clamp_"] = "torch.Tensor" # clamps need min max etc
# allowlist["torch.Tensor.clamp_max_"] = "torch.Tensor"
# allowlist["torch.Tensor.clamp_max"] = "torch.Tensor"
# allowlist["torch.Tensor.clamp_min_"] = "torch.Tensor"
# allowlist["torch.Tensor.clamp_min"] = "torch.Tensor"
# allowlist["torch.Tensor.clamp"] = "torch.Tensor"
# allowlist["torch.Tensor.index_select"] = "torch.Tensor"
# allowlist["torch.Tensor.item"] = "syft.lib.python.Float" # Union[bool, int, float]
# allowlist["torch.Tensor.layout"] = "torch.layout" # requires torch layout
# allowlist["torch.layout"] = "torch.layout" # requires protobuf serialization
# allowlist["torch.Tensor.max"] = "torch.Tensor" # requires torch.return_types.max
# allowlist["torch.Tensor.mean"] = "torch.Tensor" # requires some test kwargs
# allowlist["torch.Tensor.min"] = "torch.Tensor" # requires some test kwargs
# allowlist["torch.Tensor.name"] = "Optional[str]" # requires named tensors and Optional
# allowlist["torch.Tensor.names"] = "Tuple[str]" # requires named tensors and Tuple
# allowlist["torch.Tensor.shape"] = "torch.Size" # requires torch.Size
# allowlist["torch.Size"] = "torch.Size" # requires protobuf serialization
# allowlist["torch.Tensor.take"] = "torch.Tensor" # requires long tensor input only
# allowlist["torch.Tensor.transpose_"] = "torch.Tensor" # requires two inputs
# allowlist["torch.Tensor.transpose"] = "torch.Tensor" # requires two inputs
# allowlist["torch.Tensor.unfold"] = "torch.Tensor" # requires three inputs
# allowlist["torch.Tensor.uniform_"] = "torch.Tensor"
# allowlist["torch.Tensor.unique_consecutive"] = "torch.Tensor" # requires Union / Tuple


# --------------------------------------------------------------------------------------
# SECTION - Tensor methods which we wont support
# --------------------------------------------------------------------------------------
# allowlist["torch.Tensor.resize_as"] = "unknown" deprecated
# allowlist["torch.Tensor.new"] = "unknown" # deprecated and returns random values

# SECTION - Other classes and modules
# allowlist["torch.zeros"] = "torch.Tensor"
# allowlist["torch.ones"] = "torch.Tensor"
# allowlist["torch.median"] = "torch.Tensor"  # requires torch.return_types.median

# SECTION - Parameter methods
# torch.nn.Parameter is a subclass of torch.Tensor
# However, we still need the constructor Class to be listed here. Everything else is
# automatically added in create_torch_ast function by doing:
# method = method.replace("torch.Tensor.", "torch.nn.Parameter.")
# allowlist["torch.nn.Parameter"] = "torch.nn.Parameter"

# MNIST
# Misc
# allowlist["torch.manual_seed"] = "torch.Generator"
# allowlist["torch.Generator"] = "torch.Generator"
# allowlist["torch.Generator.get_state"] = "torch.Tensor"
# allowlist["torch.Generator.set_state"] = "torch.Generator"

# Modules
# allowlist["torch.nn.Module"] = "torch.nn.Module"
# allowlist["torch.nn.Module.__call__"] = "torch.nn.Tensor"
# allowlist["torch.nn.Module.parameters"] = "syft.lib.python.List"
# allowlist["torch.nn.Module.train"] = "torch.nn.Module"
# allowlist["torch.nn.Module.cuda"] = "torch.nn.Module"
# allowlist["torch.nn.Module.cpu"] = "torch.nn.Module"

# allowlist["torch.nn.Conv2d"] = "torch.nn.Conv2d"
# allowlist["torch.nn.Conv2d.__call__"] = "torch.nn.Conv2d"
# allowlist["torch.nn.Conv2d.parameters"] = "syft.lib.python.List"
# allowlist["torch.nn.Conv2d.train"] = "torch.nn.Conv2d"
# allowlist["torch.nn.Conv2d.cuda"] = "torch.nn.Conv2d"
# allowlist["torch.nn.Conv2d.cpu"] = "torch.nn.Conv2d"

# allowlist["torch.nn.Dropout2d"] = "torch.nn.Dropout2d"
# allowlist["torch.nn.Dropout2d.__call__"] = "torch.nn.Dropout2d"
# allowlist["torch.nn.Dropout2d.parameters"] = "syft.lib.python.List"
# allowlist["torch.nn.Dropout2d.train"] = "torch.nn.Dropout2d"
# allowlist["torch.nn.Dropout2d.cuda"] = "torch.nn.Dropout2d"
# allowlist["torch.nn.Dropout2d.cpu"] = "torch.nn.Dropout2d"

# allowlist["torch.nn.Linear"] = "torch.nn.Linear"
# allowlist["torch.nn.Linear.__call__"] = "torch.nn.Linear"
# allowlist["torch.nn.Linear.parameters"] = "syft.lib.python.List"
# allowlist["torch.nn.Linear.train"] = "torch.nn.Linear"
# allowlist["torch.nn.Linear.cuda"] = "torch.nn.Linear"
# allowlist["torch.nn.Linear.cpu"] = "torch.nn.Linear"

# DataLoader
# allowlist["torch.utils.data.DataLoader"] = "torch.utils.data.DataLoader"
# allowlist["torch.utils.data.DataLoader.dataset"] = "torchvision.datasets.VisionDataset"
# allowlist[
#     "torch.utils.data.DataLoader.__iter__"
# ] = "torch.utils.data.dataloader._SingleProcessDataLoaderIter"
# allowlist["torch.utils.data.DataLoader.__len__"] = "syft.lib.python.Int"
# allowlist[
#     "torch.utils.data.dataloader._SingleProcessDataLoaderIter"
# ] = "torch.utils.data.dataloader._SingleProcessDataLoaderIter"
# allowlist[
#     "torch.utils.data.dataloader._SingleProcessDataLoaderIter.__len__"
# ] = "syft.lib.python.Int"
# allowlist[
#     "torch.utils.data.dataloader._SingleProcessDataLoaderIter.__iter__"
# ] = "torch.utils.data.dataloader._SingleProcessDataLoaderIter"

# working for part
# allowlist[
#     "torch.utils.data.dataloader._SingleProcessDataLoaderIter.__next__"
# ] = "torch.Tensor"
# allowlist[
#     "torch.utils.data.dataloader._SingleProcessDataLoaderIter.next"
# ] = "torch.Tensor"

# we are returning syft.lib.python.List so that we can __getitem__ on the return of
# enumerate(train_loader)
# allowlist[
#     "torch.utils.data.dataloader._SingleProcessDataLoaderIter.__next__"
# ] = "syft.lib.python.List"
# allowlist[
#     "torch.utils.data.dataloader._SingleProcessDataLoaderIter.next"
# ] = "syft.lib.python.List"

# Functional
# allowlist["torch.nn.functional.relu"] = "torch.Tensor"
# allowlist["torch.nn.functional.max_pool2d"] = "torch.Tensor"
# allowlist["torch.nn.functional.log_softmax"] = "torch.Tensor"
# allowlist["torch.nn.functional.nll_loss"] = "torch.Tensor"
# allowlist["torch.flatten"] = "torch.Tensor"

# Optimizers
# allowlist["torch.optim.Adadelta"] = "torch.optim.Adadelta"
# allowlist["torch.optim.lr_scheduler.StepLR"] = "torch.optim.lr_scheduler.StepLR"
# allowlist["torch.optim.lr_scheduler.StepLR.step"] = "syft.lib.python.SyNone"
# allowlist["torch.optim.Adadelta.zero_grad"] = "syft.lib.python.SyNone"
# allowlist["torch.optim.Adadelta.step"] = "syft.lib.python.SyNone"

# allowlist["torch.no_grad"] = "torch.autograd.grad_mode.no_grad"
# allowlist["torch.autograd.grad_mode.no_grad"] = "torch.autograd.grad_mode.no_grad"
# allowlist["torch.autograd.grad_mode.no_grad.__enter__"] = "syft.lib.python.SyNone"
# allowlist["torch.autograd.grad_mode.no_grad.__exit__"] = "syft.lib.python.SyNone"
