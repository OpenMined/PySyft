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

# SECTION - Tensor methods which have serde issues
# allowlist["torch.Tensor.to_dense"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.Tensor.to_mkldnn"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
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
allowlist["torch.Tensor.atanh_"] = {
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

# SECTION - Added in 1.7.0


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
allowlist["torch.Tensor.all"] = "torch.Tensor"
allowlist["torch.Tensor.allclose"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.angle"] = "torch.Tensor"
allowlist["torch.Tensor.any"] = "torch.Tensor"
allowlist["torch.Tensor.argmax"] = "torch.Tensor"
allowlist["torch.Tensor.argmin"] = "torch.Tensor"
allowlist["torch.Tensor.argsort"] = "torch.Tensor"
allowlist["torch.Tensor.asin_"] = "torch.Tensor"
allowlist["torch.Tensor.asin"] = "torch.Tensor"
allowlist["torch.Tensor.atan_"] = "torch.Tensor"
allowlist["torch.Tensor.atan"] = "torch.Tensor"
allowlist["torch.Tensor.atan2_"] = "torch.Tensor"
allowlist["torch.Tensor.atan2"] = "torch.Tensor"
allowlist["torch.Tensor.backward"] = "syft.lib.python._SyNone"
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
allowlist["torch.Tensor.chunk"] = "syft.lib.python.List"
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
allowlist["torch.Tensor.expm1_"] = "torch.Tensor"
allowlist["torch.Tensor.expm1"] = "torch.Tensor"
allowlist["torch.Tensor.exponential_"] = "torch.Tensor"
allowlist["torch.Tensor.fft"] = "torch.Tensor"
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
allowlist["torch.Tensor.ge_"] = "torch.Tensor"
allowlist["torch.Tensor.ge"] = "torch.Tensor"
allowlist["torch.Tensor.ger"] = "torch.Tensor"
allowlist["torch.Tensor.get_device"] = "syft.lib.python.Int"
allowlist["torch.Tensor.gt_"] = "torch.Tensor"
allowlist["torch.Tensor.gt"] = "torch.Tensor"
allowlist["torch.Tensor.half"] = "torch.Tensor"
allowlist["torch.Tensor.hardshrink"] = "torch.Tensor"
allowlist["torch.Tensor.histc"] = "torch.Tensor"
allowlist["torch.Tensor.ifft"] = "torch.Tensor"
allowlist["torch.Tensor.indices"] = "torch.Tensor"
allowlist["torch.Tensor.int_repr"] = "torch.Tensor"
allowlist["torch.Tensor.int"] = "torch.Tensor"
allowlist["torch.Tensor.inverse"] = "torch.Tensor"
allowlist["torch.Tensor.irfft"] = "torch.Tensor"
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
allowlist["torch.Tensor.le_"] = "torch.Tensor"
allowlist["torch.Tensor.le"] = "torch.Tensor"
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
allowlist["torch.Tensor.lt_"] = "torch.Tensor"
allowlist["torch.Tensor.lt"] = "torch.Tensor"
allowlist["torch.Tensor.lu"] = "syft.lib.python.List"  # actually a tuple
allowlist["torch.Tensor.matmul"] = "torch.Tensor"
allowlist["torch.Tensor.matrix_power"] = "torch.Tensor"
allowlist["torch.Tensor.mm"] = "torch.Tensor"
allowlist["torch.Tensor.mul_"] = "torch.Tensor"
allowlist["torch.Tensor.mul"] = "torch.Tensor"
allowlist["torch.Tensor.mvlgamma_"] = "torch.Tensor"
allowlist["torch.Tensor.mvlgamma"] = "torch.Tensor"
allowlist["torch.Tensor.ndim"] = "syft.lib.python.Int"
allowlist["torch.Tensor.ndimension"] = "syft.lib.python.Int"
allowlist["torch.Tensor.ne_"] = "torch.Tensor"
allowlist["torch.Tensor.ne"] = "torch.Tensor"
allowlist["torch.Tensor.neg_"] = "torch.Tensor"
allowlist["torch.Tensor.neg"] = "torch.Tensor"
allowlist["torch.Tensor.nelement"] = "syft.lib.python.Int"  # is this INSECURE???
allowlist["torch.Tensor.new_empty"] = "torch.Tensor"
allowlist["torch.Tensor.new_ones"] = "torch.Tensor"
allowlist["torch.Tensor.new_tensor"] = "torch.Tensor"
allowlist["torch.Tensor.new_zeros"] = "torch.Tensor"
allowlist["torch.Tensor.new"] = "torch.Tensor"
allowlist["torch.Tensor.nonzero"] = "torch.Tensor"
allowlist["torch.Tensor.norm"] = "torch.Tensor"
allowlist["torch.Tensor.normal_"] = "torch.Tensor"
allowlist["torch.Tensor.numel"] = "syft.lib.python.Int"  # is this INSECURE???
allowlist["torch.Tensor.orgqr"] = "torch.Tensor"
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
allowlist["torch.Tensor.q_per_channel_axis"] = "syft.lib.python.Int"
allowlist["torch.Tensor.q_per_channel_scales"] = "torch.Tensor"
allowlist["torch.Tensor.q_per_channel_zero_points"] = "torch.Tensor"
allowlist["torch.Tensor.q_scale"] = "syft.lib.python.Float"
allowlist["torch.Tensor.q_zero_point"] = "syft.lib.python.Int"
allowlist["torch.Tensor.random_"] = "torch.Tensor"
allowlist["torch.Tensor.reciprocal_"] = "torch.Tensor"
allowlist["torch.Tensor.reciprocal"] = "torch.Tensor"
allowlist["torch.Tensor.relu_"] = "torch.Tensor"
allowlist["torch.Tensor.relu"] = "torch.Tensor"
allowlist["torch.Tensor.repeat_interleave"] = "torch.Tensor"
allowlist["torch.Tensor.repeat"] = "torch.Tensor"
allowlist["torch.Tensor.requires_grad_"] = "torch.Tensor"
allowlist["torch.Tensor.requires_grad"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.reshape_as"] = "torch.Tensor"
allowlist["torch.Tensor.reshape"] = "torch.Tensor"
allowlist["torch.Tensor.resize_"] = "torch.Tensor"
allowlist["torch.Tensor.resize_as_"] = "torch.Tensor"
allowlist["torch.Tensor.resize"] = "torch.Tensor"
allowlist["torch.Tensor.retain_grad"] = "syft.lib.python._SyNone"
allowlist["torch.Tensor.rfft"] = "torch.Tensor"
allowlist["torch.Tensor.roll"] = "torch.Tensor"
allowlist["torch.Tensor.rot90"] = "torch.Tensor"
allowlist["torch.Tensor.round_"] = "torch.Tensor"
allowlist["torch.Tensor.round"] = "torch.Tensor"
allowlist["torch.Tensor.rsqrt_"] = "torch.Tensor"
allowlist["torch.Tensor.rsqrt"] = "torch.Tensor"
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
allowlist["torch.Tensor.softmax"] = "torch.Tensor"
allowlist["torch.Tensor.split"] = "syft.lib.python.List"
allowlist["torch.Tensor.sqrt_"] = "torch.Tensor"
allowlist["torch.Tensor.sqrt"] = "torch.Tensor"
allowlist["torch.Tensor.squeeze_"] = "torch.Tensor"
allowlist["torch.Tensor.squeeze"] = "torch.Tensor"
allowlist["torch.Tensor.std"] = "torch.Tensor"
allowlist["torch.Tensor.sub_"] = "torch.Tensor"
allowlist["torch.Tensor.sub"] = "torch.Tensor"
allowlist["torch.Tensor.sum_to_size"] = "torch.Tensor"
allowlist["torch.Tensor.sum"] = "torch.Tensor"
allowlist["torch.Tensor.t_"] = "torch.Tensor"
allowlist["torch.Tensor.t"] = "torch.Tensor"
allowlist["torch.Tensor.t"] = "torch.Tensor"
allowlist["torch.Tensor.T"] = "torch.Tensor"
allowlist["torch.Tensor.tan_"] = "torch.Tensor"
allowlist["torch.Tensor.tan"] = "torch.Tensor"
allowlist["torch.Tensor.tanh_"] = "torch.Tensor"
allowlist["torch.Tensor.tanh"] = "torch.Tensor"
allowlist["torch.Tensor.to"] = "torch.Tensor"
allowlist["torch.Tensor.tolist"] = "syft.lib.python.List"
allowlist["torch.Tensor.trace"] = "torch.Tensor"
allowlist["torch.Tensor.tril_"] = "torch.Tensor"
allowlist["torch.Tensor.tril"] = "torch.Tensor"
allowlist["torch.Tensor.triu_"] = "torch.Tensor"
allowlist["torch.Tensor.triu"] = "torch.Tensor"
allowlist["torch.Tensor.trunc_"] = "torch.Tensor"
allowlist["torch.Tensor.trunc"] = "torch.Tensor"
allowlist["torch.Tensor.type_as"] = "torch.Tensor"
allowlist["torch.Tensor.type"] = "syft.lib.python.String"
allowlist["torch.Tensor.unbind"] = "syft.lib.python.List"
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

# allowlist["torch.Tensor.__complex__"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.__torch_function__"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.amax"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.amin"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.arccos"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.arccos_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.arccosh"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.arccosh_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.arcsin"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.arcsin_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.arcsinh"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.arcsinh_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.arctan"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.arctan_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.arctanh"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.arctanh_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.clip"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.clip_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.count_nonzero"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.divide"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.divide_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.exp2"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.exp2_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.fix"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.fix_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.gcd"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.gcd_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.greater"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.greater_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.greater_equal"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.greater_equal_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.heaviside"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.heaviside_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.hypot"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.hypot_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.i0"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.i0_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.isneginf"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.isposinf"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.isreal"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.lcm"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.lcm_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.less"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.less_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.less_equal"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.less_equal_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.logit"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.logit_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.matrix_exp"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.maximum"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.minimum"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.movedim"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.multiply"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.multiply_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.nanquantile"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.nansum"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.negative"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.negative_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.nextafter"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.nextafter_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.outer"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.quantile"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.sgn"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.sgn_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.signbit"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.subtract"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.subtract_"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.unsafe_chunk"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.unsafe_split"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.unsafe_split_with_sizes"] = "unknown" # 1.7.0
# allowlist["torch.Tensor.vdot"] = "unknown" # 1.7.0

# --------------------------------------------------------------------------------------
# SECTION - Tensor methods with specific issues or require a special test combination
# --------------------------------------------------------------------------------------
# required for MNIST but marked as skip in the allowlist_test.json
allowlist["torch.Tensor.item"] = "syft.lib.python.Float"  # Union[bool, int, float]

# allowlist["torch.layout"] = "torch.layout" # requires protobuf serialization
# allowlist["torch.Size"] = "torch.Size" # requires protobuf serialization
# allowlist["torch.Tensor.__iter__"] = "unknown"  # How to handle return iterator?
# allowlist["torch.Tensor.__setitem__"] = "torch.Tensor"
# allowlist["torch.Tensor.addbmm_"] = "torch.Tensor" # multiple args?
# allowlist["torch.Tensor.addbmm"] = "torch.Tensor" # multiple args?
# allowlist["torch.Tensor.addcdiv_"] = "torch.Tensor" # multiple args?
# allowlist["torch.Tensor.addcdiv"] = "torch.Tensor" # multiple args?
# allowlist["torch.Tensor.addcmul_"] = "torch.Tensor" # multiple args?
# allowlist["torch.Tensor.addcmul"] = "torch.Tensor" # multiple args?
# allowlist["torch.Tensor.addmm_"] = "torch.Tensor" # multiple args?
# allowlist["torch.Tensor.addmm"] = "torch.Tensor" # multiple args?
# allowlist["torch.Tensor.addmv_"] = "torch.Tensor" # multiple args?
# allowlist["torch.Tensor.addmv"] = "torch.Tensor" # multiple args?
# allowlist["torch.Tensor.addr_"] = "torch.Tensor" # multiple args?
# allowlist["torch.Tensor.addr"] = "torch.Tensor" # multiple args?
# allowlist["torch.Tensor.align_as"] = "torch.Tensor" # named args
# allowlist["torch.Tensor.align_to"] = "torch.Tensor" # named args
# allowlist["torch.Tensor.apply_"] = "torch.Tensor" # takes a callable
# allowlist["torch.Tensor.as_strided_"] = "torch.Tensor" # multiple args?
# allowlist["torch.Tensor.as_strided"] = "torch.Tensor" # multiple args?
# allowlist["torch.Tensor.as_subclass"] = "torch.Tensor" # needs subclass passed in
# allowlist["torch.Tensor.baddbmm_"] = # multiple args?
# allowlist["torch.Tensor.baddbmm"] = # multiple args?
# allowlist["torch.Tensor.cholesky"] = "torch.Tensor" # needs correct example tensors
# allowlist["torch.Tensor.clamp_"] = "torch.Tensor" # clamps need min max etc
# allowlist["torch.Tensor.clamp_max_"] = "torch.Tensor"
# allowlist["torch.Tensor.clamp_max"] = "torch.Tensor"
# allowlist["torch.Tensor.clamp_min_"] = "torch.Tensor"
# allowlist["torch.Tensor.clamp_min"] = "torch.Tensor"
# allowlist["torch.Tensor.clamp"] = "torch.Tensor"
# allowlist["torch.Tensor.cummax"] = "torch.Tensor" # requires type torch.return_types.cummax
# allowlist["torch.Tensor.cummin"] = "torch.Tensor" # requires type torch.return_types.cummin
# allowlist["torch.Tensor.detach_"] = "torch.Tensor" # some issue with gradient_as_bucket_view
# allowlist["torch.Tensor.device"] = "torch.device" # requires torch.device serde
# allowlist["torch.Tensor.eig"] = "torch.Tensor" # requires torch.return_types.eig
# allowlist["torch.Tensor.expand"] = "torch.Tensor" # requires tuple input
# allowlist["torch.Tensor.gather"] = "torch.Tensor" # needs multiple inputs
# allowlist["torch.Tensor.geometric_"] = "torch.Tensor" # needs correct input or tuples
# allowlist["torch.Tensor.geqrf"] = "torch.Tensor"  # requires torch.return_types.geqrf
# allowlist["torch.Tensor.grad"] = "unknown"  # example with grad
# allowlist["torch.Tensor.imag"] = "torch.Tensor"  # requires complex or 1.6.0?
# allowlist["torch.Tensor.index_add_"] = "unknown" # requires three inputs
# allowlist["torch.Tensor.index_add"] = "unknown"  # requires three inputs
# allowlist["torch.Tensor.index_copy_"] = "unknown" # requires multiple inputs
# allowlist["torch.Tensor.index_copy"] = "unknown"  # requires multiple inputs
# allowlist["torch.Tensor.index_fill_"] = "unknown" # requires multiple inputs
# allowlist["torch.Tensor.index_fill"] = "unknown"  # requires multiple inputs
# allowlist["torch.Tensor.index_put_"] = "unknown" # requires multiple inputs
# allowlist["torch.Tensor.index_put"] = "unknown" # requires multiple inputs
# allowlist["torch.Tensor.index_select"] = "torch.Tensor"
# allowlist["torch.Tensor.istft"] = "torch.Tensor" # needs args
# allowlist["torch.Tensor.kthvalue"] = "unknown" # requires torch.return_types.kthvalue
# allowlist["torch.Tensor.layout"] = "torch.layout" # requires torch layout
# allowlist["torch.Tensor.lerp_"] = "unknown"  # requires multiple inputs
# allowlist["torch.Tensor.lerp"] = "unknown"  # requires multiple inputs
# allowlist["torch.Tensor.lstsq"] = "unknown"  # torch.return_types.lstsq
# allowlist["torch.Tensor.lu_solve"] = "torch.Tensor" # requires multiple inputs
# allowlist["torch.Tensor.map_"] = "unknown"  # requires callables
# allowlist["torch.Tensor.map2_"] = "unknown"  # requires callables
# allowlist["torch.Tensor.masked_fill_"] = "unknown"  # requires multiple inputs
# allowlist["torch.Tensor.masked_fill"] = "unknown"  # requires multiple inputs
# allowlist["torch.Tensor.masked_scatter_"] = "unknown" # requires multiple inputs
# allowlist["torch.Tensor.masked_scatter"] = "unknown" # requires multiple inputs
# allowlist["torch.Tensor.masked_select"] = "torch.Tensor"  # set input tensor data type
# allowlist["torch.Tensor.max"] = "torch.Tensor" # requires torch.return_types.max
# allowlist["torch.Tensor.mean"] = "torch.Tensor" # requires some test kwargs
# allowlist["torch.Tensor.median"] = "unknown"  # requires torch.return_types.median
# allowlist["torch.Tensor.min"] = "torch.Tensor" # requires some test kwargs
# allowlist["torch.Tensor.mode"] = "unknown"  # requires torch.return_types.mode
# allowlist["torch.Tensor.multinomial"] = "unknown"  # requires multiple args
# allowlist["torch.Tensor.mv"] = "unknown"  # needs the right tensor shapes
# allowlist["torch.Tensor.name"] = "Optional[str]" # requires named tensors and Optional
# allowlist["torch.Tensor.names"] = "Tuple[str]" # requires named tensors and Tuple
# allowlist["torch.Tensor.narrow_copy"] = "torch.Tensor"  # requires multiple args
# allowlist["torch.Tensor.narrow"] = "unknown" # requires multiple inputs
# allowlist["torch.Tensor.new_full"] = "unknown" # requires multiple inputs
# allowlist["torch.Tensor.numpy"] = "numpy.ndarray"  # requires numpy.ndarray
# allowlist["torch.Tensor.ormqr"] = "unknown"  # requires two tensors as arguments
# allowlist["torch.Tensor.put_"] = "unknown"  # requires multiple inputs
# allowlist["torch.Tensor.qr"] = "unknown"  # requires torch.return_types.qr
# allowlist["torch.Tensor.qscheme"] = "unknown"  # requires  torch.qscheme
# allowlist["torch.Tensor.real"] = "torch.Tensor"  # requires complex or 1.6.0?
# allowlist["torch.Tensor.refine_names"] = "unknown" # requires multiple inputs
# allowlist["torch.Tensor.reinforce"] = "unknown"  # requires reinforce
# allowlist["torch.Tensor.rename_"] = "torch.Tensor"  # requires multiple inputs
# allowlist["torch.Tensor.rename"] = "torch.Tensor"  # requires multiple inputs
# allowlist["torch.Tensor.renorm_"] = "unknown"  # requires multiple inputs
# allowlist["torch.Tensor.renorm"] = "unknown"  # requires multiple inputs
# allowlist["torch.Tensor.scatter_"] = "torch.Tensor"  # requires multiple inputs
# allowlist["torch.Tensor.scatter_add_"] = "unknown"  # requires multiple inputs
# allowlist["torch.Tensor.scatter_add"] = "unknown"  # requires multiple inputs
# allowlist["torch.Tensor.scatter"] = "unknown"   # requires multiple inputs
# allowlist["torch.Tensor.select"] = "unknown"   # requires multiple inputs
# allowlist["torch.Tensor.shape"] = "torch.Size" # requires torch.Size
# allowlist["torch.Tensor.size"] = "unknown"  # requires union and torch.size
# allowlist["torch.Tensor.slogdet"] = "unknown"  # torch.return_types.slogdet
# allowlist["torch.Tensor.smm"] = "unknown"  # requires sparse tensors
# allowlist["torch.Tensor.solve"] = "torch.Tensor"  # torch.return_types.solve
# allowlist["torch.Tensor.sort"] = "unknown"  # torch.return_types.sort
# allowlist["torch.Tensor.sparse_dim"] = "unknown"  # requires sparse
# allowlist["torch.Tensor.sparse_mask"] = "unknown"  # requires sparse
# allowlist["torch.Tensor.sparse_resize_"] = "unknown" # requires sparse and multiple inputs
# allowlist["torch.Tensor.sparse_resize_and_clear_"] = "unknown" # requires sparse and multiple inputs
# allowlist["torch.Tensor.split_with_sizes"] = "unknown"  # requires tuples input
# allowlist["torch.Tensor.sspaddmm"] = "unknown"  # multiple inputs
# allowlist["torch.Tensor.stft"] = "unknown"  # multiple inputs
# allowlist["torch.Tensor.stride"] = "unknown"  # union type int or tuple / list
# allowlist["torch.Tensor.svd"] = "unknown"  # torch.return_types.svd
# allowlist["torch.Tensor.symeig"] = "unknown"  #torch.return_types.symeig
# allowlist["torch.Tensor.take"] = "torch.Tensor" # requires long tensor input only
# allowlist["torch.Tensor.topk"] = "unknown"  # torch.return_types.topk
# allowlist["torch.Tensor.transpose_"] = "torch.Tensor" # requires two inputs
# allowlist["torch.Tensor.transpose"] = "torch.Tensor" # requires two inputs
# allowlist["torch.Tensor.triangular_solve"] = "unknown"  # torch.return_types.triangular_solve
# allowlist["torch.Tensor.unflatten"] = "unknown"  # requires multiple args
# allowlist["torch.Tensor.unfold"] = "torch.Tensor" # requires three inputs
# allowlist["torch.Tensor.uniform_"] = "torch.Tensor"
# allowlist["torch.Tensor.unique_consecutive"] = "torch.Tensor" # requires Union / Tuple
# allowlist["torch.Tensor.values"] = "unknown"  # requires sparse


# --------------------------------------------------------------------------------------
# SECTION - Tensor methods which we wont support
# --------------------------------------------------------------------------------------
# allowlist["torch.Tensor.resize_as"] = "unknown" deprecated


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
allowlist["torch.manual_seed"] = "torch.Generator"
allowlist["torch.Generator"] = "torch.Generator"
allowlist["torch.Generator.get_state"] = "torch.Tensor"
allowlist["torch.Generator.set_state"] = "torch.Generator"
allowlist["torch.exp"] = "torch.Tensor"

# Modules
allowlist["torch.nn.Module"] = "torch.nn.Module"
allowlist["torch.nn.Module.__call__"] = "torch.nn.Tensor"
allowlist["torch.nn.Module.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Module.train"] = "torch.nn.Module"
allowlist["torch.nn.Module.cuda"] = "torch.nn.Module"
allowlist["torch.nn.Module.cpu"] = "torch.nn.Module"
allowlist["torch.nn.Module.state_dict"] = "syft.lib.python.Dict"

allowlist["torch.nn.Conv2d"] = "torch.nn.Conv2d"
allowlist["torch.nn.Conv2d.__call__"] = "torch.nn.Conv2d"
allowlist["torch.nn.Conv2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Conv2d.train"] = "torch.nn.Conv2d"
allowlist["torch.nn.Conv2d.cuda"] = "torch.nn.Conv2d"
allowlist["torch.nn.Conv2d.cpu"] = "torch.nn.Conv2d"
allowlist["torch.nn.Conv2d.state_dict"] = "syft.lib.python.Dict"

allowlist["torch.nn.Dropout2d"] = "torch.nn.Dropout2d"
allowlist["torch.nn.Dropout2d.__call__"] = "torch.nn.Dropout2d"
allowlist["torch.nn.Dropout2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Dropout2d.train"] = "torch.nn.Dropout2d"
allowlist["torch.nn.Dropout2d.cuda"] = "torch.nn.Dropout2d"
allowlist["torch.nn.Dropout2d.cpu"] = "torch.nn.Dropout2d"

allowlist["torch.nn.Linear"] = "torch.nn.Linear"
allowlist["torch.nn.Linear.__call__"] = "torch.nn.Linear"
allowlist["torch.nn.Linear.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Linear.train"] = "torch.nn.Linear"
allowlist["torch.nn.Linear.cuda"] = "torch.nn.Linear"
allowlist["torch.nn.Linear.cpu"] = "torch.nn.Linear"
allowlist["torch.nn.Linear.state_dict"] = "syft.lib.python.Dict"

# DataLoader
allowlist["torch.utils.data.DataLoader"] = "torch.utils.data.DataLoader"
allowlist[
    "torch.utils.data.DataLoader.__iter__"
] = "torch.utils.data.dataloader._SingleProcessDataLoaderIter"
allowlist["torch.utils.data.DataLoader.__len__"] = "syft.lib.python.Int"
allowlist[
    "torch.utils.data.dataloader._SingleProcessDataLoaderIter"
] = "torch.utils.data.dataloader._SingleProcessDataLoaderIter"
allowlist[
    "torch.utils.data.dataloader._SingleProcessDataLoaderIter.__len__"
] = "syft.lib.python.Int"
allowlist[
    "torch.utils.data.dataloader._SingleProcessDataLoaderIter.__iter__"
] = "torch.utils.data.dataloader._SingleProcessDataLoaderIter"

# working for part
allowlist[
    "torch.utils.data.dataloader._SingleProcessDataLoaderIter.__next__"
] = "torch.Tensor"
allowlist[
    "torch.utils.data.dataloader._SingleProcessDataLoaderIter.next"
] = "torch.Tensor"

# we are returning syft.lib.python.List so that we can __getitem__ on the return of
# enumerate(train_loader)
allowlist[
    "torch.utils.data.dataloader._SingleProcessDataLoaderIter.__next__"
] = "syft.lib.python.List"
allowlist[
    "torch.utils.data.dataloader._SingleProcessDataLoaderIter.next"
] = "syft.lib.python.List"

# Functional
allowlist["torch.nn.functional.relu"] = "torch.Tensor"
allowlist["torch.nn.functional.max_pool2d"] = "torch.Tensor"
allowlist["torch.nn.functional.log_softmax"] = "torch.Tensor"
allowlist["torch.nn.functional.nll_loss"] = "torch.Tensor"
allowlist["torch.flatten"] = "torch.Tensor"

# Optimizers
allowlist["torch.optim.Adadelta"] = "torch.optim.Adadelta"
allowlist["torch.optim.lr_scheduler.StepLR"] = "torch.optim.lr_scheduler.StepLR"
allowlist["torch.optim.lr_scheduler.StepLR.step"] = "syft.lib.python._SyNone"
allowlist["torch.optim.Adadelta.zero_grad"] = "syft.lib.python._SyNone"
allowlist["torch.optim.Adadelta.step"] = "syft.lib.python._SyNone"

# Autograd
allowlist["torch.no_grad"] = "torch.autograd.grad_mode.no_grad"
allowlist["torch.autograd.grad_mode.no_grad"] = "torch.autograd.grad_mode.no_grad"
allowlist["torch.autograd.grad_mode.no_grad.__enter__"] = "syft.lib.python._SyNone"
allowlist["torch.autograd.grad_mode.no_grad.__exit__"] = "syft.lib.python._SyNone"


allowlist["torch.nn.Sequential"] = "torch.nn.Sequential"
allowlist["torch.nn.Sequential.cpu"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Sequential.cuda"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Sequential.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Sequential.train"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Sequential.eval"] = "syft.lib.python._SyNone"
allowlist["torch.nn.Sequential.__call__"] = "torch.Tensor"

allowlist["torch.nn.ReLU"] = "torch.nn.ReLU"
allowlist["torch.nn.MaxPool2d"] = "torch.nn.MaxPool2d"
allowlist["torch.nn.Flatten"] = "torch.nn.Flatten"
allowlist["torch.nn.Softmax"] = "torch.nn.Softmax"
allowlist["torch.nn.LogSoftmax"] = "torch.nn.LogSoftmax"
