# stdlib
from typing import Dict
from typing import Union

allowlist: Dict[str, Union[str, Dict[str, str]]] = {}  # (path: str, return_type:type)

# SECTION - add the capital constructor
allowlist["torch.Tensor"] = "torch.Tensor"
# TODO here for testing purpose
allowlist["torch.zeros_like"] = "torch.Tensor"

# SECTION - Tensor methods which return a torch tensor object

allowlist["torch.Tensor.T"] = "torch.Tensor"
allowlist["torch.Tensor.t"] = "torch.Tensor"
allowlist["torch.Tensor.__abs__"] = "torch.Tensor"
allowlist["torch.Tensor.__add__"] = "torch.Tensor"
allowlist["torch.Tensor.__and__"] = "torch.Tensor"
# allowlist['torch.Tensor.__array__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__array_priority__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__array_wrap__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__bool__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__class__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__contains__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__deepcopy__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__delattr__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__delitem__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__dict__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__dir__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
allowlist[
    "torch.Tensor.__div__"
] = {  # exists in 1.4.0 but causes fatal exception on non floats
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}

# allowlist['torch.Tensor.__doc__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
allowlist["torch.Tensor.__eq__"] = "torch.Tensor"
allowlist["torch.Tensor.__float__"] = "torch.Tensor"
allowlist[
    "torch.Tensor.__floordiv__"
] = {  # exists in 1.4.0 but causes fatal exception on non floats
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
# allowlist['torch.Tensor.__format__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
allowlist["torch.Tensor.__ge__"] = "torch.Tensor"
# allowlist['torch.Tensor.__getattribute__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
allowlist["torch.Tensor.__getitem__"] = "torch.Tensor"
allowlist["torch.Tensor.__gt__"] = "torch.Tensor"
# allowlist['torch.Tensor.__hash__'] = #
# allowlist['torch.Tensor.__iadd__'] = #
# allowlist['torch.Tensor.__iand__'] = #
# allowlist['torch.Tensor.__idiv__'] = #
# allowlist['torch.Tensor.__ifloordiv__'] = #
# allowlist['torch.Tensor.__ilshift__'] = #
# allowlist['torch.Tensor.__imul__'] = #
# allowlist['torch.Tensor.__index__'] = #
# allowlist['torch.Tensor.__init__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__init_subclass__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
allowlist["torch.Tensor.__int__"] = "torch.Tensor"
allowlist["torch.Tensor.__invert__"] = "torch.Tensor"
# allowlist['torch.Tensor.__ior__'] = #
# allowlist['torch.Tensor.__ipow__'] = #
# allowlist['torch.Tensor.__irshift__'] = #
# allowlist['torch.Tensor.__isub__'] = #
# allowlist['torch.Tensor.__iter__'] = #
# allowlist['torch.Tensor.__itruediv__'] = #
# allowlist['torch.Tensor.__ixor__'] = #
allowlist["torch.Tensor.__le__"] = "torch.Tensor"
# allowlist['torch.Tensor.__len__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST YET - talk to TRASK
allowlist["torch.Tensor.__long__"] = "torch.Tensor"
allowlist["torch.Tensor.__lshift__"] = "torch.Tensor"
allowlist["torch.Tensor.__lt__"] = "torch.Tensor"
allowlist["torch.Tensor.__matmul__"] = "torch.Tensor"
# allowlist['torch.Tensor.__mod__'] = #
# allowlist['torch.Tensor.__module__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
allowlist["torch.Tensor.__mul__"] = "torch.Tensor"
allowlist["torch.Tensor.__ne__"] = "torch.Tensor"
allowlist["torch.Tensor.__neg__"] = "torch.Tensor"
# allowlist['torch.Tensor.__new__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__nonzero__'] = #
allowlist["torch.Tensor.__or__"] = "torch.Tensor"
allowlist["torch.Tensor.__pow__"] = "torch.Tensor"
allowlist["torch.Tensor.__radd__"] = "torch.Tensor"
allowlist["torch.Tensor.__rdiv__"] = "torch.Tensor"
# allowlist['torch.Tensor.__reduce__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__reduce_ex__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__repr__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__reversed__'] = #
allowlist[
    "torch.Tensor.__rfloordiv__"
] = {  # exists in 1.4.0 but causes fatal exception on non floats
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.__rmul__"] = "torch.Tensor"
allowlist["torch.Tensor.__rpow__"] = "torch.Tensor"
allowlist["torch.Tensor.__rshift__"] = "torch.Tensor"
allowlist["torch.Tensor.__rsub__"] = "torch.Tensor"
allowlist["torch.Tensor.__rtruediv__"] = "torch.Tensor"
# allowlist['torch.Tensor.__setattr__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__setitem__'] = #
# allowlist['torch.Tensor.__setstate__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__sizeof__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST YET - talk to TRASK
# allowlist['torch.Tensor.__str__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
allowlist["torch.Tensor.__sub__"] = "torch.Tensor"
# allowlist['torch.Tensor.__subclasshook__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__truediv__'] = #
# allowlist['torch.Tensor.__weakref__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
allowlist["torch.Tensor.__xor__"] = "torch.Tensor"
# allowlist['torch.Tensor._backward_hooks'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor._base'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor._cdata'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor._coalesced_'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor._dimI'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor._dimV'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor._grad'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor._grad_fn'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor._indices'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor._is_view'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor._make_subclass'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor._nnz'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor._update_names'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor._values'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor._version'] = #
allowlist["torch.Tensor.abs"] = "torch.Tensor"
allowlist["torch.Tensor.abs_"] = "torch.Tensor"
allowlist["torch.Tensor.acos"] = "torch.Tensor"
allowlist["torch.Tensor.acos_"] = "torch.Tensor"
allowlist["torch.Tensor.add"] = "torch.Tensor"
allowlist["torch.Tensor.add_"] = "torch.Tensor"
# allowlist['torch.Tensor.addbmm'] = #
# allowlist['torch.Tensor.addbmm_'] = #
# allowlist['torch.Tensor.addcdiv'] = #
# allowlist['torch.Tensor.addcdiv_'] = #
# allowlist['torch.Tensor.addcmul'] = #
# allowlist['torch.Tensor.addcmul_'] = #
# allowlist['torch.Tensor.addmm'] = #
# allowlist['torch.Tensor.addmm_'] = #
# allowlist['torch.Tensor.addmv') - trie] = #
# allowlist['torch.Tensor.addmv_'] = #
# allowlist['torch.Tensor.addr'] = #
# allowlist['torch.Tensor.addr_'] = #
# allowlist['torch.Tensor.align_as'] = #
# allowlist['torch.Tensor.align_to'] = #
# allowlist['torch.Tensor.all'] = #
# allowlist['torch.Tensor.allclose'] = #
# allowlist['torch.Tensor.angle'] = #
# allowlist['torch.Tensor.any'] = #
# allowlist['torch.Tensor.apply_'] = #
# allowlist['torch.Tensor.argmax'] = #
# allowlist['torch.Tensor.argmin'] = #
# allowlist['torch.Tensor.argsort'] = #
# allowlist['torch.Tensor.as_strided'] = #
# allowlist['torch.Tensor.as_strided_'] = #
allowlist["torch.Tensor.asin"] = "torch.Tensor"
allowlist["torch.Tensor.asin_"] = "torch.Tensor"
allowlist["torch.Tensor.atan"] = "torch.Tensor"
allowlist["torch.Tensor.atan_"] = "torch.Tensor"
allowlist["torch.Tensor.atan2"] = "torch.Tensor"
allowlist["torch.Tensor.atan2_"] = "torch.Tensor"
allowlist["torch.Tensor.backward"] = "syft.lib.python.SyNone"
# allowlist['torch.Tensor.baddbmm'] = #
# allowlist['torch.Tensor.baddbmm_'] = #
# allowlist['torch.Tensor.bernoulli'] = #
# allowlist['torch.Tensor.bernoulli_'] = #
# allowlist['torch.Tensor.bfloat16'] = #
# allowlist['torch.Tensor.binary'] = #
# allowlist['torch.Tensor.bincount'] = #
allowlist["torch.Tensor.bitwise_and"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.bitwise_and_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.bitwise_not"] = "torch.Tensor"
allowlist["torch.Tensor.bitwise_not_"] = "torch.Tensor"
allowlist["torch.Tensor.bitwise_or"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.bitwise_or_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.bitwise_xor"] = "torch.Tensor"
allowlist["torch.Tensor.bitwise_xor_"] = "torch.Tensor"
# allowlist['torch.Tensor.bmm'] = #
# allowlist['torch.Tensor.bool'] = #
# allowlist['torch.Tensor.byte'] = #
# allowlist['torch.Tensor.cauchy_'] = #
allowlist["torch.Tensor.ceil"] = "torch.Tensor"
allowlist["torch.Tensor.ceil_"] = "torch.Tensor"
allowlist["torch.Tensor.char"] = "torch.Tensor"
# allowlist['torch.Tensor.cholesky'] = #
# allowlist['torch.Tensor.cholesky_inverse'] = #
# allowlist['torch.Tensor.chunk'] = #
allowlist["torch.Tensor.clamp"] = "torch.Tensor"
allowlist["torch.Tensor.clamp_"] = "torch.Tensor"
allowlist["torch.Tensor.clamp_max"] = "torch.Tensor"
allowlist["torch.Tensor.clamp_max_"] = "torch.Tensor"
allowlist["torch.Tensor.clamp_min"] = "torch.Tensor"
allowlist["torch.Tensor.clamp_min_"] = "torch.Tensor"
allowlist["torch.Tensor.clone"] = "torch.Tensor"
# allowlist['torch.Tensor.coalesce'] = #
# allowlist['torch.Tensor.conj'] = #
# allowlist['torch.Tensor.contiguous'] = #
allowlist["torch.Tensor.copy_"] = "torch.Tensor"
allowlist["torch.Tensor.cos"] = "torch.Tensor"
allowlist["torch.Tensor.cos_"] = "torch.Tensor"
allowlist["torch.Tensor.cosh"] = "torch.Tensor"
allowlist["torch.Tensor.cosh_"] = "torch.Tensor"
allowlist["torch.Tensor.cpu"] = "torch.Tensor"
# allowlist['torch.Tensor.cross'] = #
# allowlist['torch.Tensor.cuda'] = #
# allowlist['torch.Tensor.cummax'] = #
# allowlist['torch.Tensor.cummin'] = #
# allowlist['torch.Tensor.cumprod'] = #
# allowlist['torch.Tensor.cumsum'] = #
allowlist["torch.Tensor.data"] = "torch.Tensor"
# allowlist['torch.Tensor.data_ptr'] = #
# allowlist['torch.Tensor.dense_dim'] = #
# allowlist['torch.Tensor.dequantize'] = #
# allowlist['torch.Tensor.describe'] = #
# allowlist['torch.Tensor.det'] = #
# allowlist['torch.Tensor.detach'] = #
# allowlist['torch.Tensor.detach_'] = #
# allowlist['torch.Tensor.device'] = #
allowlist["torch.Tensor.diag"] = "torch.Tensor"
# allowlist['torch.Tensor.diag_embed'] = #
# allowlist['torch.Tensor.diagflat'] = #
allowlist["torch.Tensor.diagonal"] = "torch.Tensor"
# allowlist['torch.Tensor.digamma'] = #
# allowlist['torch.Tensor.digamma_'] = #
# allowlist['torch.Tensor.dim'] = #
# allowlist['torch.Tensor.dist'] = #
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
allowlist["torch.Tensor.dot"] = "torch.Tensor"
allowlist["torch.Tensor.double"] = "torch.Tensor"
# allowlist['torch.Tensor.dtype'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST YET - talk to TRASK
# allowlist['torch.Tensor.eig'] = #
# allowlist['torch.Tensor.element_size'] = #
allowlist["torch.Tensor.eq"] = "torch.Tensor"
allowlist["torch.Tensor.eq_"] = "torch.Tensor"
# allowlist['torch.Tensor.equal'] = #
allowlist["torch.Tensor.erf"] = "torch.Tensor"
allowlist["torch.Tensor.erf_"] = "torch.Tensor"
allowlist["torch.Tensor.erfc"] = "torch.Tensor"
allowlist["torch.Tensor.erfc_"] = "torch.Tensor"
allowlist["torch.Tensor.erfinv"] = "torch.Tensor"
allowlist["torch.Tensor.erfinv_"] = "torch.Tensor"
allowlist["torch.Tensor.exp"] = "torch.Tensor"
allowlist["torch.Tensor.exp_"] = "torch.Tensor"
# allowlist['torch.Tensor.expand'] = #
# allowlist['torch.Tensor.expand_as'] = #
allowlist["torch.Tensor.expm1"] = "torch.Tensor"
allowlist["torch.Tensor.expm1_"] = "torch.Tensor"
# allowlist['torch.Tensor.exponential_'] = #
# allowlist['torch.Tensor.fft'] = #
# allowlist['torch.Tensor.fill_'] = #
# allowlist['torch.Tensor.fill_diagonal_'] = #
allowlist["torch.Tensor.flatten"] = "torch.Tensor"
# allowlist['torch.Tensor.flip'] = #
allowlist["torch.Tensor.float"] = "torch.Tensor"
allowlist["torch.Tensor.floor"] = "torch.Tensor"
allowlist["torch.Tensor.floor_"] = "torch.Tensor"
allowlist["torch.Tensor.floor_divide"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.floor_divide_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
# allowlist['torch.Tensor.fmod'] = #
# allowlist['torch.Tensor.fmod_'] = #
allowlist["torch.Tensor.frac"] = "torch.Tensor"
allowlist["torch.Tensor.frac_"] = "torch.Tensor"
# allowlist['torch.Tensor.gather'] = #
allowlist["torch.Tensor.ge"] = "torch.Tensor"
allowlist["torch.Tensor.ge_"] = "torch.Tensor"
# allowlist['torch.Tensor.geometric_'] = #
# allowlist['torch.Tensor.geqrf'] = #
allowlist["torch.Tensor.ger"] = "torch.Tensor"
# allowlist['torch.Tensor.get_device'] = #
# allowlist['torch.Tensor.grad'] = #
# allowlist['torch.Tensor.grad_fn'] = #
allowlist["torch.Tensor.gt"] = "torch.Tensor"
allowlist["torch.Tensor.gt_"] = "torch.Tensor"
allowlist["torch.Tensor.half"] = "torch.Tensor"
# allowlist['torch.Tensor.hardshrink'] = #
# allowlist['torch.Tensor.has_names'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.hex'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.histc'] = #
# allowlist['torch.Tensor.id'] = #
# allowlist['torch.Tensor.ifft'] = #
# allowlist['torch.Tensor.imag'] = "torch.Tensor" # requires complex
# allowlist['torch.Tensor.index_add'] = #
# allowlist['torch.Tensor.index_add_'] = #
# allowlist['torch.Tensor.index_copy'] = #
# allowlist['torch.Tensor.index_copy_'] = #
# allowlist['torch.Tensor.index_fill'] = #
# allowlist['torch.Tensor.index_fill_'] = #
# allowlist['torch.Tensor.index_put'] = #
# allowlist['torch.Tensor.index_put_'] = #
# allowlist['torch.Tensor.index_select'] = #
# allowlist['torch.Tensor.indices'] = #
allowlist["torch.Tensor.int"] = "torch.Tensor"
# allowlist['torch.Tensor.int_repr'] = #
# allowlist['torch.Tensor.inverse'] = #
# allowlist['torch.Tensor.irfft'] = #
# allowlist['torch.Tensor.is_coalesced'] = #
# allowlist['torch.Tensor.is_complex'] = #
# allowlist['torch.Tensor.is_contiguous'] = #
allowlist["torch.Tensor.is_cuda"] = "syft.lib.python.Bool"
# allowlist['torch.Tensor.is_distributed'] = #
# allowlist["torch.Tensor.is_floating_point"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_leaf"] = "syft.lib.python.Bool"
allowlist["torch.Tensor.is_meta"] = {
    "return_type": "syft.lib.python.Bool",
    "min_version": "1.6.0",
}
allowlist["torch.Tensor.is_mkldnn"] = "syft.lib.python.Bool"
# allowlist['torch.Tensor.is_nonzero'] = #
# allowlist['torch.Tensor.is_pinned'] = #
allowlist["torch.Tensor.is_quantized"] = "syft.lib.python.Bool"
# allowlist['torch.Tensor.is_same_size'] = #
# allowlist['torch.Tensor.is_set_to'] = #
# allowlist['torch.Tensor.is_shared'] = #
# allowlist['torch.Tensor.is_signed'] = #
allowlist["torch.Tensor.is_sparse"] = "syft.lib.python.Bool"
# allowlist['torch.Tensor.isclose'] = #
# allowlist['torch.Tensor.item'] = #
# allowlist['torch.Tensor.json'] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.kthvalue'] = #
# allowlist["torch.Tensor.layout"] = "torch.layout"
allowlist["torch.Tensor.le"] = "torch.Tensor"
allowlist["torch.Tensor.le_"] = "torch.Tensor"
# allowlist['torch.Tensor.lerp'] = #
# allowlist['torch.Tensor.lerp_'] = #
allowlist["torch.Tensor.lgamma"] = "torch.Tensor"
allowlist["torch.Tensor.lgamma_"] = "torch.Tensor"
allowlist["torch.Tensor.log"] = "torch.Tensor"
allowlist["torch.Tensor.log_"] = "torch.Tensor"
allowlist["torch.Tensor.log10"] = "torch.Tensor"
allowlist["torch.Tensor.log10_"] = "torch.Tensor"
allowlist["torch.Tensor.log1p"] = "torch.Tensor"
allowlist["torch.Tensor.log1p_"] = "torch.Tensor"
allowlist["torch.Tensor.log2"] = "torch.Tensor"
allowlist["torch.Tensor.log2_"] = "torch.Tensor"
# allowlist['torch.Tensor.log_normal_'] = #
# allowlist['torch.Tensor.log_softmax'] = #

# allowlist['torch.Tensor.logdet'] = #
allowlist["torch.Tensor.logical_and"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.logical_and_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.logical_not"] = "torch.Tensor"
allowlist["torch.Tensor.logical_not_"] = "torch.Tensor"
allowlist["torch.Tensor.logical_or"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.logical_or_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.logical_xor"] = "torch.Tensor"
allowlist["torch.Tensor.logical_xor_"] = "torch.Tensor"
# allowlist['torch.Tensor.logsumexp'] = #
allowlist["torch.Tensor.long"] = "torch.Tensor"
# allowlist['torch.Tensor.lstsq'] = #
allowlist["torch.Tensor.lt"] = "torch.Tensor"
allowlist["torch.Tensor.lt_"] = "torch.Tensor"
# allowlist['torch.Tensor.lu'] = #
# allowlist['torch.Tensor.lu_solve'] = #
# allowlist['torch.Tensor.map2_'] = #
# allowlist['torch.Tensor.map_'] = #
# allowlist['torch.Tensor.masked_fill'] = #
# allowlist['torch.Tensor.masked_fill_'] = #
# allowlist['torch.Tensor.masked_scatter'] = #
# allowlist['torch.Tensor.masked_scatter_'] = #
# allowlist['torch.Tensor.masked_select'] = #
allowlist["torch.Tensor.matmul"] = "torch.Tensor"
# allowlist['torch.Tensor.matrix_power'] = #
# allowlist["torch.Tensor.max"] = "torch.Tensor" # some issues
allowlist["torch.Tensor.mean"] = "torch.Tensor"
# allowlist["torch.median"] = # "torch.Tensor"  # requires torch.return_types.median
# allowlist["torch.Tensor.median"] = # "torch.Tensor"  # requires torch.return_types.median
# allowlist["torch.Tensor.min"] = "torch.Tensor" # some issues
allowlist["torch.Tensor.mm"] = "torch.Tensor"
# allowlist['torch.Tensor.mode'] = #
allowlist["torch.Tensor.mul"] = "torch.Tensor"
allowlist["torch.Tensor.mul_"] = "torch.Tensor"
# allowlist['torch.Tensor.multinomial'] = #
# allowlist['torch.Tensor.mv'] = #
# allowlist['torch.Tensor.mvlgamma'] = #
# allowlist['torch.Tensor.mvlgamma_'] = #
# allowlist["torch.Tensor.name"] = "Optional[str]"
# allowlist["torch.Tensor.names"] = "Tuple[str]"
# allowlist['torch.Tensor.narrow'] = #
# allowlist['torch.Tensor.narrow_copy'] = #
allowlist["torch.Tensor.ndim"] = "syft.lib.python.Int"
# allowlist['torch.Tensor.ndimension'] = #
allowlist["torch.Tensor.ne"] = "torch.Tensor"
allowlist["torch.Tensor.ne_"] = "torch.Tensor"
allowlist["torch.Tensor.neg"] = "torch.Tensor"
allowlist["torch.Tensor.neg_"] = "torch.Tensor"
# allowlist['torch.Tensor.nelement'] = #
# allowlist["torch.Tensor.new"] = # "torch.Tensor" # deprecated and returns random values
# allowlist['torch.Tensor.new_empty'] = #
# allowlist['torch.Tensor.new_full'] = #
# allowlist['torch.Tensor.new_ones'] = #
allowlist["torch.Tensor.new_tensor"] = "torch.Tensor"
# allowlist['torch.Tensor.new_zeros'] = #
allowlist["torch.Tensor.nonzero"] = "torch.Tensor"
allowlist["torch.Tensor.norm"] = "torch.Tensor"
# allowlist['torch.Tensor.normal_'] = #
# allowlist['torch.Tensor.numel'] = #
# allowlist['torch.Tensor.numpy'] = #
allowlist["torch.Tensor.orgqr"] = "torch.Tensor"
# allowlist['torch.Tensor.ormqr') # requires two tensors as argument] = #
allowlist["torch.Tensor.output_nr"] = "syft.lib.python.Int"
# allowlist['torch.Tensor.permute'] = #
# allowlist['torch.Tensor.pin_memory'] = #
allowlist["torch.Tensor.pinverse"] = "torch.Tensor"
# allowlist['torch.Tensor.polygamma'] = #
# allowlist['torch.Tensor.polygamma_'] = #
allowlist["torch.Tensor.pow"] = "torch.Tensor"
allowlist["torch.Tensor.pow_"] = "torch.Tensor"
# allowlist['torch.Tensor.prelu'] = #
allowlist["torch.Tensor.prod"] = "torch.Tensor"
# allowlist['torch.Tensor.proto'] = #
# allowlist['torch.Tensor.put_'] = #
# allowlist['torch.Tensor.q_per_channel_axis'] = #
# allowlist['torch.Tensor.q_per_channel_scales'] = #
# allowlist['torch.Tensor.q_per_channel_zero_points'] = #
# allowlist['torch.Tensor.q_scale'] = #
# allowlist['torch.Tensor.q_zero_point'] = #
# allowlist['torch.Tensor.qr'] = #
# allowlist['torch.Tensor.qscheme'] = #
# allowlist['torch.Tensor.random_'] = #
# allowlist["torch.Tensor.real"] = "torch.Tensor"  # requires complex
allowlist["torch.Tensor.reciprocal"] = "torch.Tensor"
allowlist["torch.Tensor.reciprocal_"] = "torch.Tensor"
# allowlist['torch.Tensor.record_stream'] =SECURITY WARNING: DO NOT ADD TO ALLOW LIST#
# allowlist['torch.Tensor.refine_names'] = #
# allowlist['torch.Tensor.register_hook'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.reinforce'] = #
allowlist["torch.Tensor.relu"] = "torch.Tensor"
allowlist["torch.Tensor.relu_"] = "torch.Tensor"
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
# allowlist['torch.Tensor.rename'] = #
# allowlist['torch.Tensor.rename_'] = #
# allowlist['torch.Tensor.renorm'] = #
# allowlist['torch.Tensor.renorm_'] = #
# allowlist['torch.Tensor.repeat'] = #
# allowlist['torch.Tensor.repeat_interleave'] = #
allowlist["torch.Tensor.requires_grad"] = "syft.lib.python.Bool"
# RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
# allowlist["torch.Tensor.requires_grad_"] = "torch.Tensor" # error above
# allowlist['torch.Tensor.reshape'] = #
allowlist["torch.Tensor.reshape_as"] = "torch.Tensor"
# allowlist['torch.Tensor.resize'] = #
# allowlist['torch.Tensor.resize_'] = #
# allowlist["torch.Tensor.resize_as"] = # deprecated
allowlist["torch.Tensor.resize_as_"] = "torch.Tensor"
# allowlist['torch.Tensor.retain_grad'] = #
# allowlist['torch.Tensor.rfft'] = #
# allowlist['torch.Tensor.roll'] = #
allowlist["torch.Tensor.rot90"] = "torch.Tensor"
allowlist["torch.Tensor.round"] = "torch.Tensor"
allowlist["torch.Tensor.round_"] = "torch.Tensor"
allowlist["torch.Tensor.rsqrt"] = "torch.Tensor"
allowlist["torch.Tensor.rsqrt_"] = "torch.Tensor"
# allowlist['torch.Tensor.scatter'] = #
# allowlist['torch.Tensor.scatter_'] = #
# allowlist['torch.Tensor.scatter_add'] = #
# allowlist['torch.Tensor.scatter_add_'] = #
# allowlist['torch.Tensor.select'] = #
# allowlist['torch.Tensor.send'] = #
# allowlist['torch.Tensor.serializable_wrapper_type'] = #
# allowlist['torch.Tensor.serialize'] = #
# allowlist['torch.Tensor.set_'] = #
# allowlist["torch.Tensor.shape"] = "torch.Size"
# allowlist['torch.Tensor.share_memory_'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
allowlist["torch.Tensor.short"] = "torch.Tensor"
allowlist["torch.Tensor.sigmoid"] = "torch.Tensor"
allowlist["torch.Tensor.sigmoid_"] = "torch.Tensor"
allowlist["torch.Tensor.sign"] = "torch.Tensor"
allowlist["torch.Tensor.sign_"] = "torch.Tensor"
allowlist["torch.Tensor.sin"] = "torch.Tensor"
allowlist["torch.Tensor.sin_"] = "torch.Tensor"
allowlist["torch.Tensor.sinh"] = "torch.Tensor"
allowlist["torch.Tensor.sinh_"] = "torch.Tensor"
# allowlist['torch.Tensor.size'] = #
# allowlist['torch.Tensor.slogdet'] = #
# allowlist['torch.Tensor.smm'] = #
# allowlist['torch.Tensor.softmax'] = #
# allowlist['torch.Tensor.solve'] = #
# allowlist['torch.Tensor.sort'] = #
# allowlist['torch.Tensor.sparse_dim'] = #
# allowlist['torch.Tensor.sparse_mask'] = #
# allowlist['torch.Tensor.sparse_resize_'] = #
# allowlist['torch.Tensor.sparse_resize_and_clear_'] = #
# allowlist['torch.Tensor.split'] = #
# allowlist['torch.Tensor.split_with_sizes'] = #
allowlist["torch.Tensor.sqrt"] = "torch.Tensor"
allowlist["torch.Tensor.sqrt_"] = "torch.Tensor"
allowlist["torch.Tensor.square"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.square_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.squeeze"] = "torch.Tensor"
allowlist["torch.Tensor.squeeze_"] = "torch.Tensor"
# allowlist['torch.Tensor.sspaddmm'] = #
allowlist["torch.Tensor.std"] = "torch.Tensor"
# allowlist['torch.Tensor.stft'] = #
# allowlist['torch.Tensor.storage'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.storage_offset'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.storage_type'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.stride'] = #
allowlist["torch.Tensor.sub"] = "torch.Tensor"
allowlist["torch.Tensor.sub_"] = "torch.Tensor"
allowlist["torch.Tensor.sum"] = "torch.Tensor"
# allowlist['torch.Tensor.sum_to_size'] = #
# allowlist['torch.Tensor.svd'] = #
# allowlist['torch.Tensor.symeig'] = #
allowlist["torch.Tensor.t"] = "torch.Tensor"
allowlist["torch.Tensor.t_"] = "torch.Tensor"
# allowlist['torch.Tensor.tag'] = #
allowlist["torch.Tensor.take"] = "torch.Tensor"
allowlist["torch.Tensor.tan"] = "torch.Tensor"
allowlist["torch.Tensor.tan_"] = "torch.Tensor"
allowlist["torch.Tensor.tanh"] = "torch.Tensor"
allowlist["torch.Tensor.tanh_"] = "torch.Tensor"
# allowlist['torch.Tensor.to'] = #
# allowlist['torch.Tensor.to_binary'] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.to_dense'] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.to_hex'] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.to_json'] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.to_mkldnn'] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.to_proto'] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.to_sparse'] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.tolist'] = #
# allowlist['torch.Tensor.topk'] = #
allowlist["torch.Tensor.trace"] = "torch.Tensor"
allowlist["torch.Tensor.transpose"] = "torch.Tensor"
allowlist["torch.Tensor.transpose_"] = "torch.Tensor"
# allowlist['torch.Tensor.triangular_solve'] = #
allowlist["torch.Tensor.tril"] = "torch.Tensor"
allowlist["torch.Tensor.tril_"] = "torch.Tensor"
allowlist["torch.Tensor.triu"] = "torch.Tensor"
allowlist["torch.Tensor.triu_"] = "torch.Tensor"
allowlist["torch.Tensor.true_divide"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.true_divide_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.Tensor.trunc"] = "torch.Tensor"
allowlist["torch.Tensor.trunc_"] = "torch.Tensor"
# allowlist['torch.Tensor.type'] =
# allowlist['torch.Tensor.type_as'] = #
# allowlist['torch.Tensor.unbind'] = #
# allowlist['torch.Tensor.unflatten'] = #
# allowlist["torch.Tensor.unfold"] = "torch.Tensor"
# https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.uniform_
# allowlist["torch.Tensor.uniform_"] = "torch.Tensor"
allowlist["torch.Tensor.unique"] = "torch.Tensor"
allowlist["torch.Tensor.unique_consecutive"] = "torch.Tensor"
allowlist["torch.Tensor.unsqueeze"] = "torch.Tensor"
allowlist["torch.Tensor.unsqueeze_"] = "torch.Tensor"
# allowlist['torch.Tensor.values'] = #
allowlist["torch.Tensor.var"] = "torch.Tensor"
# allowlist['torch.Tensor.view'] = #
# allowlist['torch.Tensor.view_as'] = #
# allowlist['torch.Tensor.where'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
allowlist["torch.Tensor.zero_"] = "torch.Tensor"


# SECTION - other classes and modules

allowlist["torch.zeros"] = "torch.Tensor"
allowlist["torch.ones"] = "torch.Tensor"


# SECTION - Parameter methods
# torch.nn.Parameter is a subclass of torch.Tensor
# However, we still need the constructor Class to be listed here. Everything else is
# automatically added in create_torch_ast function by doing:
# method = method.replace("torch.Tensor.", "torch.nn.Parameter.")
allowlist["torch.nn.Parameter"] = "torch.nn.Parameter"


# MNIST
# Misc
allowlist["torch.manual_seed"] = "torch.Generator"
allowlist["torch.Generator"] = "torch.Generator"
allowlist["torch.Generator.get_state"] = "torch.Tensor"
allowlist["torch.Generator.set_state"] = "torch.Generator"
allowlist["torch.device"] = "torch.device"
allowlist["torch.device.index"] = "syft.lib.python.Int"
allowlist["torch.device.type"] = "syft.lib.python.String"
allowlist["torch.cuda.is_available"] = "syft.lib.python.Bool"
allowlist["torch.random.initial_seed"] = "syft.lib.python.Int"

# Modules
allowlist["torch.nn.Module"] = "torch.nn.Module"
allowlist["torch.nn.Module.__call__"] = "torch.nn.Module"
allowlist["torch.nn.Module.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Module.train"] = "torch.nn.Module"

allowlist["torch.nn.Conv2d"] = "torch.nn.Conv2d"
allowlist["torch.nn.Conv2d.__call__"] = "torch.nn.Conv2d"
allowlist["torch.nn.Conv2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Conv2d.train"] = "torch.nn.Conv2d"

allowlist["torch.nn.Dropout2d"] = "torch.nn.Dropout2d"
allowlist["torch.nn.Dropout2d.__call__"] = "torch.nn.Dropout2d"
allowlist["torch.nn.Dropout2d.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Dropout2d.train"] = "torch.nn.Dropout2d"

allowlist["torch.nn.Linear"] = "torch.nn.Linear"
allowlist["torch.nn.Linear.__call__"] = "torch.nn.Linear"
allowlist["torch.nn.Linear.parameters"] = "syft.lib.python.List"
allowlist["torch.nn.Linear.train"] = "torch.nn.Linear"

# DataLoader
allowlist["torch.utils.data.DataLoader"] = "torch.utils.data.DataLoader"
# allowlist["torch.utils.data.DataLoader.dataset"] = "torchvision.datasets.VisionDataset"
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
# allowlist[
#     "torch.utils.data.dataloader._SingleProcessDataLoaderIter.__next__"
# ] = "torch.Tensor"
# allowlist[
#     "torch.utils.data.dataloader._SingleProcessDataLoaderIter.next"
# ] = "torch.Tensor"

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
allowlist["torch.optim.Adadelta.zero_grad"] = "syft.lib.python.SyNone"
allowlist["torch.optim.Adadelta.step"] = "syft.lib.python.SyNone"
