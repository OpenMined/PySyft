from typing import Union
from typing import Dict

allowlist: Dict[str, Union[str, Dict[str, str]]] = {}  # (path: str, return_type:type)

# SECTION - add the capital constructor
allowlist["torch.nn.Parameter"] = "torch.nn.Parameter"

# SECTION - Parameter methods which return a torch tensor object

# TODO: T is a property and requires detecting properties and supporting them in
# run_class_method_action.py
# allowlist['torch.Tensor.T'] = #
allowlist["torch.nn.Parameter.t"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__abs__"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__add__"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__and__"] = "torch.Tensor"
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
allowlist["torch.nn.Parameter.__eq__"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__float__"] = "torch.Tensor"
allowlist[
    "torch.Tensor.__floordiv__"
] = {  # exists in 1.4.0 but causes fatal exception on non floats
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
# allowlist['torch.Tensor.__format__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
allowlist["torch.nn.Parameter.__ge__"] = "torch.Tensor"
# allowlist['torch.Tensor.__getattribute__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__getitem__'] = #
allowlist["torch.nn.Parameter.__gt__"] = "torch.Tensor"
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
allowlist["torch.nn.Parameter.__int__"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__invert__"] = "torch.Tensor"
# allowlist['torch.Tensor.__ior__'] = #
# allowlist['torch.Tensor.__ipow__'] = #
# allowlist['torch.Tensor.__irshift__'] = #
# allowlist['torch.Tensor.__isub__'] = #
# allowlist['torch.Tensor.__iter__'] = #
# allowlist['torch.Tensor.__itruediv__'] = #
# allowlist['torch.Tensor.__ixor__'] = #
allowlist["torch.nn.Parameter.__le__"] = "torch.Tensor"
# allowlist['torch.Tensor.__len__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST YET - talk to TRASK
allowlist["torch.nn.Parameter.__long__"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__lshift__"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__lt__"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__matmul__"] = "torch.Tensor"
# allowlist['torch.Tensor.__mod__'] = #
# allowlist['torch.Tensor.__module__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
allowlist["torch.nn.Parameter.__mul__"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__ne__"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__neg__"] = "torch.Tensor"
# allowlist['torch.Tensor.__new__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__nonzero__'] = #
allowlist["torch.nn.Parameter.__or__"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__pow__"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__radd__"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__rdiv__"] = "torch.Tensor"
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
allowlist["torch.nn.Parameter.__rmul__"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__rpow__"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__rshift__"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__rsub__"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__rtruediv__"] = "torch.Tensor"
# allowlist['torch.Tensor.__setattr__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__setitem__'] = #
# allowlist['torch.Tensor.__setstate__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__sizeof__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST YET - talk to TRASK
# allowlist['torch.Tensor.__str__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
allowlist["torch.nn.Parameter.__sub__"] = "torch.Tensor"
# allowlist['torch.Tensor.__subclasshook__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__truediv__'] = #
# allowlist['torch.Tensor.__weakref__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
allowlist["torch.nn.Parameter.__xor__"] = "torch.Tensor"
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
allowlist["torch.nn.Parameter.abs"] = "torch.Tensor"
allowlist["torch.nn.Parameter.abs_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.acos"] = "torch.Tensor"
allowlist["torch.nn.Parameter.acos_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.add"] = "torch.Tensor"
allowlist["torch.nn.Parameter.add_"] = "torch.Tensor"
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
allowlist["torch.nn.Parameter.asin"] = "torch.Tensor"
allowlist["torch.nn.Parameter.asin_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.atan"] = "torch.Tensor"
allowlist["torch.nn.Parameter.atan_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.atan2"] = "torch.Tensor"
allowlist["torch.nn.Parameter.atan2_"] = "torch.Tensor"
# allowlist['torch.Tensor.backward'] = #
# allowlist['torch.Tensor.baddbmm'] = #
# allowlist['torch.Tensor.baddbmm_'] = #
# allowlist['torch.Tensor.bernoulli'] = #
# allowlist['torch.Tensor.bernoulli_'] = #
# allowlist['torch.Tensor.bfloat16'] = #
# allowlist['torch.Tensor.binary'] = #
# allowlist['torch.Tensor.bincount'] = #
allowlist["torch.nn.Parameter.bitwise_and"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Parameter.bitwise_and_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Parameter.bitwise_not"] = "torch.Tensor"
allowlist["torch.nn.Parameter.bitwise_not_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.bitwise_or"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Parameter.bitwise_or_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Parameter.bitwise_xor"] = "torch.Tensor"
allowlist["torch.nn.Parameter.bitwise_xor_"] = "torch.Tensor"
# allowlist['torch.Tensor.bmm'] = #
# allowlist['torch.Tensor.bool'] = #
# allowlist['torch.Tensor.byte'] = #
# allowlist['torch.Tensor.cauchy_'] = #
allowlist["torch.nn.Parameter.ceil"] = "torch.Tensor"
allowlist["torch.nn.Parameter.ceil_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.char"] = "torch.Tensor"
# allowlist['torch.Tensor.cholesky'] = #
# allowlist['torch.Tensor.cholesky_inverse'] = #
# allowlist['torch.Tensor.chunk'] = #
allowlist["torch.nn.Parameter.clamp"] = "torch.Tensor"
allowlist["torch.nn.Parameter.clamp_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.clamp_max"] = "torch.Tensor"
allowlist["torch.nn.Parameter.clamp_max_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.clamp_min"] = "torch.Tensor"
allowlist["torch.nn.Parameter.clamp_min_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.clone"] = "torch.Tensor"
# allowlist['torch.Tensor.coalesce'] = #
# allowlist['torch.Tensor.conj'] = #
# allowlist['torch.Tensor.contiguous'] = #
allowlist["torch.nn.Parameter.copy_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.cos"] = "torch.Tensor"
allowlist["torch.nn.Parameter.cos_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.cosh"] = "torch.Tensor"
allowlist["torch.nn.Parameter.cosh_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.cpu"] = "torch.Tensor"
# allowlist['torch.Tensor.cross'] = #
# allowlist['torch.Tensor.cuda'] = #
# allowlist['torch.Tensor.cummax'] = #
# allowlist['torch.Tensor.cummin'] = #
# allowlist['torch.Tensor.cumprod'] = #
# allowlist['torch.Tensor.cumsum'] = #
# allowlist['torch.Tensor.data'] = #
# allowlist['torch.Tensor.data_ptr'] = #
# allowlist['torch.Tensor.dense_dim'] = #
# allowlist['torch.Tensor.dequantize'] = #
# allowlist['torch.Tensor.describe'] = #
# allowlist['torch.Tensor.det'] = #
# allowlist['torch.Tensor.detach'] = #
# allowlist['torch.Tensor.detach_'] = #
# allowlist['torch.Tensor.device'] = #
allowlist["torch.nn.Parameter.diag"] = "torch.Tensor"
# allowlist['torch.Tensor.diag_embed'] = #
# allowlist['torch.Tensor.diagflat'] = #
allowlist["torch.nn.Parameter.diagonal"] = "torch.Tensor"
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
allowlist["torch.nn.Parameter.dot"] = "torch.Tensor"
allowlist["torch.nn.Parameter.double"] = "torch.Tensor"
# allowlist['torch.Tensor.dtype'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST YET - talk to TRASK
# allowlist['torch.Tensor.eig'] = #
# allowlist['torch.Tensor.element_size'] = #
allowlist["torch.nn.Parameter.eq"] = "torch.Tensor"
allowlist["torch.nn.Parameter.eq_"] = "torch.Tensor"
# allowlist['torch.Tensor.equal'] = #
allowlist["torch.nn.Parameter.erf"] = "torch.Tensor"
allowlist["torch.nn.Parameter.erf_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.erfc"] = "torch.Tensor"
allowlist["torch.nn.Parameter.erfc_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.erfinv"] = "torch.Tensor"
allowlist["torch.nn.Parameter.erfinv_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.exp"] = "torch.Tensor"
allowlist["torch.nn.Parameter.exp_"] = "torch.Tensor"
# allowlist['torch.Tensor.expand'] = #
# allowlist['torch.Tensor.expand_as'] = #
allowlist["torch.nn.Parameter.expm1"] = "torch.Tensor"
allowlist["torch.nn.Parameter.expm1_"] = "torch.Tensor"
# allowlist['torch.Tensor.exponential_'] = #
# allowlist['torch.Tensor.fft'] = #
# allowlist['torch.Tensor.fill_'] = #
# allowlist['torch.Tensor.fill_diagonal_'] = #
allowlist["torch.nn.Parameter.flatten"] = "torch.Tensor"
# allowlist['torch.Tensor.flip'] = #
allowlist["torch.nn.Parameter.float"] = "torch.Tensor"
allowlist["torch.nn.Parameter.floor"] = "torch.Tensor"
allowlist["torch.nn.Parameter.floor_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.floor_divide"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Parameter.floor_divide_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
# allowlist['torch.Tensor.fmod'] = #
# allowlist['torch.Tensor.fmod_'] = #
allowlist["torch.nn.Parameter.frac"] = "torch.Tensor"
allowlist["torch.nn.Parameter.frac_"] = "torch.Tensor"
# allowlist['torch.Tensor.gather'] = #
allowlist["torch.nn.Parameter.ge"] = "torch.Tensor"
allowlist["torch.nn.Parameter.ge_"] = "torch.Tensor"
# allowlist['torch.Tensor.geometric_'] = #
# allowlist['torch.Tensor.geqrf'] = #
allowlist["torch.nn.Parameter.ger"] = "torch.Tensor"
# allowlist['torch.Tensor.get_device'] = #
# allowlist['torch.Tensor.grad'] = #
# allowlist['torch.Tensor.grad_fn'] = #
allowlist["torch.nn.Parameter.gt"] = "torch.Tensor"
allowlist["torch.nn.Parameter.gt_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.half"] = "torch.Tensor"
# allowlist['torch.Tensor.hardshrink'] = #
# allowlist['torch.Tensor.has_names'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.hex'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.histc'] = #
# allowlist['torch.Tensor.id'] = #
# allowlist['torch.Tensor.ifft'] = #
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
allowlist["torch.nn.Parameter.int"] = "torch.Tensor"
# allowlist['torch.Tensor.int_repr'] = #
# allowlist['torch.Tensor.inverse'] = #
# allowlist['torch.Tensor.irfft'] = #
# allowlist['torch.Tensor.is_coalesced'] = #
# allowlist['torch.Tensor.is_complex'] = #
# allowlist['torch.Tensor.is_contiguous'] = #
# allowlist['torch.Tensor.is_cuda'] = #
# allowlist['torch.Tensor.is_distributed'] = #
# allowlist["torch.nn.Parameter.is_floating_point"] = "builtins.bool"
# allowlist['torch.Tensor.is_leaf'] = #
# allowlist['torch.Tensor.is_mkldnn'] = #
# allowlist['torch.Tensor.is_nonzero'] = #
# allowlist['torch.Tensor.is_pinned'] = #
# allowlist['torch.Tensor.is_quantized'] = #
# allowlist['torch.Tensor.is_same_size'] = #
# allowlist['torch.Tensor.is_set_to'] = #
# allowlist['torch.Tensor.is_shared'] = #
# allowlist['torch.Tensor.is_signed'] = #
# allowlist['torch.Tensor.is_sparse'] = #
# allowlist['torch.Tensor.isclose'] = #
# allowlist['torch.Tensor.item'] = #
# allowlist['torch.Tensor.json'] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.kthvalue'] = #
# allowlist['torch.Tensor.layout'] = #
allowlist["torch.nn.Parameter.le"] = "torch.Tensor"
allowlist["torch.nn.Parameter.le_"] = "torch.Tensor"
# allowlist['torch.Tensor.lerp'] = #
# allowlist['torch.Tensor.lerp_'] = #
allowlist["torch.nn.Parameter.lgamma"] = "torch.Tensor"
allowlist["torch.nn.Parameter.lgamma_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.log"] = "torch.Tensor"
allowlist["torch.nn.Parameter.log_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.log10"] = "torch.Tensor"
allowlist["torch.nn.Parameter.log10_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.log1p"] = "torch.Tensor"
allowlist["torch.nn.Parameter.log1p_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.log2"] = "torch.Tensor"
allowlist["torch.nn.Parameter.log2_"] = "torch.Tensor"
# allowlist['torch.Tensor.log_normal_'] = #
# allowlist['torch.Tensor.log_softmax'] = #

# allowlist['torch.Tensor.logdet'] = #
allowlist["torch.nn.Parameter.logical_and"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Parameter.logical_and_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Parameter.logical_not"] = "torch.Tensor"
allowlist["torch.nn.Parameter.logical_not_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.logical_or"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Parameter.logical_or_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Parameter.logical_xor"] = "torch.Tensor"
allowlist["torch.nn.Parameter.logical_xor_"] = "torch.Tensor"
# allowlist['torch.Tensor.logsumexp'] = #
allowlist["torch.nn.Parameter.long"] = "torch.Tensor"
# allowlist['torch.Tensor.lstsq'] = #
allowlist["torch.nn.Parameter.lt"] = "torch.Tensor"
allowlist["torch.nn.Parameter.lt_"] = "torch.Tensor"
# allowlist['torch.Tensor.lu'] = #
# allowlist['torch.Tensor.lu_solve'] = #
# allowlist['torch.Tensor.map2_'] = #
# allowlist['torch.Tensor.map_'] = #
# allowlist['torch.Tensor.masked_fill'] = #
# allowlist['torch.Tensor.masked_fill_'] = #
# allowlist['torch.Tensor.masked_scatter'] = #
# allowlist['torch.Tensor.masked_scatter_'] = #
# allowlist['torch.Tensor.masked_select'] = #
allowlist["torch.nn.Parameter.matmul"] = "torch.Tensor"
# allowlist['torch.Tensor.matrix_power'] = #
allowlist["torch.nn.Parameter.max"] = "torch.Tensor"
allowlist["torch.nn.Parameter.mean"] = "torch.Tensor"
allowlist["torch.nn.Parameter.median"] = "torch.Tensor"
allowlist["torch.nn.Parameter.min"] = "torch.Tensor"
allowlist["torch.nn.Parameter.mm"] = "torch.Tensor"
# allowlist['torch.Tensor.mode'] = #
allowlist["torch.nn.Parameter.mul"] = "torch.Tensor"
allowlist["torch.nn.Parameter.mul_"] = "torch.Tensor"
# allowlist['torch.Tensor.multinomial'] = #
# allowlist['torch.Tensor.mv'] = #
# allowlist['torch.Tensor.mvlgamma'] = #
# allowlist['torch.Tensor.mvlgamma_'] = #
# allowlist['torch.Tensor.name'] = #
# allowlist['torch.Tensor.names'] = #
# allowlist['torch.Tensor.narrow'] = #
# allowlist['torch.Tensor.narrow_copy'] = #
# allowlist['torch.Tensor.ndim'] = #
# allowlist['torch.Tensor.ndimension'] = #
allowlist["torch.nn.Parameter.ne"] = "torch.Tensor"
allowlist["torch.nn.Parameter.ne_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.neg"] = "torch.Tensor"
allowlist["torch.nn.Parameter.neg_"] = "torch.Tensor"
# allowlist['torch.Tensor.nelement'] = #
allowlist["torch.nn.Parameter.new"] = "torch.Tensor"
# allowlist['torch.Tensor.new_empty'] = #
# allowlist['torch.Tensor.new_full'] = #
# allowlist['torch.Tensor.new_ones'] = #
# allowlist['torch.Tensor.new_tensor'] = #
# allowlist['torch.Tensor.new_zeros'] = #
allowlist["torch.nn.Parameter.nonzero"] = "torch.Tensor"
allowlist["torch.nn.Parameter.norm"] = "torch.Tensor"
# allowlist['torch.Tensor.normal_'] = #
# allowlist['torch.Tensor.numel'] = #
# allowlist['torch.Tensor.numpy'] = #
allowlist["torch.nn.Parameter.orgqr"] = "torch.Tensor"
# allowlist['torch.Tensor.ormqr') # requires two tensors as argument] = #
# allowlist['torch.Tensor.output_nr'] = #
# allowlist['torch.Tensor.permute'] = #
# allowlist['torch.Tensor.pin_memory'] = #
allowlist["torch.nn.Parameter.pinverse"] = "torch.Tensor"
# allowlist['torch.Tensor.polygamma'] = #
# allowlist['torch.Tensor.polygamma_'] = #
allowlist["torch.nn.Parameter.pow"] = "torch.Tensor"
allowlist["torch.nn.Parameter.pow_"] = "torch.Tensor"
# allowlist['torch.Tensor.prelu'] = #
allowlist["torch.nn.Parameter.prod"] = "torch.Tensor"
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
allowlist["torch.nn.Parameter.reciprocal"] = "torch.Tensor"
allowlist["torch.nn.Parameter.reciprocal_"] = "torch.Tensor"
# allowlist['torch.Tensor.record_stream'] =SECURITY WARNING: DO NOT ADD TO ALLOW LIST#
# allowlist['torch.Tensor.refine_names'] = #
# allowlist['torch.Tensor.register_hook'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.reinforce'] = #
allowlist["torch.nn.Parameter.relu"] = "torch.Tensor"
allowlist["torch.nn.Parameter.relu_"] = "torch.Tensor"
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
# allowlist['torch.Tensor.requires_grad'] = #
# allowlist['torch.Tensor.requires_grad_'] = #
# allowlist['torch.Tensor.reshape'] = #
allowlist["torch.nn.Parameter.reshape_as"] = "torch.Tensor"
# allowlist['torch.Tensor.resize'] = #
# allowlist['torch.Tensor.resize_'] = #
allowlist["torch.nn.Parameter.resize_as"] = "torch.Tensor"
allowlist["torch.nn.Parameter.resize_as_"] = "torch.Tensor"
# allowlist['torch.Tensor.retain_grad'] = #
# allowlist['torch.Tensor.rfft'] = #
# allowlist['torch.Tensor.roll'] = #
allowlist["torch.nn.Parameter.rot90"] = "torch.Tensor"
allowlist["torch.nn.Parameter.round"] = "torch.Tensor"
allowlist["torch.nn.Parameter.round_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.rsqrt"] = "torch.Tensor"
allowlist["torch.nn.Parameter.rsqrt_"] = "torch.Tensor"
# allowlist['torch.Tensor.scatter'] = #
# allowlist['torch.Tensor.scatter_'] = #
# allowlist['torch.Tensor.scatter_add'] = #
# allowlist['torch.Tensor.scatter_add_'] = #
# allowlist['torch.Tensor.select'] = #
# allowlist['torch.Tensor.send'] = #
# allowlist['torch.Tensor.serializable_wrapper_type'] = #
# allowlist['torch.Tensor.serialize'] = #
# allowlist['torch.Tensor.set_'] = #
# allowlist['torch.Tensor.shape'] = #
# allowlist['torch.Tensor.share_memory_'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
allowlist["torch.nn.Parameter.short"] = "torch.Tensor"
allowlist["torch.nn.Parameter.sigmoid"] = "torch.Tensor"
allowlist["torch.nn.Parameter.sigmoid_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.sign"] = "torch.Tensor"
allowlist["torch.nn.Parameter.sign_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.sin"] = "torch.Tensor"
allowlist["torch.nn.Parameter.sin_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.sinh"] = "torch.Tensor"
allowlist["torch.nn.Parameter.sinh_"] = "torch.Tensor"
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
allowlist["torch.nn.Parameter.sqrt"] = "torch.Tensor"
allowlist["torch.nn.Parameter.sqrt_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.square"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Parameter.square_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Parameter.squeeze"] = "torch.Tensor"
allowlist["torch.nn.Parameter.squeeze_"] = "torch.Tensor"
# allowlist['torch.Tensor.sspaddmm'] = #
allowlist["torch.nn.Parameter.std"] = "torch.Tensor"
# allowlist['torch.Tensor.stft'] = #
# allowlist['torch.Tensor.storage'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.storage_offset'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.storage_type'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.stride'] = #
allowlist["torch.nn.Parameter.sub"] = "torch.Tensor"
allowlist["torch.nn.Parameter.sub_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.sum"] = "torch.Tensor"
# allowlist['torch.Tensor.sum_to_size'] = #
# allowlist['torch.Tensor.svd'] = #
# allowlist['torch.Tensor.symeig'] = #
allowlist["torch.nn.Parameter.t"] = "torch.Tensor"
allowlist["torch.nn.Parameter.t_"] = "torch.Tensor"
# allowlist['torch.Tensor.tag'] = #
allowlist["torch.nn.Parameter.take"] = "torch.Tensor"
allowlist["torch.nn.Parameter.tan"] = "torch.Tensor"
allowlist["torch.nn.Parameter.tan_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.tanh"] = "torch.Tensor"
allowlist["torch.nn.Parameter.tanh_"] = "torch.Tensor"
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
allowlist["torch.nn.Parameter.trace"] = "torch.Tensor"
allowlist["torch.nn.Parameter.transpose"] = "torch.Tensor"
allowlist["torch.nn.Parameter.transpose_"] = "torch.Tensor"
# allowlist['torch.Tensor.triangular_solve'] = #
allowlist["torch.nn.Parameter.tril"] = "torch.Tensor"
allowlist["torch.nn.Parameter.tril_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.triu"] = "torch.Tensor"
allowlist["torch.nn.Parameter.triu_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.true_divide"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Parameter.true_divide_"] = {
    "return_type": "torch.Tensor",
    "min_version": "1.5.0",
}
allowlist["torch.nn.Parameter.trunc"] = "torch.Tensor"
allowlist["torch.nn.Parameter.trunc_"] = "torch.Tensor"
# allowlist['torch.Tensor.type'] =
# allowlist['torch.Tensor.type_as'] = #
# allowlist['torch.Tensor.unbind'] = #
# allowlist['torch.Tensor.unflatten'] = #
# allowlist['torch.Tensor.unfold'] = #
# https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.uniform_
# allowlist["torch.nn.Parameter.uniform_"] = "torch.Tensor"
allowlist["torch.nn.Parameter.unique"] = "torch.Tensor"
allowlist["torch.nn.Parameter.unique_consecutive"] = "torch.Tensor"
allowlist["torch.nn.Parameter.unsqueeze"] = "torch.Tensor"
allowlist["torch.nn.Parameter.unsqueeze_"] = "torch.Tensor"
# allowlist['torch.Tensor.values'] = #
allowlist["torch.nn.Parameter.var"] = "torch.Tensor"
# allowlist['torch.Tensor.view'] = #
# allowlist['torch.Tensor.view_as'] = #
# allowlist['torch.Tensor.where'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
allowlist["torch.nn.Parameter.zero_"] = "torch.Tensor"


# SECTION - other classes and modules

allowlist["torch.zeros"] = "torch.Tensor"
allowlist["torch.ones"] = "torch.Tensor"
allowlist["torch.nn.Linear"] = "torch.nn.Linear"
