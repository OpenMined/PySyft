from typing import Dict

allowlist: Dict[str, str] = {}  # (path: str, return_type:type)

# SECTION - add the capital constructor
allowlist["torch.Tensor"] = "torch.Tensor"

# SECTION - Tensor methods which return a torch tensor object

# allowlist['torch.Tensor.T'] = #
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
allowlist["torch.Tensor.__div__"] = "torch.Tensor"
# allowlist['torch.Tensor.__doc__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
allowlist["torch.Tensor.__eq__"] = "torch.Tensor"
allowlist["torch.Tensor.__float__"] = "torch.Tensor"
allowlist["torch.Tensor.__floordiv__"] = "torch.Tensor"
# allowlist['torch.Tensor.__format__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
allowlist["torch.Tensor.__ge__"] = "torch.Tensor"
# allowlist['torch.Tensor.__getattribute__'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.__getitem__'] = #
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
allowlist["torch.Tensor.__rfloordiv__"] = "torch.Tensor"
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
# allowlist['torch.Tensor.abs_'] = #
allowlist["torch.Tensor.acos"] = "torch.Tensor"
# allowlist['torch.Tensor.acos_'] = #
allowlist["torch.Tensor.add"] = "torch.Tensor"
# allowlist['torch.Tensor.add_'] = #
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
# allowlist['torch.Tensor.asin_'] = #
allowlist["torch.Tensor.atan"] = "torch.Tensor"
allowlist["torch.Tensor.atan2"] = "torch.Tensor"
# allowlist['torch.Tensor.atan2_'] = #
# allowlist['torch.Tensor.atan_'] = #
# allowlist['torch.Tensor.backward'] = #
# allowlist['torch.Tensor.baddbmm'] = #
# allowlist['torch.Tensor.baddbmm_'] = #
# allowlist['torch.Tensor.bernoulli'] = #
# allowlist['torch.Tensor.bernoulli_'] = #
# allowlist['torch.Tensor.bfloat16'] = #
# allowlist['torch.Tensor.binary'] = #
# allowlist['torch.Tensor.bincount'] = #
allowlist["torch.Tensor.bitwise_and"] = "torch.Tensor"
# allowlist['torch.Tensor.bitwise_and_'] = #
allowlist["torch.Tensor.bitwise_not"] = "torch.Tensor"
# allowlist['torch.Tensor.bitwise_not_'] = #
allowlist["torch.Tensor.bitwise_or"] = "torch.Tensor"
# allowlist['torch.Tensor.bitwise_or_'] = #
allowlist["torch.Tensor.bitwise_xor"] = "torch.Tensor"
# allowlist['torch.Tensor.bitwise_xor_'] = #
# allowlist['torch.Tensor.bmm'] = #
# allowlist['torch.Tensor.bool'] = #
# allowlist['torch.Tensor.byte'] = #
# allowlist['torch.Tensor.cauchy_'] = #
allowlist["torch.Tensor.ceil"] = "torch.Tensor"
# allowlist['torch.Tensor.ceil_'] = #
allowlist["torch.Tensor.char"] = "torch.Tensor"
# allowlist['torch.Tensor.cholesky'] = #
# allowlist['torch.Tensor.cholesky_inverse'] = #
# allowlist['torch.Tensor.chunk'] = #
allowlist["torch.Tensor.clamp"] = "torch.Tensor"
# allowlist['torch.Tensor.clamp_'] = #
allowlist["torch.Tensor.clamp_max"] = "torch.Tensor"
# allowlist['torch.Tensor.clamp_max_'] = #
allowlist["torch.Tensor.clamp_min"] = "torch.Tensor"
# allowlist['torch.Tensor.clamp_min_'] = #
allowlist["torch.Tensor.clone"] = "torch.Tensor"
# allowlist['torch.Tensor.coalesce'] = #
# allowlist['torch.Tensor.conj'] = #
# allowlist['torch.Tensor.contiguous'] = #
# allowlist['torch.Tensor.copy_'] = #
allowlist["torch.Tensor.cos"] = "torch.Tensor"
# allowlist['torch.Tensor.cos_'] = #
allowlist["torch.Tensor.cosh"] = "torch.Tensor"
# allowlist['torch.Tensor.cosh_'] = #
allowlist["torch.Tensor.cpu"] = "torch.Tensor"
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
allowlist["torch.Tensor.diag"] = "torch.Tensor"
# allowlist['torch.Tensor.diag_embed'] = #
# allowlist['torch.Tensor.diagflat'] = #
allowlist["torch.Tensor.diagonal"] = "torch.Tensor"
# allowlist['torch.Tensor.digamma'] = #
# allowlist['torch.Tensor.digamma_'] = #
# allowlist['torch.Tensor.dim'] = #
# allowlist['torch.Tensor.dist'] = #
allowlist["torch.Tensor.div"] = "torch.Tensor"
# allowlist['torch.Tensor.div_'] = #
allowlist["torch.Tensor.dot"] = "torch.Tensor"
allowlist["torch.Tensor.double"] = "torch.Tensor"
# allowlist['torch.Tensor.dtype'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST YET - talk to TRASK
# allowlist['torch.Tensor.eig'] = #
# allowlist['torch.Tensor.element_size'] = #
allowlist["torch.Tensor.eq"] = "torch.Tensor"
# allowlist['torch.Tensor.eq_'] = #
# allowlist['torch.Tensor.equal'] = #
allowlist["torch.Tensor.erf"] = "torch.Tensor"
# allowlist['torch.Tensor.erf_'] = #
allowlist["torch.Tensor.erfc"] = "torch.Tensor"
# allowlist['torch.Tensor.erfc_'] = #
allowlist["torch.Tensor.erfinv"] = "torch.Tensor"
# allowlist['torch.Tensor.erfinv_'] = #
allowlist["torch.Tensor.exp"] = "torch.Tensor"
# allowlist['torch.Tensor.exp_'] = #
# allowlist['torch.Tensor.expand'] = #
# allowlist['torch.Tensor.expand_as'] = #
allowlist["torch.Tensor.expm1"] = "torch.Tensor"
# allowlist['torch.Tensor.expm1_'] = #
# allowlist['torch.Tensor.exponential_'] = #
# allowlist['torch.Tensor.fft'] = #
# allowlist['torch.Tensor.fill_'] = #
# allowlist['torch.Tensor.fill_diagonal_'] = #
allowlist["torch.Tensor.flatten"] = "torch.Tensor"
# allowlist['torch.Tensor.flip'] = #
allowlist["torch.Tensor.float"] = "torch.Tensor"
allowlist["torch.Tensor.floor"] = "torch.Tensor"
# allowlist['torch.Tensor.floor_'] = #
allowlist["torch.Tensor.floor_divide"] = "torch.Tensor"
# allowlist['torch.Tensor.floor_divide_'] = #
# allowlist['torch.Tensor.fmod'] = #
# allowlist['torch.Tensor.fmod_'] = #
allowlist["torch.Tensor.frac"] = "torch.Tensor"
# allowlist['torch.Tensor.frac_'] = #
# allowlist['torch.Tensor.gather'] = #
allowlist["torch.Tensor.ge"] = "torch.Tensor"
# allowlist['torch.Tensor.ge_'] = #
# allowlist['torch.Tensor.geometric_'] = #
# allowlist['torch.Tensor.geqrf'] = #
allowlist["torch.Tensor.ger"] = "torch.Tensor"
# allowlist['torch.Tensor.get_device'] = #
# allowlist['torch.Tensor.grad'] = #
# allowlist['torch.Tensor.grad_fn'] = #
allowlist["torch.Tensor.gt"] = "torch.Tensor"
# allowlist['torch.Tensor.gt_'] = #
allowlist["torch.Tensor.half"] = "torch.Tensor"
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
allowlist["torch.Tensor.int"] = "torch.Tensor"
# allowlist['torch.Tensor.int_repr'] = #
# allowlist['torch.Tensor.inverse'] = #
# allowlist['torch.Tensor.irfft'] = #
# allowlist['torch.Tensor.is_coalesced'] = #
# allowlist['torch.Tensor.is_complex'] = #
# allowlist['torch.Tensor.is_contiguous'] = #
# allowlist['torch.Tensor.is_cuda'] = #
# allowlist['torch.Tensor.is_distributed'] = #
allowlist["torch.Tensor.is_floating_point"] = "bool"
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
allowlist["torch.Tensor.le"] = "torch.Tensor"
# allowlist['torch.Tensor.le_'] = #
# allowlist['torch.Tensor.lerp'] = #
# allowlist['torch.Tensor.lerp_'] = #
allowlist["torch.Tensor.lgamma"] = "torch.Tensor"
# allowlist['torch.Tensor.lgamma_'] = #
allowlist["torch.Tensor.log"] = "torch.Tensor"
allowlist["torch.Tensor.log10"] = "torch.Tensor"
# allowlist['torch.Tensor.log10_'] = #
allowlist["torch.Tensor.log1p"] = "torch.Tensor"
# allowlist['torch.Tensor.log1p_'] = #
allowlist["torch.Tensor.log2"] = "torch.Tensor"
# allowlist['torch.Tensor.log2_'] = #
# allowlist['torch.Tensor.log_'] = #
# allowlist['torch.Tensor.log_normal_'] = #
# allowlist['torch.Tensor.log_softmax'] = #

# allowlist['torch.Tensor.logdet'] = #
allowlist["torch.Tensor.logical_and"] = "torch.Tensor"
# allowlist['torch.Tensor.logical_and_'] = #
allowlist["torch.Tensor.logical_not"] = "torch.Tensor"
# allowlist['torch.Tensor.logical_not_'] = #
allowlist["torch.Tensor.logical_or"] = "torch.Tensor"
# allowlist['torch.Tensor.logical_or_'] = #
allowlist["torch.Tensor.logical_xor"] = "torch.Tensor"
# allowlist['torch.Tensor.logical_xor_'] = #
# allowlist['torch.Tensor.logsumexp'] = #
allowlist["torch.Tensor.long"] = "torch.Tensor"
# allowlist['torch.Tensor.lstsq'] = #
allowlist["torch.Tensor.lt"] = "torch.Tensor"
# allowlist['torch.Tensor.lt_'] = #
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
allowlist["torch.Tensor.max"] = "torch.Tensor"
allowlist["torch.Tensor.mean"] = "torch.Tensor"
allowlist["torch.Tensor.median"] = "torch.Tensor"
allowlist["torch.Tensor.min"] = "torch.Tensor"
allowlist["torch.Tensor.mm"] = "torch.Tensor"
# allowlist['torch.Tensor.mode'] = #
allowlist["torch.Tensor.mul"] = "torch.Tensor"
# allowlist['torch.Tensor.mul_'] = #
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
allowlist["torch.Tensor.ne"] = "torch.Tensor"
# allowlist['torch.Tensor.ne_'] = #
allowlist["torch.Tensor.neg"] = "torch.Tensor"
# allowlist['torch.Tensor.neg_'] = #
# allowlist['torch.Tensor.nelement'] = #
allowlist["torch.Tensor.new"] = "torch.Tensor"
# allowlist['torch.Tensor.new_empty'] = #
# allowlist['torch.Tensor.new_full'] = #
# allowlist['torch.Tensor.new_ones'] = #
# allowlist['torch.Tensor.new_tensor'] = #
# allowlist['torch.Tensor.new_zeros'] = #
allowlist["torch.Tensor.nonzero"] = "torch.Tensor"
allowlist["torch.Tensor.norm"] = "torch.Tensor"
# allowlist['torch.Tensor.normal_'] = #
# allowlist['torch.Tensor.numel'] = #
# allowlist['torch.Tensor.numpy'] = #
allowlist["torch.Tensor.orgqr"] = "torch.Tensor"
# allowlist['torch.Tensor.ormqr') # requires two tensors as argument] = #
# allowlist['torch.Tensor.output_nr'] = #
# allowlist['torch.Tensor.permute'] = #
# allowlist['torch.Tensor.pin_memory'] = #
allowlist["torch.Tensor.pinverse"] = "torch.Tensor"
# allowlist['torch.Tensor.polygamma'] = #
# allowlist['torch.Tensor.polygamma_'] = #
allowlist["torch.Tensor.pow"] = "torch.Tensor"
# allowlist['torch.Tensor.pow_'] = #
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
allowlist["torch.Tensor.reciprocal"] = "torch.Tensor"
# allowlist['torch.Tensor.reciprocal_'] = #
# allowlist['torch.Tensor.record_stream'] =SECURITY WARNING: DO NOT ADD TO ALLOW LIST#
# allowlist['torch.Tensor.refine_names'] = #
# allowlist['torch.Tensor.register_hook'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.reinforce'] = #
allowlist["torch.Tensor.relu"] = "torch.Tensor"
# allowlist['torch.Tensor.relu_'] = #
allowlist["torch.Tensor.remainder"] = "torch.Tensor"
# allowlist['torch.Tensor.remainder_'] = #
# allowlist['torch.Tensor.rename'] = #
# allowlist['torch.Tensor.rename_'] = #
# allowlist['torch.Tensor.renorm'] = #
# allowlist['torch.Tensor.renorm_'] = #
# allowlist['torch.Tensor.repeat'] = #
# allowlist['torch.Tensor.repeat_interleave'] = #
# allowlist['torch.Tensor.requires_grad'] = #
# allowlist['torch.Tensor.requires_grad_'] = #
# allowlist['torch.Tensor.reshape'] = #
allowlist["torch.Tensor.reshape_as"] = "torch.Tensor"
# allowlist['torch.Tensor.resize'] = #
# allowlist['torch.Tensor.resize_'] = #
allowlist["torch.Tensor.resize_as"] = "torch.Tensor"
# allowlist['torch.Tensor.resize_as_'] = #
# allowlist['torch.Tensor.retain_grad'] = #
# allowlist['torch.Tensor.rfft'] = #
# allowlist['torch.Tensor.roll'] = #
allowlist["torch.Tensor.rot90"] = "torch.Tensor"
allowlist["torch.Tensor.round"] = "torch.Tensor"
# allowlist['torch.Tensor.round_'] = #
allowlist["torch.Tensor.rsqrt"] = "torch.Tensor"
# allowlist['torch.Tensor.rsqrt_'] = #
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
allowlist["torch.Tensor.short"] = "torch.Tensor"
allowlist["torch.Tensor.sigmoid"] = "torch.Tensor"
# allowlist['torch.Tensor.sigmoid_'] = #
allowlist["torch.Tensor.sign"] = "torch.Tensor"
# allowlist['torch.Tensor.sign_'] = #
allowlist["torch.Tensor.sin"] = "torch.Tensor"
# allowlist['torch.Tensor.sin_'] = #
allowlist["torch.Tensor.sinh"] = "torch.Tensor"
# allowlist['torch.Tensor.sinh_'] = #
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
# allowlist['torch.Tensor.sqrt_'] = #
allowlist["torch.Tensor.square"] = "torch.Tensor"
# allowlist['torch.Tensor.square_'] = #
allowlist["torch.Tensor.squeeze"] = "torch.Tensor"
# allowlist['torch.Tensor.squeeze_'] = #
# allowlist['torch.Tensor.sspaddmm'] = #
allowlist["torch.Tensor.std"] = "torch.Tensor"
# allowlist['torch.Tensor.stft'] = #
# allowlist['torch.Tensor.storage'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.storage_offset'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.storage_type'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.stride'] = #
allowlist["torch.Tensor.sub"] = "torch.Tensor"
# allowlist['torch.Tensor.sub_'] = #
allowlist["torch.Tensor.sum"] = "torch.Tensor"
# allowlist['torch.Tensor.sum_to_size'] = #
# allowlist['torch.Tensor.svd'] = #
# allowlist['torch.Tensor.symeig'] = #
allowlist["torch.Tensor.t"] = "torch.Tensor"
# allowlist['torch.Tensor.t_'] = #
# allowlist['torch.Tensor.tag'] = #
allowlist["torch.Tensor.take"] = "torch.Tensor"
allowlist["torch.Tensor.tan"] = "torch.Tensor"
# allowlist['torch.Tensor.tan_'] = #
allowlist["torch.Tensor.tanh"] = "torch.Tensor"
# allowlist['torch.Tensor.tanh_'] = #
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
# allowlist['torch.Tensor.transpose_'] = #
# allowlist['torch.Tensor.triangular_solve'] = #
allowlist["torch.Tensor.tril"] = "torch.Tensor"
# allowlist['torch.Tensor.tril_'] = #
allowlist["torch.Tensor.triu"] = "torch.Tensor"
# allowlist['torch.Tensor.triu_'] = #
allowlist["torch.Tensor.true_divide"] = "torch.Tensor"
allowlist["torch.Tensor.true_divide_"] = "torch.Tensor"
allowlist["torch.Tensor.trunc"] = "torch.Tensor"
# allowlist['torch.Tensor.trunc_'] = #
# allowlist['torch.Tensor.type'] =
# allowlist['torch.Tensor.type_as'] = #
# allowlist['torch.Tensor.unbind'] = #
# allowlist['torch.Tensor.unflatten'] = #
# allowlist['torch.Tensor.unfold'] = #
# allowlist['torch.Tensor.uniform_'] = #
allowlist["torch.Tensor.unique"] = "torch.Tensor"
allowlist["torch.Tensor.unique_consecutive"] = "torch.Tensor"
# allowlist['torch.Tensor.unsqueeze'] = #
allowlist["torch.Tensor.unsqueeze_"] = "torch.Tensor"
# allowlist['torch.Tensor.values'] = #
allowlist["torch.Tensor.var"] = "torch.Tensor"
# allowlist['torch.Tensor.view'] = #
# allowlist['torch.Tensor.view_as'] = #
# allowlist['torch.Tensor.where'] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist['torch.Tensor.zero_'] = #


# SECTION - other classes and modules

allowlist["torch.zeros"] = "torch.Tensor"
allowlist["torch.ones"] = "torch.Tensor"
allowlist["torch.nn.Linear"] = "torch.nn.Linear"
# allowlist.add("torch.nn.Linear.parameters")
allowlist["torch.nn.parameter.Parameter"] = "torch.nn.parameter.Parameter"
allowlist["torch.nn.parameter.Parameter.__add__"] = "torch.nn.parameter.Parameter"
