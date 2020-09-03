from typing import Union
from typing import Dict

allowlist: Dict[str, Union[str, Dict[str, str]]] = {}  # (path: str, return_type:type)

# SECTION - add the capital constructor
allowlist["torch.nn.Parameter"] = "torch.nn.Parameter"

# SECTION - Parameter methods which return a torch tensor object

# TODO: T is a property and requires detecting properties and supporting them in
# run_class_method_action.py
# allowlist["torch.nn.Parameter.T"] = #
# allowlist["torch.nn.Parameter.t"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__abs__"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__add__"] = "torch.Tensor"
allowlist["torch.nn.Parameter.__and__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__array__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__array_priority__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__array_wrap__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__bool__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__class__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__contains__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__deepcopy__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__delattr__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__delitem__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__dict__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__dir__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__div__"] = {  # exists in 1.4.0 but causes fatal exception on non floats
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }

# allowlist["torch.nn.Parameter.__doc__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__eq__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__float__"] = "torch.Tensor"
# allowlist[
#     "torch.Tensor.__floordiv__"
# ] = {  # exists in 1.4.0 but causes fatal exception on non floats
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.__format__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__ge__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__getattribute__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__getitem__"] = #
# allowlist["torch.nn.Parameter.__gt__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__hash__"] = #
# allowlist["torch.nn.Parameter.__iadd__"] = #
# allowlist["torch.nn.Parameter.__iand__"] = #
# allowlist["torch.nn.Parameter.__idiv__"] = #
# allowlist["torch.nn.Parameter.__ifloordiv__"] = #
# allowlist["torch.nn.Parameter.__ilshift__"] = #
# allowlist["torch.nn.Parameter.__imul__"] = #
# allowlist["torch.nn.Parameter.__index__"] = #
# allowlist["torch.nn.Parameter.__init__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__init_subclass__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__int__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__invert__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__ior__"] = #
# allowlist["torch.nn.Parameter.__ipow__"] = #
# allowlist["torch.nn.Parameter.__irshift__"] = #
# allowlist["torch.nn.Parameter.__isub__"] = #
# allowlist["torch.nn.Parameter.__iter__"] = #
# allowlist["torch.nn.Parameter.__itruediv__"] = #
# allowlist["torch.nn.Parameter.__ixor__"] = #
# allowlist["torch.nn.Parameter.__le__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__len__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST YET - talk to TRASK
# allowlist["torch.nn.Parameter.__long__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__lshift__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__lt__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__matmul__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__mod__"] = #
# allowlist["torch.nn.Parameter.__module__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__mul__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__ne__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__neg__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__new__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__nonzero__"] = #
# allowlist["torch.nn.Parameter.__or__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__pow__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__radd__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__rdiv__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__reduce__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__reduce_ex__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__repr__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__reversed__"] = #
# allowlist[
#     "torch.Tensor.__rfloordiv__"
# ] = {  # exists in 1.4.0 but causes fatal exception on non floats
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.__rmul__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__rpow__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__rshift__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__rsub__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__rtruediv__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__setattr__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__setitem__"] = #
# allowlist["torch.nn.Parameter.__setstate__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__sizeof__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST YET - talk to TRASK
# allowlist["torch.nn.Parameter.__str__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__sub__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__subclasshook__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__truediv__"] = #
# allowlist["torch.nn.Parameter.__weakref__"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.__xor__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter._backward_hooks"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter._base"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter._cdata"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter._coalesced_"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter._dimI"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter._dimV"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter._grad"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter._grad_fn"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter._indices"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter._is_view"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter._make_subclass"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter._nnz"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter._update_names"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter._values"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter._version"] = #
# allowlist["torch.nn.Parameter.abs"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.abs_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.acos"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.acos_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.add"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.add_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.addbmm"] = #
# allowlist["torch.nn.Parameter.addbmm_"] = #
# allowlist["torch.nn.Parameter.addcdiv"] = #
# allowlist["torch.nn.Parameter.addcdiv_"] = #
# allowlist["torch.nn.Parameter.addcmul"] = #
# allowlist["torch.nn.Parameter.addcmul_"] = #
# allowlist["torch.nn.Parameter.addmm"] = #
# allowlist["torch.nn.Parameter.addmm_"] = #
# allowlist["torch.nn.Parameter.addmv') - trie] = #
# allowlist["torch.nn.Parameter.addmv_"] = #
# allowlist["torch.nn.Parameter.addr"] = #
# allowlist["torch.nn.Parameter.addr_"] = #
# allowlist["torch.nn.Parameter.align_as"] = #
# allowlist["torch.nn.Parameter.align_to"] = #
# allowlist["torch.nn.Parameter.all"] = #
# allowlist["torch.nn.Parameter.allclose"] = #
# allowlist["torch.nn.Parameter.angle"] = #
# allowlist["torch.nn.Parameter.any"] = #
# allowlist["torch.nn.Parameter.apply_"] = #
# allowlist["torch.nn.Parameter.argmax"] = #
# allowlist["torch.nn.Parameter.argmin"] = #
# allowlist["torch.nn.Parameter.argsort"] = #
# allowlist["torch.nn.Parameter.as_strided"] = #
# allowlist["torch.nn.Parameter.as_strided_"] = #
# allowlist["torch.nn.Parameter.asin"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.asin_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.atan"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.atan_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.atan2"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.atan2_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.backward"] = #
# allowlist["torch.nn.Parameter.baddbmm"] = #
# allowlist["torch.nn.Parameter.baddbmm_"] = #
# allowlist["torch.nn.Parameter.bernoulli"] = #
# allowlist["torch.nn.Parameter.bernoulli_"] = #
# allowlist["torch.nn.Parameter.bfloat16"] = #
# allowlist["torch.nn.Parameter.binary"] = #
# allowlist["torch.nn.Parameter.bincount"] = #
# allowlist["torch.nn.Parameter.bitwise_and"] = {
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.bitwise_and_"] = {
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.bitwise_not"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.bitwise_not_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.bitwise_or"] = {
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.bitwise_or_"] = {
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.bitwise_xor"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.bitwise_xor_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.bmm"] = #
# allowlist["torch.nn.Parameter.bool"] = #
# allowlist["torch.nn.Parameter.byte"] = #
# allowlist["torch.nn.Parameter.cauchy_"] = #
# allowlist["torch.nn.Parameter.ceil"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.ceil_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.char"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.cholesky"] = #
# allowlist["torch.nn.Parameter.cholesky_inverse"] = #
# allowlist["torch.nn.Parameter.chunk"] = #
# allowlist["torch.nn.Parameter.clamp"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.clamp_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.clamp_max"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.clamp_max_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.clamp_min"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.clamp_min_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.clone"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.coalesce"] = #
# allowlist["torch.nn.Parameter.conj"] = #
# allowlist["torch.nn.Parameter.contiguous"] = #
# allowlist["torch.nn.Parameter.copy_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.cos"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.cos_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.cosh"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.cosh_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.cpu"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.cross"] = #
# allowlist["torch.nn.Parameter.cuda"] = #
# allowlist["torch.nn.Parameter.cummax"] = #
# allowlist["torch.nn.Parameter.cummin"] = #
# allowlist["torch.nn.Parameter.cumprod"] = #
# allowlist["torch.nn.Parameter.cumsum"] = #
# allowlist["torch.nn.Parameter.data"] = #
# allowlist["torch.nn.Parameter.data_ptr"] = #
# allowlist["torch.nn.Parameter.dense_dim"] = #
# allowlist["torch.nn.Parameter.dequantize"] = #
# allowlist["torch.nn.Parameter.describe"] = #
# allowlist["torch.nn.Parameter.det"] = #
# allowlist["torch.nn.Parameter.detach"] = #
# allowlist["torch.nn.Parameter.detach_"] = #
# allowlist["torch.nn.Parameter.device"] = #
# allowlist["torch.nn.Parameter.diag"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.diag_embed"] = #
# allowlist["torch.nn.Parameter.diagflat"] = #
# allowlist["torch.nn.Parameter.diagonal"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.digamma"] = #
# allowlist["torch.nn.Parameter.digamma_"] = #
# allowlist["torch.nn.Parameter.dim"] = #
# allowlist["torch.nn.Parameter.dist"] = #
# allowlist["torch.nn.Parameter.div"] = {  # exists in 1.4.0 but causes fatal exception on non floats
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.div_"] = {  # exists in 1.4.0 but causes fatal exception on non floats
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.dot"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.double"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.dtype"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST YET - talk to TRASK
# allowlist["torch.nn.Parameter.eig"] = #
# allowlist["torch.nn.Parameter.element_size"] = #
# allowlist["torch.nn.Parameter.eq"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.eq_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.equal"] = #
# allowlist["torch.nn.Parameter.erf"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.erf_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.erfc"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.erfc_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.erfinv"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.erfinv_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.exp"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.exp_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.expand"] = #
# allowlist["torch.nn.Parameter.expand_as"] = #
# allowlist["torch.nn.Parameter.expm1"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.expm1_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.exponential_"] = #
# allowlist["torch.nn.Parameter.fft"] = #
# allowlist["torch.nn.Parameter.fill_"] = #
# allowlist["torch.nn.Parameter.fill_diagonal_"] = #
# allowlist["torch.nn.Parameter.flatten"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.flip"] = #
# allowlist["torch.nn.Parameter.float"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.floor"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.floor_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.floor_divide"] = {
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.floor_divide_"] = {
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.fmod"] = #
# allowlist["torch.nn.Parameter.fmod_"] = #
# allowlist["torch.nn.Parameter.frac"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.frac_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.gather"] = #
# allowlist["torch.nn.Parameter.ge"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.ge_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.geometric_"] = #
# allowlist["torch.nn.Parameter.geqrf"] = #
# allowlist["torch.nn.Parameter.ger"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.get_device"] = #
# allowlist["torch.nn.Parameter.grad"] = #
# allowlist["torch.nn.Parameter.grad_fn"] = #
# allowlist["torch.nn.Parameter.gt"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.gt_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.half"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.hardshrink"] = #
# allowlist["torch.nn.Parameter.has_names"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.hex"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.histc"] = #
# allowlist["torch.nn.Parameter.id"] = #
# allowlist["torch.nn.Parameter.ifft"] = #
# allowlist["torch.nn.Parameter.index_add"] = #
# allowlist["torch.nn.Parameter.index_add_"] = #
# allowlist["torch.nn.Parameter.index_copy"] = #
# allowlist["torch.nn.Parameter.index_copy_"] = #
# allowlist["torch.nn.Parameter.index_fill"] = #
# allowlist["torch.nn.Parameter.index_fill_"] = #
# allowlist["torch.nn.Parameter.index_put"] = #
# allowlist["torch.nn.Parameter.index_put_"] = #
# allowlist["torch.nn.Parameter.index_select"] = #
# allowlist["torch.nn.Parameter.indices"] = #
# allowlist["torch.nn.Parameter.int"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.int_repr"] = #
# allowlist["torch.nn.Parameter.inverse"] = #
# allowlist["torch.nn.Parameter.irfft"] = #
# allowlist["torch.nn.Parameter.is_coalesced"] = #
# allowlist["torch.nn.Parameter.is_complex"] = #
# allowlist["torch.nn.Parameter.is_contiguous"] = #
# allowlist["torch.nn.Parameter.is_cuda"] = #
# allowlist["torch.nn.Parameter.is_distributed"] = #
# allowlist["torch.nn.Parameter.is_floating_point"] = "builtins.bool"
# allowlist["torch.nn.Parameter.is_leaf"] = #
# allowlist["torch.nn.Parameter.is_mkldnn"] = #
# allowlist["torch.nn.Parameter.is_nonzero"] = #
# allowlist["torch.nn.Parameter.is_pinned"] = #
# allowlist["torch.nn.Parameter.is_quantized"] = #
# allowlist["torch.nn.Parameter.is_same_size"] = #
# allowlist["torch.nn.Parameter.is_set_to"] = #
# allowlist["torch.nn.Parameter.is_shared"] = #
# allowlist["torch.nn.Parameter.is_signed"] = #
# allowlist["torch.nn.Parameter.is_sparse"] = #
# allowlist["torch.nn.Parameter.isclose"] = #
# allowlist["torch.nn.Parameter.item"] = #
# allowlist["torch.nn.Parameter.json"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.kthvalue"] = #
# allowlist["torch.nn.Parameter.layout"] = #
# allowlist["torch.nn.Parameter.le"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.le_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.lerp"] = #
# allowlist["torch.nn.Parameter.lerp_"] = #
# allowlist["torch.nn.Parameter.lgamma"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.lgamma_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.log"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.log_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.log10"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.log10_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.log1p"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.log1p_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.log2"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.log2_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.log_normal_"] = #
# allowlist["torch.nn.Parameter.log_softmax"] = #

# allowlist["torch.nn.Parameter.logdet"] = #
# allowlist["torch.nn.Parameter.logical_and"] = {
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.logical_and_"] = {
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.logical_not"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.logical_not_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.logical_or"] = {
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.logical_or_"] = {
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.logical_xor"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.logical_xor_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.logsumexp"] = #
# allowlist["torch.nn.Parameter.long"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.lstsq"] = #
# allowlist["torch.nn.Parameter.lt"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.lt_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.lu"] = #
# allowlist["torch.nn.Parameter.lu_solve"] = #
# allowlist["torch.nn.Parameter.map2_"] = #
# allowlist["torch.nn.Parameter.map_"] = #
# allowlist["torch.nn.Parameter.masked_fill"] = #
# allowlist["torch.nn.Parameter.masked_fill_"] = #
# allowlist["torch.nn.Parameter.masked_scatter"] = #
# allowlist["torch.nn.Parameter.masked_scatter_"] = #
# allowlist["torch.nn.Parameter.masked_select"] = #
# allowlist["torch.nn.Parameter.matmul"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.matrix_power"] = #
# allowlist["torch.nn.Parameter.max"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.mean"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.median"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.min"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.mm"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.mode"] = #
# allowlist["torch.nn.Parameter.mul"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.mul_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.multinomial"] = #
# allowlist["torch.nn.Parameter.mv"] = #
# allowlist["torch.nn.Parameter.mvlgamma"] = #
# allowlist["torch.nn.Parameter.mvlgamma_"] = #
# allowlist["torch.nn.Parameter.name"] = #
# allowlist["torch.nn.Parameter.names"] = #
# allowlist["torch.nn.Parameter.narrow"] = #
# allowlist["torch.nn.Parameter.narrow_copy"] = #
# allowlist["torch.nn.Parameter.ndim"] = #
# allowlist["torch.nn.Parameter.ndimension"] = #
# allowlist["torch.nn.Parameter.ne"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.ne_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.neg"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.neg_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.nelement"] = #
# allowlist["torch.nn.Parameter.new"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.new_empty"] = #
# allowlist["torch.nn.Parameter.new_full"] = #
# allowlist["torch.nn.Parameter.new_ones"] = #
# allowlist["torch.nn.Parameter.new_tensor"] = #
# allowlist["torch.nn.Parameter.new_zeros"] = #
# allowlist["torch.nn.Parameter.nonzero"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.norm"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.normal_"] = #
# allowlist["torch.nn.Parameter.numel"] = #
# allowlist["torch.nn.Parameter.numpy"] = #
# allowlist["torch.nn.Parameter.orgqr"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.ormqr') # requires two tensors as argument] = #
# allowlist["torch.nn.Parameter.output_nr"] = #
# allowlist["torch.nn.Parameter.permute"] = #
# allowlist["torch.nn.Parameter.pin_memory"] = #
# allowlist["torch.nn.Parameter.pinverse"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.polygamma"] = #
# allowlist["torch.nn.Parameter.polygamma_"] = #
# allowlist["torch.nn.Parameter.pow"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.pow_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.prelu"] = #
# allowlist["torch.nn.Parameter.prod"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.proto"] = #
# allowlist["torch.nn.Parameter.put_"] = #
# allowlist["torch.nn.Parameter.q_per_channel_axis"] = #
# allowlist["torch.nn.Parameter.q_per_channel_scales"] = #
# allowlist["torch.nn.Parameter.q_per_channel_zero_points"] = #
# allowlist["torch.nn.Parameter.q_scale"] = #
# allowlist["torch.nn.Parameter.q_zero_point"] = #
# allowlist["torch.nn.Parameter.qr"] = #
# allowlist["torch.nn.Parameter.qscheme"] = #
# allowlist["torch.nn.Parameter.random_"] = #
# allowlist["torch.nn.Parameter.reciprocal"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.reciprocal_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.record_stream"] =SECURITY WARNING: DO NOT ADD TO ALLOW LIST#
# allowlist["torch.nn.Parameter.refine_names"] = #
# allowlist["torch.nn.Parameter.register_hook"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.reinforce"] = #
# allowlist["torch.nn.Parameter.relu"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.relu_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.remainder"] = {  # exists in 1.4.0 but causes fatal exception on non floats
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.remainder_"] = {  # exists in 1.4.0 but causes fatal exception on non floats
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.rename"] = #
# allowlist["torch.nn.Parameter.rename_"] = #
# allowlist["torch.nn.Parameter.renorm"] = #
# allowlist["torch.nn.Parameter.renorm_"] = #
# allowlist["torch.nn.Parameter.repeat"] = #
# allowlist["torch.nn.Parameter.repeat_interleave"] = #
# allowlist["torch.nn.Parameter.requires_grad"] = #
# allowlist["torch.nn.Parameter.requires_grad_"] = #
# allowlist["torch.nn.Parameter.reshape"] = #
# allowlist["torch.nn.Parameter.reshape_as"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.resize"] = #
# allowlist["torch.nn.Parameter.resize_"] = #
# allowlist["torch.nn.Parameter.resize_as"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.resize_as_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.retain_grad"] = #
# allowlist["torch.nn.Parameter.rfft"] = #
# allowlist["torch.nn.Parameter.roll"] = #
# allowlist["torch.nn.Parameter.rot90"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.round"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.round_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.rsqrt"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.rsqrt_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.scatter"] = #
# allowlist["torch.nn.Parameter.scatter_"] = #
# allowlist["torch.nn.Parameter.scatter_add"] = #
# allowlist["torch.nn.Parameter.scatter_add_"] = #
# allowlist["torch.nn.Parameter.select"] = #
# allowlist["torch.nn.Parameter.send"] = #
# allowlist["torch.nn.Parameter.serializable_wrapper_type"] = #
# allowlist["torch.nn.Parameter.serialize"] = #
# allowlist["torch.nn.Parameter.set_"] = #
# allowlist["torch.nn.Parameter.shape"] = #
# allowlist["torch.nn.Parameter.share_memory_"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.short"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.sigmoid"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.sigmoid_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.sign"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.sign_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.sin"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.sin_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.sinh"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.sinh_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.size"] = #
# allowlist["torch.nn.Parameter.slogdet"] = #
# allowlist["torch.nn.Parameter.smm"] = #
# allowlist["torch.nn.Parameter.softmax"] = #
# allowlist["torch.nn.Parameter.solve"] = #
# allowlist["torch.nn.Parameter.sort"] = #
# allowlist["torch.nn.Parameter.sparse_dim"] = #
# allowlist["torch.nn.Parameter.sparse_mask"] = #
# allowlist["torch.nn.Parameter.sparse_resize_"] = #
# allowlist["torch.nn.Parameter.sparse_resize_and_clear_"] = #
# allowlist["torch.nn.Parameter.split"] = #
# allowlist["torch.nn.Parameter.split_with_sizes"] = #
# allowlist["torch.nn.Parameter.sqrt"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.sqrt_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.square"] = {
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.square_"] = {
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.squeeze"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.squeeze_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.sspaddmm"] = #
# allowlist["torch.nn.Parameter.std"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.stft"] = #
# allowlist["torch.nn.Parameter.storage"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.storage_offset"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.storage_type"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.stride"] = #
# allowlist["torch.nn.Parameter.sub"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.sub_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.sum"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.sum_to_size"] = #
# allowlist["torch.nn.Parameter.svd"] = #
# allowlist["torch.nn.Parameter.symeig"] = #
# allowlist["torch.nn.Parameter.t"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.t_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.tag"] = #
# allowlist["torch.nn.Parameter.take"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.tan"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.tan_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.tanh"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.tanh_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.to"] = #
# allowlist["torch.nn.Parameter.to_binary"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.to_dense"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.to_hex"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.to_json"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.to_mkldnn"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.to_proto"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.to_sparse"] = SERDE WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.tolist"] = #
# allowlist["torch.nn.Parameter.topk"] = #
# allowlist["torch.nn.Parameter.trace"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.transpose"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.transpose_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.triangular_solve"] = #
# allowlist["torch.nn.Parameter.tril"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.tril_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.triu"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.triu_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.true_divide"] = {
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.true_divide_"] = {
#     "return_type": "torch.Tensor",
#     "min_version": "1.5.0",
# }
# allowlist["torch.nn.Parameter.trunc"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.trunc_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.type"] =
# allowlist["torch.nn.Parameter.type_as"] = #
# allowlist["torch.nn.Parameter.unbind"] = #
# allowlist["torch.nn.Parameter.unflatten"] = #
# allowlist["torch.nn.Parameter.unfold"] = #
# https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.uniform_
# allowlist["torch.nn.Parameter.uniform_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.unique"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.unique_consecutive"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.unsqueeze"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.unsqueeze_"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.values"] = #
# allowlist["torch.nn.Parameter.var"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.view"] = #
# allowlist["torch.nn.Parameter.view_as"] = #
# allowlist["torch.nn.Parameter.where"] = SECURITY WARNING: DO NOT ADD TO ALLOW LIST
# allowlist["torch.nn.Parameter.zero_"] = "torch.Tensor"


# SECTION - other classes and modules

# allowlist["torch.zeros"] = "torch.Tensor"
# allowlist["torch.ones"] = "torch.Tensor"
# allowlist["torch.nn.Linear"] = "torch.nn.Linear"
# allowlist.add("torch.nn.Linear.parameters")
# allowlist["torch.nn.parameter.Parameter"] = "torch.nn.parameter.Parameter"

# SECTION - Parameter methods

# allowlist["torch.nn.Parameter.t"] = "torch.nn.Parameter"
# allowlist["torch.nn.Parameter.__abs__"] = "torch.nn.Parameter"
# allowlist["torch.nn.Parameter.__add__"] = "torch.Tensor"
# allowlist["torch.nn.Parameter.__and__"] = "torch.nn.Parameter"
