# # from unittest import TestCase
# # import torch
#
# # from syft.core.hooks import TorchHook
#
#
# # class TestTorchHook(TestCase):
#
# #     def test_types_guard(self):
# #         hook = TorchHook(verbose=False)
#
# #         with self.assertRaises(Exception) as context:
# #             # can't serialize an int, so should raise a TypError
# #             obj_type = hook.types_guard(3)
#
# #         with self.assertRaises(Exception) as context:
#
# #             # can't serialize a random string as a type, so should raise a TypError
# #             obj_type = hook.guard.types_guard("asdf")
#
# #             assert obj_type == obj_type
#
# #             self.assertTrue('TypeError' in context.exception)
#
# <<<<<<< HEAD
#         tensor_types = {
#             'torch.FloatTensor': torch.FloatTensor,
#             'torch.DoubleTensor': torch.DoubleTensor,
#             'torch.HalfTensor': torch.HalfTensor,
#             'torch.ByteTensor': torch.ByteTensor,
#             'torch.CharTensor': torch.CharTensor,
#             'torch.ShortTensor': torch.ShortTensor,
#             'torch.IntTensor': torch.IntTensor,
#             'torch.LongTensor': torch.LongTensor,
#         }
# =======
# #         tensor_types = {
# #             'torch.FloatTensor': torch.FloatTensor,
# #             'torch.DoubleTensor': torch.DoubleTensor,
# #             'torch.HalfTensor': torch.HalfTensor,
# #             'torch.ByteTensor': torch.ByteTensor,
# #             'torch.CharTensor': torch.CharTensor,
# #             'torch.ShortTensor': torch.ShortTensor,
# #             'torch.IntTensor': torch.IntTensor,
# #             'torch.LongTensor': torch.LongTensor
# #         }
# >>>>>>> pointers_are_tensors
#
# #         for k, v in tensor_types.items():
# #             assert hook.guard.types_guard(k) == v
