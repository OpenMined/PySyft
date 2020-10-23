# """
# File copied from cpython test suite:
# https://github.com/python/cpython/blob/3.8/Lib/test/test_userstring.py
#
# Those are the UserString tests that are used for our String (from Syft) class
#
# UserString is a wrapper around the native builtin string type.
# UserString instances should behave similar to builtin string objects.
# """
#
# # stdlib
# import sys
# import unittest
#
# # third party
# import string_utils_test as string_tests
#
# # syft absolute
# from syft.lib.python.string import String
#
#
# class UserStringTest(
#     string_tests.CommonTest,
#     string_tests.MixinStrUnicodeUserStringTest,
#     unittest.TestCase,
# ):
#
#     type2test = String
#
#     # Overwrite the three testing methods, because UserString
#     # can't cope with arguments propagated to UserString
#     # (and we don't test with subclasses)
#     def checkequal(self, result, object, methodname, *args, **kwargs):
#         result = self.fixtype(result)
#         object = self.fixtype(object)
#         # we don't fix the arguments, because UserString can't cope with it
#         realresult = getattr(object, methodname)(*args, **kwargs)
#
#         self.assertEqual(result, realresult)
#
#     def checkraises(self, exc, obj, methodname, *args):
#         obj = self.fixtype(obj)
#         # we don't fix the arguments, because UserString can't cope with it
#         with self.assertRaises(exc) as cm:
#             getattr(obj, methodname)(*args)
#         self.assertNotEqual(str(cm.exception), "")
#
#     def checkcall(self, object, methodname, *args):
#         object = self.fixtype(object)
#         # we don't fix the arguments, because UserString can't cope with it
#         getattr(object, methodname)(*args)
#
#     def test_rmod(self):
#         class ustr2(String):
#             pass
#
#         class ustr3(ustr2):
#             def __rmod__(self, other):
#                 return super().__rmod__(other)
#
#         fmt2 = ustr2("value is %s")
#         str3 = ustr3("TEST")
#         self.assertEqual(fmt2 % str3, "value is TEST")
#
#     # added as a regression test in 3.8
#     # https://github.com/python/cpython/commit/2cb82d2a88710b0af10b9d9721a9710ecc037e72
#     def test_encode_default_args(self):
#         if sys.version_info >= (3, 8):
#             self.checkequal(b"hello", "hello", "encode")
#             # Check that encoding defaults to utf-8
#             self.checkequal(b"\xf0\xa3\x91\x96", "\U00023456", "encode")
#             # Check that errors defaults to 'strict'
#             self.checkraises(UnicodeError, "\ud800", "encode")
#
#     # added as a regression test in 3.8
#     # https://github.com/python/cpython/commit/2cb82d2a88710b0af10b9d9721a9710ecc037e72
#     def test_encode_explicit_none_args(self):
#         if sys.version_info >= (3, 8):
#             self.checkequal(b"hello", "hello", "encode", None, None)
#             # Check that encoding defaults to utf-8
#             self.checkequal(b"\xf0\xa3\x91\x96", "\U00023456", "encode", None, None)
#             # Check that errors defaults to 'strict'
#             self.checkraises(UnicodeError, "\ud800", "encode", None, None)
