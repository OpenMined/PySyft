# import torch as th
# from functools import reduce
#
# prod = lambda xs: reduce(lambda x, y: x * y, xs)
#
#
# def _egcd(a, b):
#     if a == 0:
#         return (b, 0, 1)
#     else:
#         g, y, x = _egcd(b % a, a)
#         return (g, x - (b // a) * y, y)
#
#
# def _inverse(a, m):
#     _, b, _ = _egcd(a, m)
#     return b % m
#
#
# moduli = [1999703, 1990007, 1996949, 1925899, 1816117]
# modulus = prod(moduli)
# moduli_inverses = [_inverse(modulus // mi, mi) for mi in moduli]
#
#
# class CrtTensor(object):
#
#     def __init__(self, values, residues=None):
#         if values is not None:
#             residues = [values % mi for mi in moduli]
#         self.residues = residues
#
#     @staticmethod
#     def sample_uniform(shape):
#         return CrtTensor(None, [
#             th.randint(0, mi, shape).type(th.LongTensor)
#             for mi in moduli
#         ])
#
#     def recombine(self, bound=2 ** 31):
#         return self._explicit_crt(bound)
#
#     def __add__(self, other):
#         return CrtTensor(None, [
#             (xi + yi) % mi
#             for xi, yi, mi in zip(self.residues, other.residues, moduli)
#         ])
#
#     def __sub__(self, other):
#         return CrtTensor(None, [
#             (xi - yi) % mi
#             for xi, yi, mi in zip(self.residues, other.residues, moduli)
#         ])
#
#     def __mul__(self, other):
#         return CrtTensor(None, [
#             (xi * yi) % mi
#             for xi, yi, mi in zip(self.residues, other.residues, moduli)
#         ])
#
#     def matmul(self, other):
#         return CrtTensor(None, [
#             th.matmul(xi, yi) % mi
#             for xi, yi, mi in zip(self.residues, other.residues, moduli)
#         ])
#
#     def __mod__(self, k):
#         return CrtTensor(self._explicit_crt(k))
#
#     def _explicit_crt(self, bound):
#         def sum(xs):
#             return th.cat(xs).view(len(moduli), *xs[0].shape).sum(0)
#
#         t = [
#             (xi * qi) % mi
#             for xi, qi, mi in zip(self.residues, moduli_inverses, moduli)
#         ]
#         alpha = sum(tuple(
#             ti.type(th.DoubleTensor) / float(mi)
#             for ti, mi in zip(t, moduli)
#         ))
#
#         b = [(modulus // mi) % bound for mi in moduli]
#         u = sum(tuple(
#             ti * bi
#             for ti, bi in zip(t, b)
#         ))
#
#         B = modulus % bound
#         v = th.round(alpha).type(th.LongTensor) * B
#         w = u.type(th.LongTensor) - v
#
#         return w % bound
#
#
# x = CrtTensor(th.LongTensor([100000, 200000, 300000, 400000]).reshape(2, 2))
# y = CrtTensor(th.LongTensor([100000, 200000, 300000, 400000]).reshape(2, 2))
# z = x.matmul(y)
# print(z.recombine(2 ** 40))
#
# x = CrtTensor.sample_uniform((2, 2))
# print(x.residues)
