import phe as paillier
import numpy as np
import pickle
import syft
from .basic import FixedPoint, PaillierTensor
from ...tensor import TensorBase
from ..abstract.keys import AbstractSecretKey, AbstractPublicKey
from ..abstract.keys import AbstractKeyPair


class SecretKey(AbstractSecretKey):

    def __init__(self, sk):
        self.sk = sk

    def decrypt(self, x):
        """Decrypts x. X can be either an encrypted int or a numpy
        vector/matrix/tensor."""

        if(type(x) == FixedPoint):
            return x.decrypt(self.sk)
        elif(type(x) == TensorBase or type(x) == PaillierTensor):
            if(x.encrypted):
                return TensorBase(self.decrypt(x.data), encrypted=False)
            else:
                return NotImplemented
        elif(type(x) == np.ndarray):
            sh = x.shape
            x_ = x.reshape(-1)
            out = list()
            for v in x_:
                out.append(v.decrypt(self.sk))
            return np.array(out).reshape(sh)
        else:
            return NotImplemented

    def serialize(self):
        return pickle.dumps(self.sk)

    def deserialize(b):
        return SecretKey(pickle.loads(b))


class PublicKey(AbstractPublicKey):

    def __init__(self, pk):
        self.pk = pk

    def zeros(self, dim, fixed_point_conf=None):
        """Returns an encrypted tensor of zeros"""
        return PaillierTensor(self, syft.zeros(dim), fixed_point_conf=fixed_point_conf)

    def ones(self, dim, fixed_point_conf=None):
        """Returns an encrypted tensor of ones"""
        return PaillierTensor(self, syft.ones(dim), fixed_point_conf=fixed_point_conf)

    def rand(self, dim, fixed_point_conf=None):
        """Returns an encrypted tensor with initial numbers sampled from a
        uniform distribution from 0 to 1."""
        return PaillierTensor(self, syft.rand(dim), fixed_point_conf=fixed_point_conf)

    def randn(self, dim, fixed_point_conf=None):
        """Returns an encrypted tensor with initial numbers sampled from a
        standard normal distribution"""
        return PaillierTensor(self, syft.randn(dim), fixed_point_conf=fixed_point_conf)

    def encrypt(self, x, same_type=False, fixed_point_conf=None):
        """Encrypts x. X can be either an encrypted int or a numpy
        vector/matrix/tensor."""
        if(type(x) == int or type(x) == float or type(x) == np.float64):
            if(same_type):
                return NotImplemented
            return FixedPoint(self, x, config=fixed_point_conf)
        elif(type(x) == TensorBase):
            if(x.encrypted or same_type):
                return NotImplemented
            return PaillierTensor(self, x.data, fixed_point_conf=fixed_point_conf)
        elif(type(x) == np.ndarray):
            sh = x.shape
            x_ = x.reshape(-1)
            out = list()
            for v in x_:
                out.append(FixedPoint(self, v, config=fixed_point_conf))
            if(same_type):
                return np.array(out).reshape(sh)
            else:
                return PaillierTensor(self, np.array(out).reshape(sh), fixed_point_conf=fixed_point_conf)
        else:
            print("format not recognized:" + str(type(x)))
            return NotImplemented

        return self.pk.encrypt(x)

    def serialize(self):
        return pickle.dumps(self.pk)

    def deserialize(b):
        return PublicKey(pickle.loads(b))


class KeyPair(AbstractKeyPair):

    def __init__(self):
        ""

    def deserialize(self, pubkey, seckey):
        self.public_key = PublicKey(pickle.loads(pubkey))
        self.secret_key = SecretKey(pickle.loads(seckey))
        return (self.public_key, self.secret_key)

    def generate(self, n_length=1024):
        pubkey, prikey = paillier.generate_paillier_keypair(n_length=n_length)
        self.public_key = PublicKey(pubkey)
        self.secret_key = SecretKey(prikey)

        return (self.public_key, self.secret_key)
