import phe as paillier
import numpy as np
import json
import syft
from .basic import PaillierTensor
from .FixedPoint import FXnum as FixedPoint
from ...tensor import TensorBase
from ..abstract.keys import AbstractSecretKey, AbstractPublicKey
from ..abstract.keys import AbstractKeyPair


class SecretKey(AbstractSecretKey):

    def __init__(self, sk):
        self.sk = sk

    def decrypt(self, x):
        """Decrypts x. X can be either an encrypted int or a numpy
        vector/matrix/tensor."""

        if(type(x) == paillier.EncryptedNumber):
            FP_num = FixedPoint()
            FP_num.scaledval = int(self.sk.decrypt(x))
            # casts the fixed point data type to python numeric type
            return float(FP_num) if FP_num.family.fraction_bits > 1 else int(FP_num)
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
                FP_num = FixedPoint()
                FP_num.scaledval = int(self.sk.decrypt(v))
                python_numeric = float(FP_num) if FP_num.family.fraction_bits > 1 else int(FP_num)
                out.append(python_numeric)
            return np.array(out).reshape(sh)
        else:
            return NotImplemented

    def serialize(self):
        seckey_dict = {}
        seckey_dict['secret_key'] = {
            'p': self.sk.p,
            'q': self.sk.q,
            'n': self.sk.public_key.n
        }
        return json.dumps(seckey_dict)

    def deserialize(b):
        seckey_dict = json.loads(b)
        sk_record = seckey_dict['secret_key']
        sk = paillier.PaillierPrivateKey(
            public_key=paillier.PaillierPublicKey(n=int(sk_record['n'])),
            p=sk_record['p'],
            q=sk_record['q'])
        return SecretKey(sk)


class PublicKey(AbstractPublicKey):

    def __init__(self, pk):
        self.pk = pk

    def zeros(self, dim):
        """Returns an encrypted tensor of zeros"""
        return PaillierTensor(self, syft.zeros(dim))

    def ones(self, dim):
        """Returns an encrypted tensor of ones"""
        return PaillierTensor(self, syft.ones(dim))

    def rand(self, dim):
        """Returns an encrypted tensor with initial numbers sampled from a
        uniform distribution from 0 to 1."""
        return PaillierTensor(self, syft.rand(dim))

    def randn(self, dim):
        """Returns an encrypted tensor with initial numbers sampled from a
        standard normal distribution"""
        return PaillierTensor(self, syft.randn(dim))

    def encrypt(self, x, same_type=False):
        """Encrypts x. X can be either an encrypted int or a numpy
        vector/matrix/tensor."""
        if(type(x) == int or type(x) == float or type(x) == np.float64):
            if(same_type):
                return NotImplemented
            return FixedPoint(self, x)
        elif(type(x) == TensorBase):
            if(x.encrypted or same_type):
                return NotImplemented
            return PaillierTensor(self, x.data)
        elif(type(x) == np.ndarray):
            sh = x.shape
            x_ = x.reshape(-1)
            out = list()
            for v in x_:
                num = FixedPoint(v)
                out.append(self.pk.encrypt(num.scaledval))
            if(same_type):
                return np.array(out).reshape(sh)
            else:
                return PaillierTensor(self, np.array(out).reshape(sh))
        else:
            print("format not recognized:" + str(type(x)))
            return NotImplemented

        return self.pk.encrypt(x)

    def serialize(self):
        pubkey_dict = {}
        pubkey_dict['public_key'] = {
            'n': self.pk.n
        }
        return json.dumps(pubkey_dict)

    def deserialize(b):
        pubkey_dict = json.loads(b)
        pk_record = pubkey_dict['public_key']
        pk = paillier.PaillierPublicKey(n=int(pk_record['n']))
        return PublicKey(pk)


class KeyPair(AbstractKeyPair):

    def __init__(self):
        ""

    def deserialize(self, pubkey, seckey):
        self.public_key = PublicKey.deserialize(pubkey)
        self.secret_key = SecretKey.deserialize(seckey)
        return (self.public_key, self.secret_key)

    def generate(self, n_length=1024):
        pubkey, prikey = paillier.generate_paillier_keypair(n_length=n_length)
        self.public_key = PublicKey(pubkey)
        self.secret_key = SecretKey(prikey)

        return (self.public_key, self.secret_key)
