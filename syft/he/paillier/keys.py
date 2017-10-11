import phe as paillier
import numpy as np
import json
import syft
from .basic import PaillierTensor
from ...tensor import TensorBase
from ..abstract.keys import AbstractSecretKey, AbstractPublicKey
from ..abstract.keys import AbstractKeyPair
import math


class SecretKey(AbstractSecretKey):

    def __init__(self, sk):
        self.sk = sk

    def decrypt(self, x, precision_conf=None):
        """Decrypts x. X can be either an encrypted int or a numpy
        vector/matrix/tensor."""

        if(type(x) == paillier.EncryptedNumber):
            # casts the fixed point data type to python numeric type
            return self.sk.decrypt(x)
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
                python_numeric = self.sk.decrypt(v)
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

    def encrypt(self, x, same_type=False, precision_conf=None):
        """Encrypts x. X can be either an encrypted int or a numpy
        vector/matrix/tensor."""
        if(type(x) == int or type(x) == float or type(x) == np.float64):
            if(same_type):
                return NotImplemented
            return self.pk.encrypt(x, precision_conf)
        elif(type(x) == TensorBase):
            if(x.encrypted or same_type):
                return NotImplemented
            return PaillierTensor(self, x.data)
        elif(type(x) == np.ndarray):
            sh = x.shape
            x_ = x.reshape(-1)
            out = list()
            for v in x_:
                out.append(self.pk.encrypt(v, precision_conf))
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


# This is our custom implementations of `encode` and `decode`. It replaces those of `phe`'s EncodedNumber.
# This gives us control over the precision of numbers.
@classmethod
def encode(cls, public_key, scalar, precision=None, max_exponent=None):

    # Calculate the maximum exponent for desired precision
    if precision is None:
        if isinstance(scalar, int):
            exponent = 0
        elif isinstance(scalar, float):
            # Encode with *at least* as much precision as the python float
            # What's the base-2 exponent on the float?
            bin_flt_exponent = math.frexp(scalar)[1]

            # What's the base-2 exponent of the least significant bit?
            # The least significant bit has value 2 ** bin_lsb_exponent
            bin_lsb_exponent = bin_flt_exponent - cls.FLOAT_MANTISSA_BITS

            # What's the corresponding base BASE exponent? Round that down.
            exponent = math.floor(bin_lsb_exponent / cls.LOG2_BASE)
        else:
            raise TypeError("Don't know the precision of type %s."
                            % type(scalar))
    else:
        exponent = -precision.fraction_bits

    int_rep = int(scalar * pow(cls.BASE, -exponent))
    if abs(int_rep) > public_key.max_int:
        raise ValueError('Integer needs to be within +/- %d but got %d'
                         % (public_key.max_int, int_rep))
    # Wrap negative numbers by adding n
    return cls(public_key, int_rep % public_key.n, exponent)


def decode(self):
    """Decode fixed-point plaintext and return the result.

    Returns:
      returns an int if `exponent` is < 1. Returns a float otherwise.

    Raises:
      OverflowError: if overflow is detected in the decrypted number.
    """
    if self.encoding >= self.public_key.n:
        # Should be mod n
        raise ValueError('Attempted to decode corrupted number')
    elif self.encoding <= self.public_key.max_int:
        # Positive
        mantissa = self.encoding
    elif self.encoding >= self.public_key.n - self.public_key.max_int:
        # Negative
        mantissa = self.encoding - self.public_key.n
    else:
        raise OverflowError('Overflow detected in decrypted number')
    if self.exponent <= 1:
        return int(mantissa * pow(self.BASE, self.exponent))
    else:
        return float(mantissa * pow(self.BASE, self.exponent))


def my__mul__(self, other):
    """Multiply by an int, float, or EncodedNumber.
    If `other` is a scalar, it is first encoded to fixed-point precision of `self`"""
    if isinstance(other, paillier.EncryptedNumber):
        raise NotImplementedError('Good luck with that...')
    if isinstance(other, paillier.EncodedNumber):
        encoding = other
    else:
        int_rep = int(other * pow(paillier.EncodedNumber.BASE, -self.exponent))
        if abs(int_rep) > self.public_key.max_int:
            raise ValueError('Integer needs to be within +/- %d but got %d'
                             % (self.public_key.max_int, int_rep))
        # Wrap negative numbers by adding n
        encoding = paillier.EncodedNumber(self.public_key, int_rep % self.public_key.n, self.exponent)
    product = self._raw_mul(encoding.encoding)
    exponent = self.exponent + encoding.exponent

    return paillier.EncryptedNumber(self.public_key, product, exponent)


paillier.EncodedNumber.encode = encode
paillier.EncodedNumber.decode = decode
paillier.EncodedNumber.BASE = 2
paillier.EncryptedNumber.__mul__ = my__mul__
