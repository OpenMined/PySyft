import phe as paillier
import numpy as np
import pickle
from .basic import Float,PaillierTensor
from ...tensor import TensorBase
# from ..abstract.keys import AbstractSecretKey, AbstractPublicKey, AbstractKeyPair

class SecretKey():

    def __init__(self,sk):
        self.sk = sk

    def decrypt(self,x):
        """Decrypts x. X can be either an encrypted int or a numpy vector/matrix/tensor."""
        if(type(x) == Float):
            return self.sk.decrypt(list(x.data))
        elif(type(x) == TensorBase):
            if(x.encrypted):
                return TensorBase(self.decrypt(x.data),encrypted=False)
            else:
                return NotImplemented
        elif(type(x) == np.ndarray):
            sh = x.shape
            x_ = x.reshape(-1)
            out = list()
            for v in x_:
                out.append(self.sk.decrypt(v.data))
            return np.array(out).reshape(sh)
        else:
            return NotImplemented

    def serialize(self):
        return pickle.dumps(self.sk)

class PublicKey():

    def __init__(self,pk):
        self.pk = pk

    def encrypt(self,x,same_type=False):
        """Encrypts x. X can be either an encrypted int or a numpy vector/matrix/tensor."""
        if(type(x) == int):
            if(same_type):
                return NotImplemented
            return Float(self,x)
        elif(type(x) == TensorBase):
            if(x.encrypted or same_type):
                return NotImplemented
            return PaillierTensor(self,x.data)
        elif(type(x) == np.ndarray):
            sh = x.shape
            x_ = x.reshape(-1)
            out = list()
            for v in x_:
                out.append(Float(self,v))
            if(same_type):
                return np.array(out).reshape(sh)
            else:
                return PaillierTensor(self,np.array(out).reshape(sh))
        else:
            print("format not recognized")
            return NotImplemented

        return self.pk.encrypt(x)

    def serialize(self):
        return pickle.dumps(self.pk)

class KeyPair():

    def __init__(self):
        ""

    def deserialize(self,pubkey,seckey):
        self.public_key = PublicKey(pickle.loads(pubkey))
        self.secret_key = SecretKey(pickle.loads(seckey))
        return (self.public_key, self.secret_key)

    def generate(self,n_length=1024):
        pubkey, prikey = paillier.generate_paillier_keypair(n_length=n_length)
        self.public_key = PublicKey(pubkey)
        self.secret_key = SecretKey(prikey)

        return (self.public_key,self.secret_key)
