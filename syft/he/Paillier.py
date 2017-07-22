import phe as paillier
import numpy as np
import pickle

class SecretKey():

    def __init__(self,sk):
        self.sk = sk

    def decrypt(self,x):
        """Decrypts x. X can be either an encrypted int or a numpy vector/matrix/tensor."""
        if(type(x) == PaillierFloat):
            return self.sk.decrypt(x)
        elif(type(x) == np.ndarray):
            sh = x.shape
            x_ = x.reshape(-1)
            out = list()
            for v in x_:
                out.append(self.sk.decrypt(v.data))
            return np.array(out).reshape(sh)

    def serialize(self):
        return pickle.dumps(self.sk)

class PublicKey():

    def __init__(self,pk):
        self.pk = pk

    def encrypt(self,x):
        """Encrypts x. X can be either an encrypted int or a numpy vector/matrix/tensor."""
        if(type(x) == int):
            return PaillierFloat(self,x)
        elif(type(x) == np.ndarray):
            sh = x.shape
            x_ = x.reshape(-1)
            out = list()
            for v in x_:
                out.append(PaillierFloat(self,v))
            return np.array(out).reshape(sh)

        else:
            print("format not recognized")

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

class PaillierFloat():

    def __init__(self,public_key,data=None):
        """Wraps pointer to encrypted integer with an interface that numpy can use."""

        self.public_key = public_key
        if(data is not None):
            self.data = self.public_key.pk.encrypt(data)
        else:
            self.data = None

    def __add__(self,y):
        """Adds two encrypted integers together."""

        out = PaillierFloat(self.public_key,None)
        out.data = self.data + y.data
        return out

    def __sub__(self,y):
        """Subtracts two encrypted integers."""

        out = PaillierFloat(self.public_key, None)
        out.data = self.data - y.data
        return out

    def __mul__(self,y):
        """Multiplies two integers. y may be encrypted or a simple integer."""

        if(type(y) == type(self)):
            out = PaillierFloat(self.public_key, None)
            out.data = self.data * y.data
            return out
        elif(type(y) == int or type(y) == float):
            out = PaillierFloat(self.public_key, None)
            out.data = self.data * y
            return out
        else:
            return None

    def __truediv__(self,y):
        """Divides two integers. y may be encrypted or a simple integer."""

        if(type(y) == type(self)):
            out = PaillierFloat(self.public_key, None)
            out.data = self.data / y.data
            return out
        elif(type(y) == int):
            out = PaillierFloat(self.public_key, None)
            out.data = self.data / y
            return out
        else:
            return None

    def __repr__(self):
        """This is kindof a boring/uninformative __repr__"""

        return 'e'

    def __str__(self):
        """This is kindof a boring/uninformative __str__"""

        return 'e'
