import phe as paillier
import numpy as np

class SecretKey():

    def __init__(self,sk):
        self.sk = sk

    def decrypt(self,x):
        if(type(x) == PaillierInteger):
            return self.sk.decrypt(x)
        elif(type(x) == np.ndarray):
            sh = x.shape
            x_ = x.reshape(-1)
            out = list()
            for v in x_:
                out.append(self.sk.decrypt(v.data))
            return np.array(out).reshape(sh)

class PublicKey():

    def __init__(self,pk):
        self.pk = pk

    def encrypt(self,x):
        if(type(x) == int):
            return PaillierInteger(self,x)
        elif(type(x) == np.ndarray):
            sh = x.shape
            x_ = x.reshape(-1)
            out = list()
            for v in x_:
                out.append(PaillierInteger(self,v))
            return np.array(out).reshape(sh)

        else:
            print("format not recognized")

        return self.pk.encrypt(x)

class KeyPair():

    def __init__(self):
        ""

    def generate(self):
        pubkey, prikey = paillier.generate_paillier_keypair(n_length=2048*2)
        self.public_key = PublicKey(pubkey)
        self.secret_key = SecretKey(prikey)

        return (self.public_key,self.secret_key)

class PaillierInteger():

    def __init__(self,public_key,data=None):

        self.public_key = public_key
        if(data is not None):
            self.data = self.public_key.pk.encrypt(data)

    def __add__(self,y):
        out = PaillierInteger(self.public_key,None)
        out.data = self.data + y.data
        return out

    def __sub__(self,y):
        out = PaillierInteger(self.public_key, None)
        out.data = self.data - y.data
        return out

    def __mul__(self,y):

        if(type(y) == type(self)):
            out = PaillierInteger(self.public_key, None)
            out.data = self.data * y.data
            return out
        elif(type(y) == int):
            out = PaillierInteger(self.public_key, None)
            out.data = self.data * y
            return out
        else:
            return None

    def __truediv__(self,y):

        if(type(y) == type(self)):
            out = PaillierInteger(self.public_key, None)
            out.data = self.data / y.data
            return out
        elif(type(y) == int):
            out = PaillierInteger(self.public_key, None)
            out.data = self.data / y
            return out
        else:
            return None

    def __repr__(self):
        return 'e'

    def __str__(self):
        return 'e'
