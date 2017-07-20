import phe as paillier

class SecretKey():

    def __init__(self,sk):
        self.sk = sk

    def decrypt(self,x):
        return self.sk.decrypt(x)

class PublicKey():

    def __init__(self,pk):
        self.pk = pk

    def encrypt(self,x):
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
        self.data = data

    def __add__(self,y):
        out = PaillierInteger(self.public_key,self.data + y.data)
        return out

    def __sub__(self,y):
        out = PaillierInteger(self.public_key, self.data - y.data)
        return out

    def __mul__(self,y):

        if(type(y) == type(self)):
            return PaillierInteger(self.public_key, self.data * y.data)
        elif(type(y) == int):
            return PaillierInteger(self.public_key, self.data * y)
        else:
            return None

    def __truediv__(self,y):

        if(type(y) == type(self)):
            return PaillierInteger(self.public_key, self.data / y.data)
        elif(type(y) == int):
            return PaillierInteger(self.public_key, self.data / y)
        else:
            return None

    def __repr__(self):
        return 'e'

    def __str__(self):
        return 'e'
