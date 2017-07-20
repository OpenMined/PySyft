import pyRserve,os
import random
import numpy as np
import tempfile

class SecretKey():
    def __init__(self,conn,sk_str,params,r1k_data,sk_data):
        self.conn = conn
        self.params = params
        self.sk_data = sk_data
        self.r1k_data = r1k_data
        self.sk_str = sk_str

    def decrypt(self,x):
        return self.conn.eval('dec('+str(self.sk_str)+', '+str(x.vector_name)+')')

class PublicKey():
    def __init__(self,conn,pk_str,params,r1k_data,pk_data):
        self.conn = conn
        self.params = params
        self.pk_data = pk_data
        self.r1k_data = r1k_data
        self.pk_str = pk_str

    def encrypt(self,x=4):
        if(type(x) == int):
            return Scalar(self.conn,self,x)
        elif(type(x) == np.ndarray):
            sh = x.shape
            x_ = x.reshape(-1)
            out = list()
            for v in x_:
                out.append(Scalar(self.conn,self,int(v)))
            return np.array(out).reshape(sh)

        else:
            print("format not recognized")

class KeyPair():
    def __init__(self,conn=None,var_name=None,file_path=None):
        if(conn is None):
            self.conn = pyRserve.connect()
        else:
            self.conn = conn
        self.conn.r('library("HomomorphicEncryption")',void=True)


    def generate(self,lambd=90,L=10):

        k_str = 'k'+str(random.randint(0,2**32))

        tmp_file = tempfile.mktemp()

        p = self.conn.eval('p <- pars("FandV", lambda='+str(lambd)+', L='+str(L)+')')
        k = self.conn.eval(k_str+' <- keygen(p)')
        self.conn.eval('saveFHE('+str(k_str)+',file="'+tmp_file+'")')

        f = open(tmp_file,'r')
        raw = f.readlines()
        f.close()

        boiler = raw[0:4]
        params = raw[4:10]
        boiler2 = raw[10:12]
        r1k = raw[12:16]
        boiler3 = raw[16:18]
        pk = raw[18:20]
        boiler4 = raw[20:22]
        sk = raw[22:23]

        self.secret_key = SecretKey(self.conn,k_str+'$sk',params,r1k,sk)
        self.public_key = PublicKey(self.conn,k_str+'$pk',params,r1k,pk)

        os.remove(tmp_file)
        return (self.public_key,self.secret_key)


class Scalar():

    def __init__(self,conn,public_key,data=None):

        self.conn = conn
        self.vector_name ='c'+str(random.randint(0,2**32))
        self.public_key = public_key
        self.length = 1
        if(data is not None):
            self.conn.eval(self.vector_name+' <- enc('+str(self.public_key.pk_str)+', c('+str(data)+'))')

    def __add__(self,y):
        vector_name='c'+str(random.randint(0,2**32))
        out = Scalar(self.conn,self.public_key)
        self.conn.eval(out.vector_name+' <- '+self.vector_name+' + '+y.vector_name)
        return out

    def __mul__(self,y):
        vector_name='c'+str(random.randint(0,2**32))
        out = Scalar(self.conn,self.public_key)
        if(type(y) == type(self)):
            self.conn.eval(out.vector_name+' <- '+self.vector_name+' * '+y.vector_name)
        elif(type(y) == int):
            adds = '+'+self.vector_name

            self.conn.eval(out.vector_name+' <- ('+self.vector_name+(adds*(y-1)) + ")")
        return out

    def __repr__(self):
        return 'e'

    def __str__(self):
        return 'e'
