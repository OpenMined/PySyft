import pyRserve,os
import random
import numpy as np
import json
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

    def serialize(self):
        ser_seckey = {}
        ser_seckey['params'] = self.params
        ser_seckey['sk_data'] = self.sk_data
        ser_seckey['sk_str'] = self.sk_str
        ser_seckey['r1k_data'] = self.r1k_data
        return json.dumps(ser_seckey)

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

    def serialize(self):
        ser_pubkey = {}
        ser_pubkey['params'] = self.params
        ser_pubkey['pk_data'] = self.pk_data
        ser_pubkey['pk_str'] = self.pk_str
        ser_pubkey['r1k_data'] = self.r1k_data
        return json.dumps(ser_pubkey)


class KeyPair():
    def __init__(self,conn=None,var_name=None,file_path=None):
        if(conn is None):
            self.conn = pyRserve.connect()
        else:
            self.conn = conn
        self.conn.r('library("HomomorphicEncryption")',void=True)

    def deserialize(self,pubkey_json,seckey_json=None):

        k_str = 'k'+str(random.randint(0,2**32))

        pubkey_obj = json.loads(pubkey_json)
        self.public_key = PublicKey(self.conn,k_str+'$pk',pubkey_obj['params'],pubkey_obj['r1k_data'],pubkey_obj['pk_data'])

        if(seckey_json is not None):
            seckey_obj = json.loads(seckey_json)
            self.secret_key = SecretKey(self.conn,k_str+'$sk',seckey_obj['params'],seckey_obj['r1k_data'],seckey_obj['sk_data'])
            filepath = self.create_file(self.public_key.params,self.public_key.r1k_data,self.public_key.pk_data,self.secret_key.sk_data)
        else:
            self.secret_key = None
            filepath = self.create_file(self.public_key.params,self.public_key.r1k_data,self.public_key.pk_data)

        self.conn.eval(self.public_key.pk_str[:-3]+' <- loadFHE(file="'+filepath+'")')
        os.remove(filepath)
        return (self.public_key,self.secret_key)



    def create_file(self,params,r1k,pk,sk=None):
        boiler = ['=> FHE pkg obj <=\n',
         'FandV_keys\n',
         '=> FHE pkg obj <=\n',
         'Rcpp_FandV_par\n']
        boiler2 = ['=> FHE pkg obj <=\n', 'Rcpp_FandV_rlk\n']
        boiler3 = ['=> FHE package object <=\n', 'Rcpp_FandV_pk\n']
        boiler4 = ['=> FHE package object <=\n', 'Rcpp_FandV_sk\n']

        # if no secret key is deserialized, make a fake one
        if(sk is None):
            out = "4097  "
            for x in (np.random.rand(4097)>0.5).astype('int'):
                out += str(x) + " "
            out[:-1]+"\n"
            sk = [out]

        raw = boiler + params + boiler2 + r1k + boiler3 + pk + boiler4 + sk

        tmp_file = tempfile.mktemp()
        f = open(tmp_file,'w')
        f.writelines(raw)
        f.close()

        return tmp_file


    def generate(self):

        lambd=90
        L=10

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

    def __sub__(self,y):
        vector_name='c'+str(random.randint(0,2**32))
        out = Scalar(self.conn,self.public_key)
        self.conn.eval(out.vector_name+' <- '+self.vector_name+' - '+y.vector_name)
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
