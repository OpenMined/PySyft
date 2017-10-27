from bv_utils import *

DEFAULT_PARAMETERS = {'n': 2048, 'modulus_size': 61, 't': 2048, 'sigma': 8}

def generate_key_pair(n = DEFAULT_PARAMETERS['n'], modulus_size = DEFAULT_PARAMETERS['modulus_size'], t = DEFAULT_PARAMETERS['t'], sigma = DEFAULT_PARAMETERS['sigma']):
    '''
    Generates a public-private key pair.
    n and q together define the ciphertext ring.
    n and t together define the plaintext ring.
    
    Args:
        n (int): degree of polynomial x^n + 1 bounding ciphertexts
        modulus_size (int): modulus q in bits, defines ciphertext ring
        t (int): modulus t, defines plaintext ring
        sigma (int): standard deviation of the error distribution
    '''
    
    #parameter initialization
    params = Parameters(n, modulus_size, t, sigma)
    
    private_key = PrivateKey(params)
    public_key = PublicKey(private_key, params)
    
    return (private_key, public_key)


class PrivateKey(object):
    '''
    Contains a private key and its decryption method.
    
    Args:
        params (:class:`Parameters`): parameters used to define the secret key
    
    Attributes:
        sk (:class:`list`): The value of the private key
        params (:class:`Parameters`): parameters
    '''
    def __init__(self, params):
        self.sk = sample_error_polynomial(params.n, params.sigma)
        self.params = params
    
    def decrypt(self, ct):
        '''
        Returns decrypted plaintext of ciphertext.
        
        Args:
            ct (:class:`Ciphertext`): ciphertext to be decrypted
        '''
        m = ct.ct[0]
        powers_of_s = self.sk
        for i in range(1, len(ct.ct)):
            m = ring_add(m, ring_mult(ct.ct[i], powers_of_s, self.params.f, self.params.q), self.params.f, self.params.q)
            powers_of_s = ring_mult(powers_of_s, self.sk, self.params.f, self.params.q)
        q_by_2 = self.params.q // 2
        for i in range(len(m)):
            if m[i] > q_by_2:
                m[i] = m[i] - self.params.q
            m[i] = m[i] % self.params.t
        return m


class PublicKey(object):
    '''
    Contains a public key and its encryption method.
    
    Args:
        sk (:class:`PrivateKey`): The secret key for which the public key will be defined
        params (:class:`Parameters`): part of public key
    
    Attributes:
        pk (tuple: two instances of :class:`list`): The value of secret key
        params (:class:`Parameters`): parameters
    '''
    def __init__(self, secret_key, params):
        self.pk = generate_public_key(secret_key.sk, params)
        self.params = params
    
    def encrypt(self, pt):
        '''
        Encrypts the integer pt using public key pk
        
        Args:
            pt (:class:`list`): The integer/integers to be encrypted
        '''
        if len(pt) > self.params.n:
            raise TypeError('Expected integer array of size < %d' % self.params.n)
        m = pt
        m.extend([0]*(self.params.n - len(pt)))
        u = sample_error_polynomial(self.params.n, self.params.sigma)
        f = sample_error_polynomial(self.params.n, self.params.sigma)
        g = sample_error_polynomial(self.params.n, self.params.sigma)
        ct = [None, None]
        ct[0] = ring_mult(self.pk[0], u, self.params.f, self.params.q)
        for i in range(len(g)):
            g[i] = self.params.t * g[i] + m[i]
        ct[0] = ring_add(ct[0], g, self.params.f, self.params.q)
        ct[1] = ring_mult(self.pk[1], u, self.params.f, self.params.q)
        for i in range(len(f)):
            f[i] = self.params.t * f[i]
        ct[1] = ring_add(ct[1], f, self.params.f, self.params.q)
        for i in range(len(ct[1])):
            ct[1][i] = -1 * ct[1][i]
        ct = tuple(ct)
        return ct


class Ciphertext(object):
    '''
    Contains encryption of a plaintext i.e. list of integers.
    
    Args:
        data (:class:`list`/tuple: two instances of :class:`list`): data to be encrypted/stored
        pk (:class:`PublicKey`): public key associated with the ciphertext
        degree (int): degree of the ciphertext (default: 1)
        encrypted (bool): data is encrypted? (default: False)
        
    Attributes:
        ct (tuple: two instances of :class:`list`): encrypted value
        pk (:class:`PublicKey`): public key associated with the ciphertext
    '''
    def __init__(self, data, pk, degree=1, encrypted=False):
        self.pk = pk
        self.degree = degree
        if encrypted:
            self.ct = data
        else:
            self.ct = pk.encrypt(data)

    def __add__(self, other):
        '''
        Returns E(a + b), given self = E(a) and other = E(b)
        
        Args:
            other (:class:`Ciphertext`/:class:`list`): Ciphertext/Plaintext to be added to `self`
        '''
        if type(other) == int:
            other = ([other],)
            other = Ciphertext(other, self.pk, 0, True)
        if type(other) == list:
            other = (other, )
            other = Ciphertext(other, self.pk, 0, True)
        if self.degree > other.degree:
            ct = list(self.ct)
            for i in range(len(other.ct)):
                ct[i] = ring_add(self.ct[i], other.ct[i], self.pk.params.f, self.pk.params.q)
            ct = tuple(ct)
        else:
            ct = list(other.ct)
            for i in range(len(self.ct)):
                ct[i] = ring_add(self.ct[i], other.ct[i], self.pk.params.f, self.pk.params.q)
            ct = tuple(ct)
        return Ciphertext(ct, self.pk, max(self.degree, other.degree), True)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        '''
        Returns E(a * b), given self = E(a) and other = E(b)
        
        Args:
            other (:class:`Ciphertext`/:class:`list`): Ciphertext/Plaintext to be added to `self`
        '''
        if type(other) == int:
            other = ([other],)
            other = Ciphertext(other, self.pk, 0, True)
        if type(other) == list:
            other = (other,)
            other = Ciphertext(other, self.pk, 0, True)
        ct = [None for i in range(self.degree + other.degree + 1)]
        for i in range(len(self.ct)):
            for j in range(len(other.ct)):
                if ct[i + j] == None:
                    ct[i + j] = ring_mult(self.ct[i], other.ct[j], self.pk.params.f, self.pk.params.q)
                else:
                    temp = ring_mult(self.ct[i], other.ct[j], self.pk.params.f, self.pk.params.q)
                    ct[i + j] = ring_add(ct[i + j], temp, self.pk.params.f, self.pk.params.q)
        return Ciphertext(ct, self.pk, self.degree + other.degree, True)

    def __rmul__(self, other):
        return self.__mul__(other)


sk, pk = generate_key_pair()
pt_1 = [1, 4, 6]
pt_2 = [1, 1, 1, 1]
ct_1 = Ciphertext(pt_1, pk)
ct_2 = Ciphertext(pt_2, pk)
ct = ct_1 * ct_2
print(sk.decrypt(ct))
