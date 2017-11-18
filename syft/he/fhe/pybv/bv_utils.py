import sympy, sympy.stats
import gmpy2
import random
from PyPari.PyPari import *

x = sympy.Symbol('x')

def get_prime(N):
    '''
    Returns a random N-bit prime number
    
    Args:
        N (int): bit_size of the required prime
    '''

    rand = random.SystemRandom()
    p = gmpy2.mpz(rand.getrandbits(N))
    p = gmpy2.bit_set(p, N - 1)
    return int(gmpy2.next_prime(p))


def sample_random_polynomial(n, N):
    '''
    Returns a polynomial of degree n - 1 with coefficient of bit_size N
    
    Args:
        n (int): size of sample/ (degree of polynomial + 1)
        N (int): bit_size of random numbers
    '''
    rand = random.SystemRandom()
    A = []
    for i in range(n):
        A.append(int(gmpy2.mpz(rand.getrandbits(N))))
    return A


def sample_error_polynomial(n, sigma):
    '''
    Generates a secret key instance.
    
    Args:
        n (int): size of sample
        sigma (int): standard deviation of error distribution
    '''
    e = []
    #error distribution
    chi = sympy.stats.Normal(x, 0, sigma)
    for i in range(n):
        e.append(int(sympy.stats.sample(chi)))
    return e


def generate_public_key(sk, params):
    '''
    Generates a public key for the given secret key and parameters
    
    Args:
        sk (:class:`list`): given secret key
        params (:class:`Parameters`): given parameters
    '''
    a = []
    e = sample_error_polynomial(params.n, params.sigma)
    a.append(sample_random_polynomial(params.n, params.modulus_size))
    r = ring_mult(sk, a[0], params.f, params.q)
    for i in range(len(e)):
        e[i] = params.t * e[i]
    a.insert(0, ring_add(e, r, params.f, params.q))
    a = tuple(a)
    return a


class Parameters:
    '''
    Initializes the parameters for the given input arguments.
    n and q together define the ciphertext ring. 
    n and t together define the plaintext ring.
    
    Args:
        n (int): degree of polynomial x^n + 1 bounding ciphertexts
        modulus_size (int): modulus q in bits, defines ciphertext ring
        t (int): modulus t, defines plaintext ring
        sigma (int): standard deviation of the error distribution
        
    Attributes:
        n (int): as defined above
        t (int): as defined above
        sigma (int): as defined above
        modulus_size (int): as defined above
        f (:class:`list`): Polynomial x^n + 1
        q (int): modulus q
    '''
    def __init__(self, n, modulus_size, t, sigma):
        self.n = n
        self.t = t
        self.sigma = sigma
        self.modulus_size = modulus_size
        F = [0] * (n - 1)
        F.insert(0, 1)
        F.append(1)
        self.f = F
        temp = 0
        while temp != 1:
            q = get_prime(modulus_size)
            temp = gmpy2.f_mod(q, 2 * n)
        self.q = q

    def __str__(self):
        return 'n: ' + str(self.n) + ', q: ' + str(self.q) + ', t: ' + str(self.t) + ', sigma: ' + str(self.sigma)
