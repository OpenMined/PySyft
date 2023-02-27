# based on https://arxiv.org/pdf/2004.00010.pdf
from fractions import Fraction  # we will work with rational numbers
import secrets  # use cryptographically secure library


def discrete_gaussian(sigma2, rng=None):
    """
    sample from a discrete Gaussian distribution N_Z(0,sigma2)
    Returns integer x with Pr[x] = exp(-x^2/(2*sigma2))/normalizing_constant(sigma2)
    mean 0 variance ~= sigma2 for large sigma2
    casts sigma2 to Fraction
    assumes sigma2>=0
    """
    if rng is None:
        rng = secrets.SystemRandom()
    sigma2 = Fraction(sigma2)
    if sigma2 == 0:
        return 0
    if sigma2 < 0:
        raise ValueError("sigma2 must be a non-negative value")
    t = floorsqrt(sigma2) + 1
    while True:
        candidate = sample_dlaplace(t, rng=rng)
        bias = ((abs(candidate) - sigma2 / t) ** 2) / (2 * sigma2)
        if sample_bernoulli_exp(bias, rng) == 1:
            return candidate


def floorsqrt(x):
    """
    compute floor(sqrt(x)) exactly
    only requires comparisons between x and integer
    """
    if x < 0:
        raise ValueError("x must be a non-negative value")
    a = 0
    b = 1
    while b * b <= x:
        b = 2 * b
    while a + 1 < b:
        c = (a + b) // 2
        if c * c <= x:
            a = c
        else:
            b = c
    return a


def sample_dlaplace(scale, rng=None):
    """
    sample from a discrete Laplace(scale) distribution
    Returns integer x with Pr[x] = exp(-abs(x)/scale)*(exp(1/scale)-1)/(exp(1/scale)+1)
    casts scale to Fraction
    assumes scale>=0
    """
    if rng is None:
        rng = secrets.SystemRandom()
    scale = Fraction(scale)
    if scale < 0:
        raise ValueError("scale must be a non-negative value")
    while True:
        sign = sample_bernoulli(Fraction(1, 2), rng)
        magnitude = sample_geometric_exp_fast(1 / scale, rng)
        if sign == 1 and magnitude == 0:
            continue
        return magnitude * (1 - 2 * sign)


def sample_uniform(m, rng):
    """
    sample uniformly from range(m)
    all randomness comes from calling this
    """
    if not isinstance(m, int):
        raise TypeError("m must be an integer value")
    if m <= 0:
        raise ValueError("m must be a positive value")
    return rng.randrange(m)

def sample_bernoulli(p, rng):
    """
    sample from a Bernoulli(p) distribution
    assumes p is a rational number in [0,1]
    """
    if not isinstance(p, Fraction):
        raise TypeError("p must be a Fraction value")
    if p < 0 or p > 1:
        raise ValueError("p must be a Fraction value in the range [0, 1]")
    
    m = sample_uniform(p.denominator, rng)
    if m < p.numerator:
        return 1
    else:
        return 0


def sample_bernoulli_exp1(x, rng):
    """
    sample from a Bernoulli(exp(-x)) distribution
    assumes x is a rational number in [0,1]
    """

    if not isinstance(x, Fraction):
        raise TypeError("x must be a Fraction value")
    if x < 0 or x > 1:
        raise ValueError("x must be a Fraction value in the range [0, 1]")
    k = 1
    while True:
        if sample_bernoulli(x / k, rng) == 1:
            k = k + 1
        else:
            break
    return k % 2


def sample_bernoulli_exp(x, rng):
    """
    sample from a Bernoulli(exp(-x)) distribution
    assumes x is a rational number >=0
    """
    if not isinstance(x, Fraction):
        raise TypeError("x must be a Fraction value")
    if x < 0:
        raise ValueError("x must be a greater than 0")
    
    while x > 1:
        if sample_bernoulli_exp1(Fraction(1, 1), rng) == 1:
            x = x - 1
        else:
            return 0
    return sample_bernoulli_exp1(x, rng)


def sample_geometric_exp_slow(x, rng):
    """
    sample from a geometric(1-exp(-x)) distribution
    assumes x is a rational number >= 0
    """
    if not isinstance(x, Fraction):
        raise TypeError("x must be a Fraction value")
    if x < 0:
        raise ValueError("x must be a greater than 0")
    
    k = 0
    while True:
        if sample_bernoulli_exp(x, rng) == 1:
            k = k + 1
        else:
            return k


def sample_geometric_exp_fast(x, rng):
    """
    sample from a geometric(1-exp(-x)) distribution
    assumes x >= 0 rational
    """
    if not isinstance(x, Fraction):
        raise TypeError("x must be a Fraction value")
    if x < 0:
        raise ValueError("x must be a greater than 0")

    t = x.denominator
    while True:
        u = sample_uniform(t, rng)
        b = sample_bernoulli_exp(Fraction(u, t), rng)
        if b == 1:
            break
    v = sample_geometric_exp_slow(Fraction(1, 1), rng)
    value = v * t + u
    return value // x.numerator
