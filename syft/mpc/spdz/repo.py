from .scalar import MPCFixedPoint, MPCNatural
import numpy as np
import random

Q = 293973345475167247070445277780365744413


class MPCRepo(object):

    def __init__(self):
        self.ints = {}

    def set_parties(self, a):
        self.another_party = [a]

    def create_float(self, secret):
        return MPCFixedPoint(secret, self)

    def create_natural(self, secret):

        shares = self.share(secret)

        return self.create_natural_with_shares(shares)

    def create_natural_with_shares(self, shares):

        id = np.random.randint(0, 2**32)
        self.create_share(id, shares[0])
        self.another_party[0].create_share(id, shares[1])
        return MPCNatural(id, self)

    def create_share(self, id, share):
        if(id not in self.ints.keys()):
            self.ints[id] = share
            return True  # everything is good
        else:
            return False  # hmm... non-unique int id

    def get_share(self, id):
        return self.ints[id]

    def add(self, new_id, id1, id2, populate_to_another_party=False):

        share = (self.ints[id1] + self.ints[id2]) % Q

        self.create_share(new_id, share)

        if(populate_to_another_party):
            self.another_party[0].add(new_id, id1, id2)

        return MPCNatural(new_id, self)

    def add_public(self, new_id, id1, scalar, populate_to_another_party=False):

        share = (self.ints[id1] + scalar) % Q

        self.create_share(new_id, share)

        if(populate_to_another_party):
            self.another_party[0].add_public(new_id, id1, 0)

        return MPCNatural(new_id, self)

    def sub(self, new_id, id1, id2, populate_to_another_party=False):

        share = (self.ints[id1] - self.ints[id2]) % Q

        self.create_share(new_id, share)

        if(populate_to_another_party):
            self.another_party[0].sub(new_id, id1, id2)

        return MPCNatural(new_id, self)

    def sub_public(self, new_id, id1, scalar, populate_to_another_party=False):

        share = (self.ints[id1] - scalar) % Q

        self.create_share(new_id, share)

        if(populate_to_another_party):
            self.another_party[0].sub(new_id, id1, 0)

        return MPCNatural(new_id, self)

    def share(self, secret):

        x0 = random.randrange(Q)
        x1 = (secret - x0) % Q

        return [x0, x1]

    def reconstruct(self, shares):
        return sum(shares) % Q

    def generate_multiplication_triple(self):
        a = random.randrange(Q)
        b = random.randrange(Q)
        c = a * b % Q
        return self.create_natural_with_shares(self.share(a)), self.create_natural_with_shares(self.share(b)), self.create_natural_with_shares(self.share(c))

    def truncate(self, x, amount=8):
        self.ints[x.id] = self.ints[x.id] // 10**amount
        self.another_party[0].ints[x.id] = Q - ((Q - self.another_party[0].ints[x.id]) // 10**amount)
        return x

    def mult(self, new_id, x, y, populate_to_another_party=False):

        a, b, c = self.generate_multiplication_triple()

        new_id1 = np.random.randint(0, 2**32)
        d = self.sub(new_id1, x, a.id, True)
        new_id2 = np.random.randint(0, 2**32)
        e = self.sub(new_id2, y, b.id, True)

        delta = self.reconstruct([self.ints[d.id], self.another_party[0].ints[d.id]])
        epsilon = self.reconstruct([self.ints[e.id], self.another_party[0].ints[e.id]])

        r = delta * epsilon % Q
        new_id3 = np.random.randint(0, 2**32)
        s = self.mult_public(new_id3, a.id, epsilon, True)
        new_id4 = np.random.randint(0, 2**32)
        t = self.mult_public(new_id4, b.id, delta, True)

        new_id5 = np.random.randint(0, 2**32)
        new_id6 = np.random.randint(0, 2**32)
        new_id7 = np.random.randint(0, 2**32)
        result = self.add(new_id7, s.id, self.add(new_id6, t.id, self.add_public(new_id5, c.id, r, True).id, True).id, True)

        return self.truncate(result)

    def mult_public(self, new_id, id1, scalar, populate_to_another_party=False):

        share = (self.ints[id1] * scalar) % Q

        self.create_share(new_id, share)

        if(populate_to_another_party):
            self.another_party[0].mult_public(new_id, id1, scalar)

        return MPCNatural(new_id, self)

    def div_public(self, new_id, id1, scalar, populate_to_another_party=False):

        share = int(self.ints[id1] / scalar) % Q

        self.create_share(new_id, share)

        if(populate_to_another_party):
            self.another_party[0].div_public(new_id, id1, scalar)

        return MPCNatural(new_id, self)
