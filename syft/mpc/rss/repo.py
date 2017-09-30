from .scalar import MPCFixedPoint, MPCNatural
import numpy as np
import random

Q = 293973345475167247070445277780365744413


class MPCRepo(object):

    def __init__(self):
        self.ints = {}

    def set_siblings(self, a, b):
        self.siblings = list()

        self.left = a
        self.right = b

        self.siblings.append(a)
        self.siblings.append(b)

    def create_float(self, secret):
        return MPCFixedPoint(secret, self)

    def create_natural(self, secret):

        shares = self.share(secret)

        return self.create_natural_with_shares(shares)

    def create_natural_with_shares(self, shares):

        id = np.random.randint(0, 2**32)
        self.create_share(id, shares[0])
        self.siblings[0].create_share(id, shares[1])
        self.siblings[1].create_share(id, shares[2])
        return MPCNatural(id, self)

    def create_share(self, id, share):
        if(id not in self.ints.keys()):
            self.ints[id] = share
            return True  # everything is goodl
        else:
            return False  # hmm... non-unique int id

    def get_share(self, id):
        return self.ints[id]

    def add(self, new_id, id1, id2, populate_to_siblings=False):

        share = (self.ints[id1] + self.ints[id2]) % Q

        self.create_share(new_id, share)

        if(populate_to_siblings):
            for s in self.siblings:
                s.add(new_id, id1, id2)

        return MPCNatural(new_id, self)

    def sub(self, new_id, id1, id2, populate_to_siblings=False):

        share = (self.ints[id1] - self.ints[id2]) % Q

        self.create_share(new_id, share)

        if(populate_to_siblings):
            for s in self.siblings:
                s.sub(new_id, id1, id2)

        return MPCNatural(new_id, self)

    def share(self, secret):

        first = random.randrange(Q)
        second = random.randrange(Q)
        third = (secret - first - second) % Q

        return [first, second, third]

    def mult_local(self, id1, id2):

        x_0 = self.ints[id1]
        y_0 = self.ints[id2]

        x_1 = self.left.ints[id1]
        y_1 = self.left.ints[id2]

        z0 = ((x_0 * y_0) + (x_0 * y_1) + (x_1 * y_0)) % Q

        return self.create_natural(z0)

    def mult(self, new_id, id1, id2, populate_to_siblings=False):

        z0 = self.mult_local(id1, id2)
        z1 = self.left.mult_local(id1, id2)
        z2 = self.right.mult_local(id1, id2)

        return z0 + z1 + z2

    def mult_scalar(self, new_id, id1, scalar, populate_to_siblings=False):

        share = (self.ints[id1] * scalar) % Q

        self.create_share(new_id, share)

        if(populate_to_siblings):
            for s in self.siblings:
                s.mult_scalar(new_id, id1, scalar)

        return MPCNatural(new_id, self)

    def div_scalar(self, new_id, id1, scalar, populate_to_siblings=False):

        share = int(self.ints[id1] / scalar) % Q

        self.create_share(new_id, share)

        if(populate_to_siblings):
            for s in self.siblings:
                s.div_scalar(new_id, id1, scalar)

        return MPCNatural(new_id, self)
