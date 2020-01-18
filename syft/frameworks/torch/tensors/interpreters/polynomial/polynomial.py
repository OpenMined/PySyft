from syft.frameworks.torch.tensors.interpreters.polynomial.term import Term


class Polynomial:
    def __init__(self, terms, constant=0, minimum_factor=0):
        if isinstance(terms, Term):
            terms = list([terms])
        self.additive_terms = terms
        self.additive_constant = constant
        self.minimum_factor = minimum_factor

    def __pow__(self, factor):
        assert isinstance(factor, int)
        if factor == 0:
            return 1
        elif factor == 1:
            return self
        elif factor > 1:
            out = self
            for i in range(factor - 1):
                out = out * self
            return out
        else:
            raise Exception("Illegal factor value:" + str(factor))

    def __mul__(self, other):

        if isinstance(other, Polynomial):
            new_terms = list()
            for self_term in self.additive_terms:
                for other_term in other.additive_terms:
                    new_terms.append(self_term * other_term)

            core = Polynomial(
                terms=new_terms,
                constant=0,
                minimum_factor=min(self.minimum_factor, other.minimum_factor),
            )
            with_left = core + (self * other.additive_constant)
            with_right = with_left + (other * self.additive_constant)
            with_right.additive_constant = with_right.additive_constant / 2
            return with_right

        new_terms = list()
        for self_term in self.additive_terms:
            new_terms.append(self_term * other)

        return Polynomial(
            terms=new_terms,
            constant=self.additive_constant * other,
            minimum_factor=self.minimum_factor,
        ).reduce()

    def __add__(self, other):
        try:
            if isinstance(other, Polynomial):
                return Polynomial(
                    self.additive_terms + other.additive_terms,
                    constant=self.additive_constant + other.additive_constant,
                    minimum_factor=min(self.minimum_factor, other.minimum_factor),
                ).reduce()
            else:
                return Polynomial(
                    self.additive_terms,
                    constant=self.additive_constant + other,
                    minimum_factor=self.minimum_factor,
                ).reduce()
        except Exception as e:
            print(e)
            print(self)
            print(other)
            raise Exception("error")

    def __call__(self, **kwargs):
        result = 0
        for term in self.additive_terms:
            result += term(**kwargs)
        return result + self.additive_constant

    def reduce(self):
        term2factor = {}
        for term in self.additive_terms:
            key = term

            if key not in term2factor:
                term2factor[key] = 0

            term2factor[key] += term.factor

        new_additive_terms = list()
        for term, factor in term2factor.items():
            if abs(factor) > self.minimum_factor:
                new_term = Term(factor=factor, variables=term.multiplicative_variables)
                new_additive_terms.append(new_term)

        self.additive_terms = new_additive_terms
        return self

    def __str__(self):
        out = ""
        for term in self.additive_terms:
            out += str(term) + " + "
        if self.additive_constant == 0:
            return out[:-2]
        else:
            return out[:-2] + " + " + str(self.additive_constant)

    def __repr__(self):
        return str(self)
