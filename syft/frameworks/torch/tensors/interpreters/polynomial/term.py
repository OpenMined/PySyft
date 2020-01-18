from syft.frameworks.torch.tensors.interpreters.polynomial.variable import Variable

class Term:

    def __init__(self, variables, factor=1):
        self.factor = factor

        if not isinstance(variables, (list, tuple)):
            variables = list([variables])

        if isinstance(variables, tuple):
            variables = list(variables)

        variables.sort()

        self.multiplicative_variables = variables

    def __mul__(self, other):

        if (isinstance(other, Term)):

            new_factor = self.factor * other.factor

            var2exp = {}

            for var in self.multiplicative_variables + other.multiplicative_variables:

                key = var.name
                if (key not in var2exp):
                    var2exp[key] = 0

                var2exp[key] += var.exponent

            new_variables = list()
            for name, exp in var2exp.items():
                new_variables.append(Variable(name=name, exponent=exp))

            return Term(factor=new_factor, variables=new_variables)

        else:

            return Term(factor=self.factor * other, variables=self.multiplicative_variables)

    def __hash__(self):
        h = 1
        for v in self.multiplicative_variables:
            h *= (hash(v.name) * 12345) * v.exponent
        return h

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __call__(self, ignore_factor=False, **kwargs):
        result = 1
        for var in self.multiplicative_variables:
            if (var.name not in kwargs):
                return 0
            else:
                result *= pow(kwargs[var.name], var.exponent)

        if(ignore_factor):
            return result

        return result * self.factor

    def __str__(self):

        variables = ""
        for v in self.multiplicative_variables:
            variables += str(v)
        if (self.factor != 1):
            return str(self.factor) + variables
        else:
            return variables

    def __repr__(self):
        return str(self)