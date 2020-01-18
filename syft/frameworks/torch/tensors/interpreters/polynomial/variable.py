class Variable:

    def __init__(self, name, exponent=1):
        # if ("^" in name):
        #     raise Exception("You cannot use ^ in variable names.")
        self.name = name
        self.exponent = exponent
        if (exponent == 0):
            raise Exception("You cannot create a variable with no exponent")

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        if (self.exponent == 1):
            return str(self.name)
        return str(self.name) + "^" + str(self.exponent)

    def __repr__(self):
        return str(self)