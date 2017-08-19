class Integer():

    def __init__(self,public_key,data=None):
        """Wraps pointer to encrypted integer with an interface that numpy can use."""

        self.public_key = public_key
        if(data is not None):
            self.data = self.public_key.pk.encrypt(data)
        else:
            self.data = None

    def __add__(self,y):
        """Adds two encrypted integers together."""

        out = Integer(self.public_key,None)
        out.data = self.data + y.data
        return out

    def __sub__(self,y):
        """Subtracts two encrypted integers."""

        out = Integer(self.public_key, None)
        out.data = self.data - y.data
        return out

    def __mul__(self,y):
        """Multiplies two integers. y may be encrypted or a simple integer."""

        if(type(y) == type(self)):
            out = Integer(self.public_key, None)
            out.data = self.data * y.data
            return out
        elif(type(y) == int or type(y) == float):
            out = Integer(self.public_key, None)
            out.data = self.data * y
            return out
        else:
            return None

    def __truediv__(self,y):
        """Divides two integers. y may be encrypted or a simple integer."""

        if(type(y) == type(self)):
            out = Integer(self.public_key, None)
            out.data = self.data / y.data
            return out
        elif(type(y) == int):
            out = Integer(self.public_key, None)
            out.data = self.data / y
            return out
        else:
            return None

    def __repr__(self):
        """This is kindof a boring/uninformative __repr__"""

        return 'e'

    def __str__(self):
        """This is kindof a boring/uninformative __str__"""

        return 'e'
