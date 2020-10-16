class Any:

    def __iter__(self):
        return self

    def __next__(self):
        return self.__next__()

    def __add__(self, other):
        return self + other

    def __radd__(self, other):
        return other + self

    def __truediv__(self, other):
        return self / other

    def __rtruediv__(self, other):
        return other / self

    def __floordiv__(self, other):
        return self / other

    def __rfloordiv__(self, other):
        return self / other

    def __mul__(self, other):
        return self * other

    def __rmul__(self, other):
        return other * self

    def __sub__(self, other):
        return self - other

    def __rsub__(self, other):
        return other - self