class PyContainer:
    def __add__(self, other):
        return self.__add__(other)

    def __iter__(self):
        return self

    def __next__(self):
        return self.__next__()

    def __radd__(self, other):
        return self.__radd__(other)

    def __truediv__(self, other):
        return self / other

    def __rtruediv__(self, other):
        return other / self

    def __floordiv__(self, other):
        return self.__floordiv__(other)

    def __rfloordiv__(self, other):
        return self.__rfloordiv__(other)

    def __instancecheck__(self, instance):
        return self.__instancecheck__(instance)

    def __mul__(self, other):
        return self.__mul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self - other

    def __rsub__(self, other):
        return self.__rsub__(other)