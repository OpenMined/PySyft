import numpy
import operator

ATTRIBUTES = "attributes"
METHODS = "methods"
OPERATORS = "operators"
LIST_OPERATORS = [x for x in dir(operator) if x.startswith("__") and x.endswith("__")] + ["__divmod__", "__idivmod__"]
LIST_OPERATORS += [x.replace("__i", "__r")
                   for x in LIST_OPERATORS
                   if x.startswith("__i") and x not in ['__index__', '__inv__', '__invert__']]

IGNORED_TYPES = ["<class 'NoneType'>", "<class 'type'>", "<class 'str'>"]


# attribute_classes = ["<class 'getset_descriptor'>", "<class 'PyCapsule'>"]

class ClassWrapperUtil(object):

    def __init__(self, class_to_wrap, ignored_types):
        self.class_to_wrap = class_to_wrap
        self.ignored_types = ignored_types

        self.__init_build_attribute_dict()

    def __init_build_attribute_dict(self):
        self.attr_dict = dict()
        self.attr_dict[ATTRIBUTES] = dict()
        self.attr_dict[METHODS] = dict()
        self.attr_dict[OPERATORS] = dict()

        for attr_name in dir(self.class_to_wrap):
            attr = getattr(self.class_to_wrap, attr_name)

            if str(type(attr)) in self.ignored_types:
                continue
            elif attr_name in LIST_OPERATORS:
                self.attr_dict[OPERATORS][attr_name] = True
            elif callable(attr):
                self.attr_dict[METHODS][attr_name] = True
            else:
                self.attr_dict[ATTRIBUTES][attr_name] = True

    def register_attributes(self, wrapper_class):
        # TODO: Document https://stackoverflow.com/a/8294654

        for attr_name in self.attr_dict[METHODS].keys():
            if not hasattr(wrapper_class, attr_name):
                setattr(wrapper_class, attr_name, lambda *x: (_ for _ in ()).throw(NotImplementedError(attr_name)))

        for attr_name in self.attr_dict[ATTRIBUTES].keys():
            if not hasattr(wrapper_class, attr_name):
                setattr(wrapper_class, attr_name, lambda *x: (_ for _ in ()).throw(NotImplementedError(attr_name)))

    """Auxiliary methods
    Methods used to check what is left from wrapping classes
    """

    def __print_missing(self, wrapped_class, dict_, attr_type):
        dict_ = dict_.copy()

        for name in dir(wrapped_class):
            if name in dict_:
                del dict_[name]
            else:
                # print(f"{name} not in {attr_type}")
                pass

        if len(dict_.keys()) == 0:
            print(f"No missing {attr_type}")
        else:
            print(f"Missing {attr_type}:")
            for name in dict_:
                print(f"\t{name}")

    def missing(self, wrapped_class, attr_type=None):
        valid = [*self.attr_dict.keys(), None]

        if attr_type not in valid:
            raise ValueError(f"Input parameter 'attr_type' must be one of {valid[:-1]}.")

        if attr_type is None:
            for attr_type in self.attr_dict:
                self.__print_missing(wrapped_class, self.attr_dict[attr_type], attr_type)
            return

        self.__print_missing(wrapped_class, self.attr_dict[attr_type], attr_type)


ndarrayWrapperUtil = ClassWrapperUtil(numpy.ndarray, IGNORED_TYPES)
class WrappedNdarray(object):
    _array = []

    def __init__(self, array_, wrapper_util=ndarrayWrapperUtil):
        self._array = array_

        # Add same attributes as the hooked array
        for attribute_name in wrapper_util.attr_dict[ATTRIBUTES]:
            setattr(self, attribute_name, getattr(self._array, attribute_name))

        # Add same methods as the hooked array
        for method_name in wrapper_util.attr_dict[METHODS]:
            setattr(self, method_name, self.method_wrapper(self._array, method_name))

    # Overload operators
    # TODO: Overload at class level to automatize it
    def __abs__(self, o):
        return self._array.__abs__(o)

    def __add__(self, o):
        return self._array.__add__(o)

    def __and__(self, o):
        return self._array.__and__(o)

    def __contains__(self, o):
        return self._array.__contains__(o)

    def __delitem__(self, o):
        return self._array.__delitem__(o)

    def __divmod__(self, o):
        return self._array.__divmod__(o)

    def __floordiv__(self, o):
        return self._array.__floordiv__(o)

    def __getitem__(self, o):
        return self._array.__getitem__(o)

    def __iadd__(self, o):
        return self._array.__iadd__(o)

    def __iand__(self, o):
        return self._array.__iand__(o)

    def __ifloordiv__(self, o):
        return self._array.__ifloordiv__(o)

    def __ilshift__(self, o):
        return self._array.__ilshift__(o)

    def __imatmul__(self, o):
        return self._array.__imatmul__(o)

    def __imod__(self, o):
        return self._array.__imod__(o)

    def __imul__(self, o):
        return self._array.__imul__(o)

    def __index__(self, o):
        return self._array.__index__(o)

    def __invert__(self, o):
        return self._array.__invert__(o)

    def __ior__(self, o):
        return self._array.__ior__(o)

    def __ipow__(self, o):
        return self._array.__ipow__(o)

    def __irshift__(self, o):
        return self._array.__irshift__(o)

    def __isub__(self, o):
        return self._array.__isub__(o)

    def __itruediv__(self, o):
        return self._array.__itruediv__(o)

    def __ixor__(self, o):
        return self._array.__ixor__(o)

    def __lshift__(self, o):
        return self._array.__lshift__(o)

    def __matmul__(self, o):
        return self._array.__matmul__(o)

    def __mod__(self, o):
        return self._array.__mod__(o)

    def __mul__(self, o):
        return self._array.__mul__(o)

    def __neg__(self, o):
        return self._array.__neg__(o)

    def __or__(self, o):
        return self._array.__or__(o)

    def __pos__(self, o):
        return self._array.__pos__(o)

    def __pow__(self, o):
        return self._array.__pow__(o)

    def __radd__(self, o):
        return self._array.__radd__(o)

    def __rand__(self, o):
        return self._array.__rand__(o)

    def __rdivmod__(self, o):
        return self._array.__rdivmod__(o)

    def __rfloordiv__(self, o):
        return self._array.__rfloordiv__(o)

    def __rlshift__(self, o):
        return self._array.__rlshift__(o)

    def __rmatmul__(self, o):
        return self._array.__rmatmul__(o)

    def __rmod__(self, o):
        return self._array.__rmod__(o)

    def __rmul__(self, o):
        return self._array.__rmul__(o)

    def __ror__(self, o):
        return self._array.__ror__(o)

    def __rpow__(self, o):
        return self._array.__rpow__(o)

    def __rrshift__(self, o):
        return self._array.__rrshift__(o)

    def __rshift__(self, o):
        return self._array.__rshift__(o)

    def __rsub__(self, o):
        return self._array.__rsub__(o)

    def __rtruediv__(self, o):
        return self._array.__rtruediv__(o)

    def __rxor__(self, o):
        return self._array.__rxor__(o)

    def __setitem__(self, o):
        return self._array.__setitem__(o)

    def __sub__(self, o):
        return self._array.__sub__(o)

    def __truediv__(self, o):
        return self._array.__truediv__(o)

    def __xor__(self, o):
        return self._array.__xor__(o)

    def __str__(self):
        return self._array.__str__()

    def __repr__(self):
        return self._array.__repr__()

    """ Utility methods
    Methods used during the automatic overload of attributes and methods
    """

    def method_wrapper(self, obj, method_name):
        def inner(*args, **kwargs):
            new_args = list(args)
            for i in range(len(args)):
                if isinstance(args[i], type(self)):
                    new_args[i] = args[i]._array

            output = getattr(obj, method_name)(*new_args, **kwargs)

            if type(output) == type(self):
                output = self(output)

            return output

        return inner

# Register ndarray attributes so they can be read during hooking process
ndarrayWrapperUtil.register_attributes(WrappedNdarray)


numpyWrapperUtil = ClassWrapperUtil(numpy, IGNORED_TYPES)
class WrappedNumpy_(object):

    __name__ = "numpy"

    def __init__(self, np_, wrapper_util = numpyWrapperUtil):
        self.WrappedNdarray = WrappedNdarray
        self._np = np_

        # Add same attributes as the hooked array
        for attribute_name in wrapper_util.attr_dict[ATTRIBUTES]:
            setattr(self, attribute_name, getattr(self._np, attribute_name))

        # Add same methods as the hooked array
        for method_name in wrapper_util.attr_dict[METHODS]:
            setattr(self, method_name, self.method_wrapper(self._np, method_name))

    # Overload operators
    pass

    """ Utility methods
    Methods used during the automatic overload of attributes and methods
    """

    def method_wrapper(self, obj, method_name):
        def inner(*args, **kwargs):
            new_args = list(args)
            for i in range(len(args)):
                if isinstance(args[i], WrappedNdarray):
                    new_args[i] = args[i]._array

            output = getattr(obj, method_name)(*new_args, **kwargs)

            if type(output) == obj.ndarray:
                output = WrappedNdarray(output)

            return output

        return inner

# Register numpy attributes so they can be read during hooking process
#numpyWrapperUtil.register_attributes(WrappedNumpy)
WrappedNumpy = WrappedNumpy_(numpy)
