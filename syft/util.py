import inspect


def is_static_method(klass, attr, value=None):
    """Test if a value of a class is static method.

    Example:

        class MyClass(object):
            @staticmethod
            def method():
                ...

    Args:
        klass (type): the class on which we want to check whether the method is statically implemented
        attr (str): the name of the method we want to check.
        value (obj): the value of the attribute (i typically leave this empty)

    Returns:
        bool: whether or not a method named <attr> is a static method of class <klass>
    """

    if not inspect.isclass(klass):
        return False

    if value is None:
        if hasattr(klass, attr):
            value = getattr(klass, attr)
        else:
            return False

    assert getattr(klass, attr) == value

    for cls in inspect.getmro(klass):
        if inspect.isroutine(value):
            if attr in cls.__dict__:
                bound_value = cls.__dict__[attr]
                if isinstance(bound_value, staticmethod):
                    return True
    return False


def copy_static_methods(from_class, to_class):
    """Copies all static methods from one class to another class

    This utility was initialized during the creation of the Constructor for PyTorch's "th.Tensor" class. Since we
    replace each original constructor (th.Tensor) with on we implement (torch_.UppercaseTensorConstructor), we also
    need to make sure that our new constructor has any static methods which were previously stored on th.Tensor.
    Otherwise, the library might look for them there, not find them, and then trigger an error.

    Args:
        from_class (Type): the class on which we look for static methods co copy
        to_class (Type): the class onto which we copy all static methods found in <from_class>

    """

    print(f"copying static methods from:{from_class} to {to_class}")

    for attr in dir(from_class):
        if is_static_method(from_class, attr):
            setattr(to_class, attr, getattr(from_class, attr))


def get_original_constructor_name(object_name):
    return f"original_{object_name}"
