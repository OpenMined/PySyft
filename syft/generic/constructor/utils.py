import inspect


def is_static_method(class_: type, attr: str):
    """Test if a value of a class is static method.

    Example:

        class MyClass(object):
            @staticmethod
            def method():
                ...

    Args:
        class_ (type): the class on which we want to check whether the method is statically implemented
        attr (str): the name of the method we want to check.

    Returns:
        bool: whether or not a method named <attr> is a static method of class <class_>
    """

    if not inspect.isclass(class_):
        return False

    if hasattr(class_, attr):
        value = getattr(class_, attr)
    else:
        return False

    assert getattr(class_, attr) == value

    for cls in inspect.getmro(class_):
        if inspect.isroutine(value):
            if attr in cls.__dict__:
                bound_value = cls.__dict__[attr]
                if isinstance(bound_value, staticmethod):
                    return True
    return False


def copy_static_methods(from_class: type, to_class: type):
    """Copies all static methods from one class to another class

    This utility was initialized during the creation of the Constructor for PyTorch's "th.Tensor" class. Since we
    replace each original constructor (th.Tensor) with on we implement (torch_.UppercaseTensorConstructor), we also
    need to make sure that our new constructor has any static methods which were previously stored on th.Tensor.
    Otherwise, the library might look for them there, not find them, and then trigger an error.

    Args:
        from_class (Type): the class on which we look for static methods co copy
        to_class (Type): the class onto which we copy all static methods found in <from_class>

    """
    # there are no static methods if from_class itself is not a type (sometimes funcs get passed in)

    for attr in dir(from_class):
        if is_static_method(class_=from_class, attr=attr):
            setattr(to_class, attr, getattr(from_class, attr))


def get_original_constructor_name(object_name: str):
    """Generate name for original constructor

    For each custom constructor, we move the original constructor to a consistent location relative to
    the original constructor so that each custom constructor automatically knows where to find the original
     method it is overloading. Namely, we move the original constructor to a different attr within the same
      module as the original constructor. This method specifies the naming convention that we use to name
      the original constructor when it is moved.

      Args:
          object_name (str): the original constructor's original name
    """

    return f"original_{object_name}"
