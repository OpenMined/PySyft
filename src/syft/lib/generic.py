from typing import Callable
from typing import Any
from typing import Tuple
from typing import Type
from typing import Optional

from .util import get_original_constructor_name
from .util import copy_static_methods
from .util import replace_classes_in_module
from ..core.common.uid import UID

import inspect
import sys


class ObjectConstructor(object):
    """Syft allows for the extension and remote execution of a range of python libraries. As such,
    a common need is the ability to modify library-specific constructors of objects.

    For example in the PyTorch framework we might wish to:
     - extend the th.Tensor constructor with new attributes such as "id" and "owner".
     - add additional functionality before/after th.Tensor() such as object registration (or memory re-use)

    However, as most deep learning frameworks are Python code wrapping C++, they often have very "locked down"
    __init__ and __new__ method definitions, which makes it difficult to modify their constructors in the
    conventional way. Furthermore, most frameworks (including Tensorflow, PyTorch, and NumPy), occasionally
    have multiple ways of constructing the same object (tensorflow.constant vs tensorflow.Tensor, and
    torch.Tensor vs torch.tensor) being two examples. Note that in each of these examples, one of the ways
    is the actual TYPE of tensor (th.Tensor) while the other is merely a method which can be used to create
    the tensor (th.tensor).

    As many constructors are difficult to overload (or perhaps have differing ways to overload them), we
    must use a more creative approach to augmenting constructor functionality. Instead of overloading __new__
    or __init__, we create a series of 3 methods (pre_init, init, and post_init) within
    this ObjectConstructor class. Then, when you call __call__ on this class, it initializes the object
    using all three methods. Once we have defined this class, we then take a 2-step approach to *install* this class
    within an existing library.

    1) COPY: we copy the existing constructor to a "original_<name>" location. For example:

            th.original_Tensor = th.Tensor

    2) REPLACE: we then replace the original class path name with an instance of this constructor. For example:

            th.Tensor = TensorConstructor() #where TensorConstructor is an instance of ObjectConstructor

    This allows us to have a SINGLE way of implementing the following functionality for any object and installing it
    into the library (ensuring that all instances of that object use our constructor, even instances created from
    within the framework itself):

        - arbitrary custom args and kwargs,
        - arbitrary arg and kwarg manipulation
        - arbitrary functionality before the underlying (original) object constructor is called
        - arbitrary functionality after the underlying (original) object constructor is called.

    Thus, if any object has its functionality extended or overridden by PySyft, it should be created using
    an extension of this class.

    GARBAGE COLLECTION NOTES:z

    There is a special case in codebases which wrap C++ functionality wherein an object can be created on the
    C++ side, a Python wrapper is created (calling our constructor above). However, some methods can destroy
    and then re-create the Python wrapper without destroying or re-creating the underlying tensor. This can be very
    tricky to deal with for a variety of reasons. However, we have observed that we can override this ability by:

    - caching the Python object somewhere so that the original python object doesn't get destroyed until it should be
    - intelligently remembering that this object has been created before when the constructor is called for the
        second time. This assume we can have stable "ID"s across creations of the tensor.

    This can be very tricky to implement but we have been able to make stable codebases with one or both of these
    approaches.
    """

    # This represents the name of the constructor this constructor is wrapping. Also, it sometimes represents the
    # type that this constructor will imitate so that isinstance(obj, ObjectConstructor) will operate as if you
    # instead called isinstance(obj, original_constructor). This allows us to replace a framework constructor with
    # our constructor.
    constructor_name = "String name of a constructor"

    # This represents the location in which our tensor constructor is stored within the library. If original_constructor
    # is 'torch.Tensor', then constructor_location is 'torch'.
    constructor_location: Optional[
        Type
    ] = None  # some python module on which the constructor lives

    # OPTIONAL: if constructor_name is actually a function which isn't init, meaning that the constructor produces
    # an object with a different name, deposit that object type here
    constructor_produces_type: Optional[Type] = None

    def install_inside_library(self) -> None:
        """Installs this custom constructor by replacing the library constructor with itself"""

        replacee = getattr(self.constructor_location, self.constructor_name)
        # If a custom constructor hasn't already been installed at this location, install it
        if not isinstance(replacee, ObjectConstructor):
            # cache original constructor at original_<constructor name>
            self.store_original_constructor()

            # save this constructor in its place
            setattr(self.constructor_location, self.constructor_name, self)

            # If the original constructor is a class (not just a standalone func like tf.constant or th.tensor)
            if isinstance(self.original_constructor, type):

                # copy static methods from previous constructor to new one
                copy_static_methods(
                    from_class=self.original_constructor, to_class=type(self)
                )

                # Replace all occurrences of the original constructor in the main module
                if self.constructor_location is not None:
                    main_module = sys.modules[
                        self.constructor_location.__name__.split(".")[0]
                    ]
                    replace_classes_in_module(
                        module=main_module, from_class=replacee, to_class=self
                    )
        else:
            raise AttributeError(
                f"You have already installed a custom constructor at location {self.constructor_location}."
                f"{self.constructor_name}. You cannot install a custom constructor for a custom "
                f"constructor. Eliminate the first, this one, or merge the two constructors by"
                f"concatenating their pre_init, init, and post_init methods."
            )

    def install_id_attribute(self, original_constructor: Type) -> Type:

        if (
            inspect.isclass(original_constructor)
            or self.constructor_produces_type is not None
        ):

            if self.constructor_produces_type is not None:
                type_to_subclass = self.constructor_produces_type
            else:
                type_to_subclass = original_constructor

            try:
                # if you are allowed to subclass this type

                class OriginalConstructorSubclass(type_to_subclass):  # type: ignore

                    __name__ = type_to_subclass.__name__

                    @property
                    def id(self) -> UID:
                        return self.__id

                    @id.setter
                    def id(self, new_id: UID) -> None:
                        self.__id = new_id

                original_constructor = OriginalConstructorSubclass

            # If this raises 'TypeError: type <> is not an acceptable base type'
            # then it's a special class of which Python cannot subtype
            # for more on this, see discussion:
            # https://stackoverflow.com/questions/10061752/which-classes-cannot-be-subclassed
            except TypeError:

                def id_get(self: Any) -> UID:
                    return self.__id

                def id_set(self: Any, new_id: UID) -> None:
                    self.__id = new_id

                original_constructor.id = property(fget=id_get, fset=id_set)

        return original_constructor

    def store_original_constructor(self) -> None:
        """Copies current object constructor to original_<constructor_name>

        Since all instances of ObjectConstructor are overloading an existing constructor within a library, we
        must first copy th original constructor (called the the "original" constructor) to a consistent location,
        as determined by the 'get_original_constructor_name' utility method.
        """

        # get the name of the place you want to move the original constructor to
        self.original_constructor_name = get_original_constructor_name(
            object_name=self.constructor_name
        )

        # save the original_constructor
        original_constructor = getattr(self.constructor_location, self.constructor_name)

        original_constructor = self.install_id_attribute(original_constructor)

        # copies the original constructor to a safe place for later use
        if not hasattr(self.constructor_location, self.original_constructor_name):
            setattr(
                self.constructor_location,
                self.original_constructor_name,
                original_constructor,
            )

    def __call__(self, *args: Tuple[Any, ...], **kwargs: Any) -> object:
        """Step-by-step method for constructing an object.

        Step 1: run pre_init() - augmenting args and kwargs as necessary.
        Step 2: run init() - initializes the object
        Step 3: run post_init() handling things like cleanup, custom attributes, and registration

        Args:
            my_type (Type): the type of object to initialize
            *args (list): a list of arguments to use to construct the object
            **kwargs (dict): a dictionary of arguments to use to construct the object

        Returns:
            obj (my_type): the newly initialized object

        """
        # TODO: ensure that constructor has been installed!!!

        new_args, new_kwargs = self.pre_init(*args, **kwargs)
        obj = self.init(*new_args, **new_kwargs)
        obj = self.post_init(obj=obj, args=new_args, kwargs=new_kwargs)

        return obj

    def pre_init(
        self, *args: Tuple[Any, ...], **kwargs: Any
    ) -> Tuple[Tuple[Any, ...], Any]:
        """Execute functionality before object is created

        Called before an object is initialized. Within this method you can
        perform behaviors which are appropriate preprations for initializing an object:
            - modify args and kwargs
            - initialize memory / re-use pre-initialized memory
            - interact with global parameters

        If you need to create metadata needed for init or post_init, store such information within
        kwargs.

        Args:
            *args (list): the arguments being used to initialize the object (including data)
            **kwargs (dict): the kwarguments beeing used to initialize the object
        Returns:
            *args (list): the (potentially modified) list of arguments for object construction
            **kwargs (dict): the (potentially modified) list of arguments for object construction
        """

        return args, kwargs

    def init(self, *args: Tuple[Any, ...], **kwargs: Any) -> object:
        """Initialize the object using original constructor

        This method selects a subset of the args and kwargs and uses them to
        initialize the underlying object using its original constructor. It returns
        the initialied object

        Args:
            *args (tuple): the arguments being used to initialize the object
            **kwargs (dict): the kwarguments eeing used to initialize the object
        Returns:
            out (SyftObject): returns the underlying syft object.
        """
        return self.original_constructor(*args, **kwargs)

    def post_init(self, obj: object, *args: Tuple[Any, ...], **kwargs: Any) -> object:
        """Execute functionality after object has been created.

        This method executes functionality which can only be
        run after an object has been initailized.
        It is particularly useful for logic which registers
        the created object into an appropriate
        registry. It is also useful for adding arbitrary
        attributes to the object after initialization.

        Args:
            *args (tuple): the arguments being used to initialize the object
            **kwargs (dict): the kwarguments eeing used to initialize the object
        Returns:
            out (SyftObject): returns the underlying syft object.
        """
        obj = self.assign_id(obj=obj)

        return obj

    def assign_id(self, obj: object) -> object:
        obj.id = UID()  # type: ignore
        return obj

    @property
    def original_constructor(self) -> Callable:
        """Return the original constructor for this method (i.e., the constructor the library had by
        default which this custom constructor overloaded.

        Note: I'm using try/except in this method instead of if/else because it's faster at runtime.
        """

        try:
            return getattr(self.constructor_location, self.original_constructor_name)
        except AttributeError:
            raise AttributeError(
                f"Syft's custom object constructor {type(self)} "
                f"cannot find the original constructor"
                f"to initialize '{self.constructor_name}' "
                f"objects, which should have been stored at "
                f"'{self.original_constructor_name}'. Either "
                f"you're doing active development and forgot "
                f"to cache the original constructor in the "
                f"right place before installing Syft's custom "
                f"constructor or something is very broken and "
                f"you should file a Github Issue. See "
                f"the documentation for ObjectConstructor for "
                f"more information on this functionality."
            )

    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:
        """Allow constructor to represent type it constructs

        Since we replace framework constructors (i.e., torch.Tensor) with instances of this
        constructor we also need this constructor to represent the type that it replaces,
        so that methods such as isinstance(my_tensor, th.Tensor) work correctly.

        Args:
            instance (object): an object of which we want to check the type against cls.

        """
        return isinstance(
            instance,
            getattr(
                cls.constructor_location,
                get_original_constructor_name(object_name=cls.constructor_name),
            ),
        )
