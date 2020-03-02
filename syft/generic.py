class ObjectConstructor(object):
    """Syft allows for the extension and remote execution of a range of python libraries. As such,
    a common need is the ability to modify library-specific constructors of objects. As many constructors
    are difficult to overload (or perhaps have differing ways to overload them), we instead offer a fake,
    static Constructor method which allows for:

        - arbitrary custom args and kwargs,
        - arbitrary arg and kwarg manipulation
        - arbitrary functionality before the underlying (native) object constructor is called
        - arbitrary functionality after the underlying (native) object constructor is called.

    Thus, if any object has it's functionality extended or overridden by PySyft, it should be created using
    an extension of this class.
    """

    def __init__(self, obj_type):
        self.type = obj_type

    def __call__(self, *args, **kwargs):
        """Step-by-step method for constructing an object.

        Step 1: run pre_init() - augmenting args and kwargs as necessary.
        Step 2: run underlying_framework_init() - initializes the object
        Step 3: run post_init() handling things like cleanup, custom attributes, and registration

        Args:
            my_type (Type): the type of object to initialize
            *args (list): a list of arguments to use to construct the object
            **kwargs (dict): a dictionary of arguments to use to construct the object

        Returns:
            obj (my_type): the newly initialized object

        """

        new_args, new_kwargs = self.pre_init(*args, **kwargs)
        obj = self.underlying_framework_init(*new_args, **new_kwargs)
        obj = self.post_init(obj, *new_args, **new_kwargs)

        return obj

    def pre_init(self, *args, **kwargs):
        """Execute functionality before object is created

        Called before an object is initialized. Within this method you can
        perform behaviors which are appropriate preprations for initializing an object:
            - modify args and kwargs
            - initialize memory / re-use pre-initialized memory
            - interact with global parameters

        If you need to create metadata needed for init or post_init, store such information within kwargs.

        Args:
            *args (list): the arguments being used to initialize the object (including data)
            **kwargs (dict): the kwarguments beeing used to initialize the object
        Returns:
            *args (list): the (potentially modified) list of arguments for object construction
            **kwargs (dict): the (potentially modified) list of arguments for object construction
        """

        return args, kwargs

    def underlying_framework_init(self, *args, **kwargs):
        """Initialize the object using native constructor

        This method selects a subset of the args and kwargs and uses them to
        initialize the underlying object using its native constructor. It returns
        the initialied object

        Args:
            *args (list): the arguments being used to initialize the object
            **kwargs (dict): the kwarguments eeing used to initialize the object
        Returns:
            out (SyftObject): returns the underlying syft object.
        """
        return self.type(*args, **kwargs)

    def post_init(self, obj, *args, **kwargs):
        """Execute functionality after object has been created.

        This method executes functionality which can only be run after an object has been initailized. It is
        particularly useful for logic which registers the created object into an appropriate registry. It is
        also useful for adding arbitrary attributes to the object after initialization.

        Args:
            *args (list): the arguments being used to initialize the object
            **kwargs (dict): the kwarguments eeing used to initialize the object
        Returns:
            out (SyftObject): returns the underlying syft object.
        """

        return obj
