import syft
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.overload_torch import overloaded


class LoggingTensor(AbstractTensor):
    def __init__(self, owner=None, id=None, tags=None, description=None):
        """Initializes a LoggingTensor, whose behaviour is to log all operations
        applied on it.

        Args:
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the LoggingTensor.
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description)

    # Method overloading

    @overloaded.method
    def add(self, _self, *args, **kwargs):
        """
        Here is an example of how to use the @overloaded.method decorator. To see
        what this decorator do, just look at the next method manual_add: it does
        exactly the same but without the decorator.

        Note the subtlety between self and _self: you should use _self and NOT self.
        """
        print("Log method add")
        response = getattr(_self, "add")(*args, **kwargs)

        return response

    def manual_add(self, *args, **kwargs):
        """
        Here is the version of the add method without the decorator: as you can see
        it is much more complicated. However you might need sometimes to specify
        some particular behaviour: so here what to start from :)
        """
        # Replace all syft tensor with their child attribute
        new_self, new_args, new_kwargs = syft.frameworks.torch.hook_args.hook_method_args(
            "add", self, args, kwargs
        )

        print("Log method manual_add")
        # Send it to the appropriate class and get the response
        response = getattr(new_self, "add")(*new_args, **new_kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = syft.frameworks.torch.hook_args.hook_response(
            "add", response, wrap_type=type(self)
        )
        return response

    # Module & Function overloading

    # We overload two torch functions:
    # - torch.add
    # - torch.nn.functional.relu

    @staticmethod
    @overloaded.module
    def torch(module):
        """
        We use the @overloaded.module to specify we're writing here
        a function which should overload the function with the same
        name in the <torch> module
        :param module: object which stores the overloading functions

        Note that we used the @staticmethod decorator as we're in a
        class
        """

        def add(x, y):
            """
            You can write the function to overload in the most natural
            way, so this will be called whenever you call torch.add on
            Logging Tensors, and the x and y you get are also Logging
            Tensors, so compared to the @overloaded.method, you see
            that the @overloaded.module does not hook the arguments.
            """
            print("Log function torch.add")
            return x + y

        # Just register it using the module variable
        module.add = add

        @overloaded.function
        def mul(x, y):
            """
            You can also add the @overloaded.function decorator to also
            hook arguments, ie all the LoggingTensor are replaced with
            their child attribute
            """
            print("Log function torch.mul")
            return x * y

        # Just register it using the module variable
        module.mul = mul

        # You can also overload functions in submodules!
        @overloaded.module
        def nn(module):
            """
            The syntax is the same, so @overloaded.module handles recursion
            Note that we don't need to add the @staticmethod decorator
            """

            @overloaded.module
            def functional(module):
                def relu(x):
                    print("Log function torch.nn.functional.relu")
                    return x * (x.child > 0)

                module.relu = relu

            module.functional = functional

        # Modules should be registered just like functions
        module.nn = nn

    @classmethod
    def on_function_call(cls, command):
        """
        Override this to perform a specific action for each call of a torch
        function with arguments containing syft tensors of the class doing
        the overloading
        """
        cmd, _, args, kwargs = command
        print("Default log", cmd)
