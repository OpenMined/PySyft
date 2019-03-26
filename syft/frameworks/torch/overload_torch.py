import syft


class Module(object):
    pass


class Overloaded:
    def __init__(self):
        self.method = Overloaded.overload_method
        self.function = Overloaded.overload_function
        self.module = Overloaded.overload_module

    @staticmethod
    def overload_method(attr):
        """
        hook args and response for methods that hold the @overloaded.method decorator
        """

        def hook_args(self, *args, **kwargs):
            # Replace all syft tensor with their child attribute
            new_self, new_args, new_kwargs = syft.frameworks.torch.hook_args.hook_method_args(
                attr.__name__, self, args, kwargs
            )

            # Send it to the appropriate class and get the response
            response = attr(self, new_self, *new_args, **new_kwargs)

            # Put back SyftTensor on the tensors found in the response
            response = syft.frameworks.torch.hook_args.hook_response(
                attr.__name__, response, wrap_type=type(self), wrap_args=self.get_class_attributes()
            )

            return response

        return hook_args

    @staticmethod
    def overload_function(attr):
        """
        hook args and response for functions that hold the @overloaded.function decorator
        """

        def hook_args(*args, **kwargs):

            # TODO have a better way
            cls = type(args[0])

            # Replace all syft tensor with their child attribute
            new_args, new_kwargs, new_type = syft.frameworks.torch.hook_args.hook_function_args(
                attr.__name__, args, kwargs
            )

            # Send it to the appropriate class and get the response
            response = attr(*new_args, **new_kwargs)

            # Put back SyftTensor on the tensors found in the response
            response = syft.frameworks.torch.hook_args.hook_response(
                attr.__name__, response, wrap_type=cls
            )

            return response

        return hook_args

    @staticmethod
    def overload_module(attr):

        module = Module()
        attr(module)

        return module


overloaded = Overloaded()
