from syft.generic.frameworks.hook import hook_args


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

        def _hook_method_args(self, *args, **kwargs):
            # Replace all syft tensor with their child attribute
            new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
                attr.__name__, self, args, kwargs
            )

            # Send it to the appropriate class and get the response
            response = attr(self, new_self, *new_args, **new_kwargs)

            # Put back SyftTensor on the tensors found in the response
            response = hook_args.hook_response(
                attr.__name__, response, wrap_type=type(self), wrap_args=self.get_class_attributes()
            )

            return response

        return _hook_method_args

    @staticmethod
    def overload_function(attr):
        """
        hook args and response for functions that hold the @overloaded.function decorator
        """

        def _hook_function_args(*args, **kwargs):

            # TODO have a better way to infer the type of tensor -> this is implies
            # that the first argument is a tensor (even if this is the case > 99%)
            tensor = args[0] if not isinstance(args[0], (tuple, list)) else args[0][0]
            cls = type(tensor)

            # Replace all syft tensor with their child attribute
            new_args, new_kwargs, new_type = hook_args.unwrap_args_from_function(
                attr.__name__, args, kwargs
            )

            # Send it to the appropriate class and get the response
            response = attr(*new_args, **new_kwargs)

            # Put back SyftTensor on the tensors found in the response
            response = hook_args.hook_response(
                attr.__name__, response, wrap_type=cls, wrap_args=tensor.get_class_attributes()
            )

            return response

        return _hook_function_args

    @staticmethod
    def overload_module(attr):

        module = Module()
        attr(module)

        return module


overloaded = Overloaded()
