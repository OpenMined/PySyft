import syft


def hook(attr):
    """
    hook args and response for methods that hold the @hook decoarator
    """

    def hook_args(self, *args, **kwargs):
        # Replace all syft tensor with their child attribute
        new_self, new_args = syft.frameworks.torch.hook_args.hook_method_args(
            attr.__name__, self, args
        )

        # Send it to the appropriate class and get the response
        response = attr(self, new_self, *new_args, **kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = syft.frameworks.torch.hook_args.hook_response(
            attr.__name__, response, wrap_type=type(self), wrap_args=self.get_class_attributes()
        )

        return response

    return hook_args
