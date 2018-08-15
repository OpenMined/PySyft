import torch


class TorchGuard():

    def __init__(self):

        # Safety checks for serializing and deserializing torch objects
        # Desperately needs stress testing before going out in the wild
        self.map_tensor_type = {
            'torch.FloatTensor': torch.FloatTensor,
            'torch.DoubleTensor': torch.DoubleTensor,
            'torch.HalfTensor': torch.HalfTensor,
            'torch.ByteTensor': torch.ByteTensor,
            'torch.CharTensor': torch.CharTensor,
            'torch.ShortTensor': torch.ShortTensor,
            'torch.IntTensor': torch.IntTensor,
            'torch.LongTensor': torch.LongTensor,
        }
        self.map_var_type = {
            'torch.autograd.variable.Variable': torch.autograd.variable.Variable,
            'torch.nn.parameter.Parameter': torch.nn.parameter.Parameter,
        }
        self.map_torch_type = dict(self.map_tensor_type, **self.map_var_type)

    def types_guard(self, torch_type_str):
        """types_guard(torch_type_str) -> torch.Tensor or torch.autograd.Variable

        This method converts strings into a type reference. This prevents
        deserialized JSON from being able to instantiate objects of arbitrary
        type which would be a security concern.

        :Parameters:

        * **torch_type_str (string)** A string representing the type of object that is
          to be returned.

        * **out (a torch type)** The type the string refersto (if it's present in the
          acceptible list self.map_torch_type)

        :Example:

        >>> from syft.core.hooks import TorchHook
        >>> hook = TorchHook()
        Hooking into Torch...
        Overloading Complete.
        >>> torch_type = hook.types_guard('torch.FloatTensor')
        >>> x = torch_type([1,2,3,4,5])
        >>> x
         1
         2
         3
         4
         5
        [torch.FloatTensor of size 5]
        """
        try:
            return self.map_torch_type[torch_type_str]
        except KeyError:
            raise TypeError(
                "Tried to receive a non-Torch object of type {}.".format(
                    torch_type_str,
                ),
            )

    def tensor_contents_guard(self, contents):
        """tensor_contents_guard(contents) -> contents
        TODO: check to make sure the incoming list isn't dangerous to use for
               constructing a tensor (likely non-trivial). Accepts the list of JSON objects
               and returns the list of JSON ojects. Should throw and exception if there's a
               security concern.
        """
        return contents
