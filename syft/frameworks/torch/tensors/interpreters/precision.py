import syft
import torch
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.tensors.interpreters.utils import hook


class FixedPrecisionTensor(AbstractTensor):
    def __init__(
        self,
        parent: AbstractTensor = None,
        owner=None,
        id=None,
        field: int = (2 ** 31) - 1,
        base: int = 10,
        precision_fractional: int = 3,
        precision_integral: int = 1,
        kappa: int = 1,
        tags: set = None,
        description: str = None,
    ):
        """Initializes a Fixed Precision tensor, which encodes all decimal point
        values using an underlying integer value.

        The FixedPrecision enables to manipulate floats over an interface which
        supports only integers, Such as _SPDZTensor.

        This is done by specifying a precision p and given a float x,
        multiply it with 10**p before rounding to an integer (hence you keep
        p decimals)

        Args:
            parent: An optional AbstractTensor wrapper around the FixedPrecisionTensor
                which makes it so that you can pass this FixedPrecisionTensor to all
                the other methods/functions that PyTorch likes to use, although
                it can also be other tensors which extend AbstractTensor, such
                as custom tensors for Secure Multi-Party Computation or
                Federated Learning.
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the FixedPrecisionTensor.
        """
        super().__init__(tags, description)

        self.parent = parent
        self.owner = owner
        self.id = id
        self.child = None

        self.field = field
        self.base = base
        self.precision_fractional = precision_fractional
        self.precision_integral = precision_integral
        self.kappa = kappa
        self.torch_max_value = torch.tensor([round(self.field / 2)])

    def fix_precision(self):
        """This method encodes the .child object using fixed precision"""

        rational = self.child

        owner = rational.owner
        upscaled = (rational * self.base ** self.precision_fractional).long()
        field_element = upscaled % self.field

        # Handle neg values
        gate = field_element.gt(self.torch_max_value).long()
        neg_nums = (field_element - self.field) * gate
        pos_nums = field_element * (1 - gate)
        field_element = neg_nums + pos_nums

        self.child = field_element
        return self

    def float_precision(self):
        """this method returns a new tensor which has the same values as this
        one, encoded with floating point precision"""

        value = self.child.long() % self.field

        if len(value.size()) == 0:
            # raise TypeError("Can't decode empty tensor")
            return None

        gate = value.native_gt(self.torch_max_value).long()
        neg_nums = (value - self.field) * gate
        pos_nums = value * (1 - gate)
        result = (neg_nums + pos_nums).float() / (self.base ** self.precision_fractional)

        return result

    @hook
    def add(self, _self, *args, **kwargs):
        """Add two fixed precision tensors together.

        TODO: fix!
        """
        response = getattr(_self, "add")(*args, **kwargs)

        return response

    @classmethod
    def handle_func_command(cls, command):
        """
        Receive an instruction for a function to be applied on a FixedPrecision Tensor,
        Perform some specific action (like logging) which depends of the
        instruction content, replace in the args all the FPTensors with
        their child attribute, forward the command instruction to the
        handle_function_command of the type of the child attributes, get the
        response and replace a FixedPrecision on top of all tensors found in
        the response.
        :param command: instruction of a function command: (command name,
        <no self>, arguments[, kwargs])
        :return: the response of the function command
        """
        # TODO: add kwargs in command
        cmd, _, args = command

        # Do what you have to
        print("Fixed Precision function", cmd)

        # TODO: I can't manage the import issue, can you?
        # Replace all FixedPrecisionTensor with their child attribute
        new_args, new_type = syft.frameworks.torch.hook_args.hook_function_args(cmd, args)

        # build the new command
        new_command = (cmd, None, new_args)

        # Send it to the appropriate class and get the response
        response = new_type.handle_func_command(new_command)

        # Put back FixedPrecisionTensor on the tensors found in the response
        response = syft.frameworks.torch.hook_args.hook_response(cmd, response, wrap_type=cls)

        return response

    def get(self):
        """Just a pass through. This is most commonly used when calling .get() on a
        FixedPrecisionTensor which has also been shared."""
        return FixedPrecisionTensor().on(self.child.get())

    def share(self, *owners):
        self.child = self.child.share(*owners)
        return self
