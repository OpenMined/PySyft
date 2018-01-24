import numpy as np

import syft.controller

class BaseTensor():
    def arithmetic_operation(self, x, name, inline=False):

        operation_cmd = name

        if (type(x) == type(self)):
            operation_cmd += "_elem"
            parameter = x.id
        else:
            operation_cmd += "_scalar"
            parameter = str(x)

        if (inline):
            operation_cmd += "_"

        response = self.controller.send_json(
            self.cmd(operation_cmd, [parameter]))  # sends the command
        if int(response) == self.id:
            return self
        else:
            return self.__class__(data=int(response), data_is_pointer=True)

    def __add__(self, x):
        """
        Performs element-wise addition arithmetic between two tensors
        Parameters
        ----------
        x : BaseTensor (Subclass)
            The Second tensor to perform addition with.
        Returns
        -------
        BaseTensor (Subclass)
            Output tensor
        """
        return self.arithmetic_operation(x, "add", False)
    def __iadd__(self, x):
        """
        Performs in place element-wise addition arithmetic between two tensors
        Parameters
        ----------
        x : BaseTensor (Subclass)
            The Second tensor to perform addition with.
        Returns
        -------
        BaseTensor (Subclass)
            Caller with values inplace
        """
        return self.arithmetic_operation(x, "add", True)

    def __truediv__(self, x):
        """
        Performs division arithmetic between two tensors
        Parameters
        ----------
        x : BaseTensor (Subclass)
            Second divident tensor
        Returns
        -------
        BaseTensor (Subclass)
            Output tensor
        """
        return self.arithmetic_operation(x, "div", False)

    def __itruediv__(self, x):
        """
        Performs division arithmetic between two tensors inplace.
        Parameters
        ----------
        x : BaseTensor (Subclass)
            Second divident tensor
        Returns
        -------
        BaseTensor (Subclass)
            Caller with values inplace
        """
        return self.arithmetic_operation(x, "div", True)

    def __pow__(self, x):
        """
        Takes the power of each element in input ('self') with 'x' and
        returns a tensor with the result.
        Parameters
        ----------
        x : BaseTensor (Subclass)
            Exponent tensor
        Returns
        -------
        BaseTensor (Subclass)
            Output tensor
        """
        return self.arithmetic_operation(x, "pow", False)

    def __ipow__(self, x):
        """
        Takes the power of each element in input ('self') with 'x' and
        returns a tensor with the result inplace.
        Parameters
        ----------
        x : BaseTensor (Subclass)
            Exponent tensor
        Returns
        -------
        BaseTensor (Subclass)
            Caller with values inplace
        """
        return self.arithmetic_operation(x, "pow", True)

    def pow(self, x):
        """
        Takes the power of each element in input ('self') with 'x' and
        returns a tensor with the result.
        Parameters
        ----------
        x : BaseTensor (Subclass)
            Exponent tensor
        Returns
        -------
        BaseTensor (Subclass)
            Output tensor
        """
        return self.arithmetic_operation(x, "pow", False)

    def pow_(self, x):
        """
        Takes the power of each element in input ('self') with 'x', inplace.
        Parameters
        ----------
        x : BaseTensor (Subclass)
            Exponent tensor
        Returns
        -------
        BaseTensor (Subclass)
            Caller with values inplace
        """
        return self.arithmetic_operation(x, "pow", True)

    def __mod__(self, x):
        """
        Performs Modulus arithmetic operation between two tensors.
        Parameters
        ----------
        x : BaseTensor (Subclass)
            Dividend tensor
        Returns
        -------
        BaseTensor (Subclass)
            Output tensor
        """
        return self.arithmetic_operation(x, "remainder", False)

    def __imod__(self, x):
        """
        Performs Modulus arithmetic operation between two tensors inplace.
        Parameters
        ----------
        x : BaseTensor (Subclass)
            Dividend tensor
        Returns
        -------
        BaseTensor (Subclass)
            Caller with values inplace
        """
        return self.arithmetic_operation(x, "remainder", True)

    def __mul__(self, x):
        """
        Performs Multiplication arithmetic operation between two tensors.
        Parameters
        ----------
        x : BaseTensor (Subclass)
            Second tensor to be multiplied with.
        Returns
        -------
        BaseTensor (Subclass)
            Output tensor
        """
        return self.arithmetic_operation(x, "mul", False)

    def __imul__(self, x):
        """
        Performs Multiplication arithmetic operation between two tensors inplace.
        Parameters
        ----------
        x : BaseTensor (Subclass)
            Second tensor to be multiplied with.
        Returns
        -------
        BaseTensor (Subclass)
            Caller with values inplace
        """
        return self.arithmetic_operation(x, "mul", True)

    def __sub__(self, x):
        """
        Performs element-wise substraction arithmetic between two tensors
        Parameters
        ----------
        x : BaseTensor (Subclass)
            The Second tensor to perform addition with.
        Returns
        -------
        BaseTensor (Subclass)
            Output tensor
        """
        return self.arithmetic_operation(x, "sub", False)

    def __isub__(self, x):
        """
        Performs element-wise substraction arithmetic between two tensors
        Parameters
        ----------
        x : BaseTensor (Subclass)
            The Second tensor to perform addition with.
        Returns
        -------
        BaseTensor (Subclass)
            Caller with values inplace
        """
        return self.arithmetic_operation(x, "sub", True)

    def remainder(self, divisor):
        """
        Computes the element-wise remainder of division.
        inplace.
        Parameters
        ----------
        Returns
        -------
        BaseTensor (Subclass)
            Output tensor
        """
        return self.arithmetic_operation(divisor, "remainder")

    def remainder_(self, divisor):
        """
        Computes the element-wise remainder of division, inplace.
        Parameters
        ----------
        Returns
        -------
        BaseTensor (Subclass)
            Caller with values inplace
        """
        return self.arithmetic_operation(divisor, "remainder", True)
