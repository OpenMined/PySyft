import pytest
import torch
import torch.nn as nn

from syft.frameworks.torch.tensors.interpreters.private import PrivateTensor
from syft.exceptions import GetNotPermittedError
from syft.exceptions import SendNotPermittedError


def test_wrap():
    """
    Test the .on() wrap functionality for LoggingTensor
    """

    x_tensor = torch.Tensor([1, 2, 3])
    x = PrivateTensor().on(x_tensor)
    assert isinstance(x, torch.Tensor)
    assert isinstance(x.child, PrivateTensor)
    assert isinstance(x.child.child, torch.Tensor)


def test_native_private_tensor_method():
    """
    Test native's private_tensor method.
    """
    x_tensor = torch.Tensor([1, 2, 3])
    private_x = x_tensor.private_tensor(allowed_users=["testing"])
    assert isinstance(private_x, torch.Tensor)
    assert isinstance(private_x.child, PrivateTensor)
    assert isinstance(private_x.child.child, torch.Tensor)


def test_allow_method():
    """
    Test native's private_tensor method.
    """

    x_tensor = torch.Tensor([1, 2, 3])

    # User credentials mockup
    class UserAuthMockup(object):
        def __init__(self, login, password):
            self.login = login
            self.password = password

        def __eq__(self, other):
            if isinstance(other, UserAuthMockup):
                return self.login == other.login and self.__password == other.password

    allowed_user = UserAuthMockup("user", "password")
    second_allowed_user = UserAuthMockup("second_user", "password")
    unallowed_user = UserAuthMockup("example", "password")

    private_x = x_tensor.private_tensor(allowed_users=[allowed_user, second_allowed_user])
    assert private_x.allow(allowed_user)
    assert private_x.allow(second_allowed_user)
    assert not private_x.allow(unallowed_user)


def test_send_method(workers):
    bob = workers["bob"]
    x_tensor = torch.tensor([4, 5, 6, 7, 8])

    private_x = x_tensor.private_tensor(allowed_users=["User"])

    # Try to call send() without credentials
    with pytest.raises(SendNotPermittedError):
        private_x.send(bob)

    # Try to call send() with wrong credentials
    with pytest.raises(SendNotPermittedError):
        private_x.send(user="unallowed_user")

    # Try to call send() with allowed credentails
    private_x_pointer = private_x.send(bob, user="User")


def test_get_method(workers):
    bob = workers["bob"]
    x_tensor = torch.Tensor([1, 2, 3])

    private_x = x_tensor.private_tensor(allowed_users=["User"])

    private_x_pointer = private_x.send(bob, user="User")

    # Try to call get() without credentials
    with pytest.raises(GetNotPermittedError):
        private_x_pointer.get()

    # Try to call get() with wrong credentials
    with pytest.raises(GetNotPermittedError):
        private_x_pointer.get(user="UnregisteredUser")

    # Try to call get() with allowed credentials
    result = private_x_pointer.get(user="User")


def test_private_tensor_registration(hook):
    with hook.local_worker.registration_enabled():
        x = torch.tensor([1.0])
        private_x = x.private_tensor(allowed_users=["User"])

        assert hook.local_worker.get_obj(x.id) == x


def test_allowed_to_get():
    x = torch.tensor([1, 2, 3, 4, 5, 6])
    assert x.allow("User")  # Public tensors always return true.

    private_x = x.private_tensor(allowed_users=["User"])

    assert private_x.allow("User")  # It Returns true to previously registered user.
    assert not private_x.allow("AnotherUser")  # It Returns False to non previously registered user.


def test_add_method():
    t = torch.tensor([0.1, 0.2, 0.3])
    x = t.private_tensor(allowed_users=["User"])

    y = x + x

    # Test if it preserves the wraper stack
    assert isinstance(x, torch.Tensor)
    assert isinstance(x.child, PrivateTensor)
    assert isinstance(x.child.child, torch.Tensor)

    assert x.allow("User")  # Test if it preserves the parent user credentials.


@pytest.mark.parametrize("method", ["t", "matmul"])
@pytest.mark.parametrize("parameter", [False, True])
def test_methods_for_linear_module(method, parameter):
    """
    Test all the methods used in the F.linear functions
    """
    if parameter:
        tensor = nn.Parameter(torch.tensor([[1.0, 2], [3, 4]]))
    else:
        tensor = torch.tensor([[1.0, 2], [3, 4]])
    fp_tensor = tensor.fix_precision()

    # ADD Private Tensor at wrapper stack
    private_fp_tensor = fp_tensor.private_tensor(allowed_users=["User"])  # ADD Private Layer

    if method != "t":
        fp_result = getattr(private_fp_tensor, method)(private_fp_tensor)
        result = getattr(tensor, method)(tensor)
    else:
        fp_result = getattr(private_fp_tensor, method)()
        result = getattr(tensor, method)()

    assert (result == fp_result.float_precision()).all()


def test_torch_add():
    x = torch.tensor([0.1, 0.2, 0.3]).fix_prec()

    # ADD Private Tensor at wrapper stack
    x = x.private_tensor(allowed_users=["User"])

    y = torch.add(x, x)

    assert (y.child.child.child == torch.LongTensor([200, 400, 600])).all()

    y_fp = y.float_prec()

    assert (y_fp == torch.tensor([0.2, 0.4, 0.6])).all()

    # Test if it preserves the parent user credentials.
    assert y.allow("User")
    assert not y.allow("NonRegisteredUser")

    # With negative numbers
    x = torch.tensor([-0.1, -0.2, 0.3]).fix_prec()
    y = torch.tensor([0.4, -0.5, -0.6]).fix_prec()

    # ADD Private Tensor at wrapper stack
    x = x.private_tensor(allowed_users=["UserCredential"])
    y = y.private_tensor(allowed_users=["UserCredential"])

    z = torch.add(x, y)
    z_fp = z.float_prec()

    assert (z_fp == torch.tensor([0.3, -0.7, -0.3])).all()

    # Test if it preserves the parent user credentials.
    assert z.allow("UserCredential")
    assert not z.allow("NonRegisteredUser")


def test_torch_sub():
    x = torch.tensor([0.5, 0.8, 1.3]).fix_prec()
    y = torch.tensor([0.1, 0.2, 0.3]).fix_prec()

    # ADD Private Tensor at wrapper stack
    x = x.private_tensor(allowed_users=["User"])
    y = y.private_tensor(allowed_users=["User"])

    z = torch.sub(x, y)

    # Test if it preserves the parent user credentials.
    assert z.allow("User")
    assert not z.allow("NonRegisteredUser")

    assert (z.child.child.child == torch.LongTensor([400, 600, 1000])).all()
    z_fp = z.float_prec()

    assert (z_fp == torch.tensor([0.4, 0.6, 1.0])).all()


def test_torch_mul():
    # mul with non standard fix precision
    x = torch.tensor([2.113]).fix_prec(precision_fractional=2)

    # ADD Private Tensor at wrapper stack
    x = x.private_tensor(allowed_users=["User"])

    y = torch.mul(x, x)

    assert y.child.child.child == torch.LongTensor([445])
    assert y.child.child.precision_fractional == 2

    # Test if it preserves the parent user credentials.
    assert y.allow("User")
    assert not y.allow("NonRegisteredUser")

    y = y.float_prec()

    assert y == torch.tensor([4.45])

    # Mul with negative numbers
    x = torch.tensor([2.113]).fix_prec()
    y = torch.tensor([-0.113]).fix_prec()

    # ADD Private Tensor at wrapper stack
    x = x.private_tensor(allowed_users=["User"])
    y = y.private_tensor(allowed_users=["User"])

    z = torch.mul(x, y)

    assert z.child.child.precision_fractional == 3

    # Test if it preserves the parent user credentials.
    assert z.allow("User")
    assert not z.allow("NonRegisteredUser")

    z = z.float_prec()
    assert z == torch.tensor([-0.2380])

    x = torch.tensor([11.0]).fix_prec(field=2 ** 16, precision_fractional=2)

    # ADD Private Tensor at wrapper stack
    x = x.private_tensor(allowed_users=["User"])

    y = torch.mul(x, x)

    y = y.float_prec()

    assert y == torch.tensor([121.0])

    # mixing + and *
    x = torch.tensor([2.113]).fix_prec()
    y = torch.tensor([-0.113]).fix_prec()

    # ADD Private Tensor at wrapper stack
    x = x.private_tensor(allowed_users=["User"])
    y = y.private_tensor(allowed_users=["User"])

    z = torch.mul(x, y + y)
    z = z.float_prec()

    assert z == torch.tensor([-0.4770])


def test_operate_with_integer_constants():
    x = torch.tensor([1.0])
    x_fp = x.fix_precision()

    # PrivateTensor at wrapper stack.
    x_fp = x_fp.private_tensor(allowed_users=["User"])

    # ADD
    r_fp = x_fp + 10

    # Test if it preserves the parent user credentials.
    assert r_fp.allow("User")
    assert not r_fp.allow("NonRegisteredUser")

    r = r_fp.float_precision()
    assert r == x + 10

    # SUB
    r_fp = x_fp - 7

    # Test if it preserves the parent user credentials.
    assert r_fp.allow("User")
    assert not r_fp.allow("NonRegisteredUser")

    r = r_fp.float_precision()
    assert r == x - 7

    # MUL
    r_fp = x_fp * 2

    # Test if it preserves the parent user credentials.
    assert r_fp.allow("User")
    assert not r_fp.allow("NonRegisteredUser")

    assert r_fp.float_precision() == x * 2

    # DIV
    r_fp = x_fp / 5

    # Test if it preserves the parent user credentials.
    assert r_fp.allow("User")
    assert not r_fp.allow("NonRegisteredUser")

    assert r_fp.float_precision() == x / 5
