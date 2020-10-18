import copy
import types
from functools import wraps

from syft.generic.frameworks.hook import hook_args
from syft.execution.plan import Plan

import crypten
import torch as th


class CryptenHook(object):
    def hook_crypten(self, to_auto_overload_crypten):
        for func_name in to_auto_overload_crypten:
            self._perform_function_overloading(func_name)

    def _perform_function_overloading(self, func_name):

        # Where the overloading happens
        # 1. Get native function
        native_func = getattr(crypten, func_name)
        # 2. Check it is a proper function
        if type(native_func) in [types.FunctionType, types.BuiltinFunctionType]:
            # 3. Build the hooked function
            new_func = self._get_hooked_func(func_name, native_func)
            # 4. Move the native function
            setattr(crypten, f"native_{func_name}", native_func)
            # 5. Put instead the hooked one
            setattr(crypten, func_name, new_func)

    def _get_hooked_func(self, func_api_name, func):
        @wraps(func)
        def overloaded_func(*args, **kwargs):
            """
            Operate the hooking
            """
            tensor_type = (
                type(args[0]) if not isinstance(args[0], (tuple, list)) else type(args[0][0])
            )

            if tensor_type == crypten.mpc.MPCTensor:
                call_func = getattr(crypten, f"native_{func_api_name}")
                new_args = args
                new_kwargs = kwargs
            else:
                # Remove the top level class (should be the PlaceHolder) and call again
                new_args, new_kwargs, _ = hook_args.unwrap_args_from_function(func, args, kwargs)
                call_func = func

            response = call_func(*new_args, **new_kwargs)

            return response

        return overloaded_func


class CrypTenPlanBuild(object):
    @staticmethod
    def f_return_none(*args, **kwargs):
        return None

    @staticmethod
    def f_return_cryptensor(*args, **kwargs):
        return crypten.cryptensor(th.zeros([]))

    @staticmethod
    def f_return_module(*args, **kwargs):
        return crypten.nn.Module()


# Methods that need to be added to the PlaceHolder when building the plan
# and hence need to be traced
crypten_to_auto_overload = {
    crypten.mpc.MPCTensor: ["get_plain_text"],
    crypten.nn.Module: [
        "encrypt",
        "decrypt",
        "__call__",
        "train",
        "zero_grad",
        "update_parameters",
        "eval",
    ],
}

"""
Methods which we overwrite when building the plan
This list might become bigger as we should support more operations

Why is needed?
When the local party builds the plan it needs to trace the operations that
are done on a CrypTensor and CrypTenModule. It does not know about
what data/model are involved in the computation.
Because of this, when building the plan, the local party can work with
only some "shells" of specific operations

Workflow for local party -- can be seen in syft/frameworks/crypten/context.py:
1. replace real functions/methods with shell like functions/methods (only for some)
2. Call "crypten_init" (to be able to perform some of the CrypTen computation on the
  shell CryptenTensors/CryptenModules)
   Eg: cryptensor + cryptensor
  If this is not done CrypTen will throw an exception because there is tried
to run crypten specific computation in a not crypten environment
3. build the plan (register the set of actions that should be performed)
4. Call "crypten_uninit"
5. Undo the operations done at step 1

Q: Why get_plain_text appears only in the crypten_to_auto_overload, but not
in crypten_plan_hook?
A: 1. The method is added to the PlaceHolder class such that when the plan is
built it would be traced.
   2. The local party will build the plan by being in a CrypTen context (there
is called "crypten.init()" before building the plan) and calling "get_plain_text"
on the "shell" CrypTensor (in our case is a CrypTensor that has only values of 0)
will not require to overwrite another function
"""


def define_crypten_plan_hook():
    global crypten_plan_hook

    crypten_plan_hook = {
        crypten: {
            "load": CrypTenPlanBuild.f_return_cryptensor,
            "cat": CrypTenPlanBuild.f_return_cryptensor,
            "load_model": CrypTenPlanBuild.f_return_module,
        },
        crypten.nn: {
            "MSELoss": CrypTenPlanBuild.f_return_module,
            "BCELoss": CrypTenPlanBuild.f_return_module,
        },
        crypten.nn.Module: {
            "encrypt": CrypTenPlanBuild.f_return_none,
            "decrypt": CrypTenPlanBuild.f_return_none,
            "__call__": CrypTenPlanBuild.f_return_cryptensor,
            "train": CrypTenPlanBuild.f_return_none,
            "zero_grad": CrypTenPlanBuild.f_return_none,
            "update_parameters": CrypTenPlanBuild.f_return_none,
            "eval": CrypTenPlanBuild.f_return_none,
        },
        crypten.mpc.MPCTensor: {
            "__getitem__": CrypTenPlanBuild.f_return_cryptensor,
            "unsqueeze": CrypTenPlanBuild.f_return_cryptensor,
            "flatten": CrypTenPlanBuild.f_return_cryptensor,
        },
    }


def hook_plan_building():
    """
    When builing the plan we should not call directly specific
    methods from CrypTen and as such we return here some "dummy" responses
    only to build the plan.
    """

    if "crypten_plan_hook" not in globals():
        define_crypten_plan_hook()

    for module, replace_dict in crypten_plan_hook.items():
        for method_name, f_replace in replace_dict.items():
            method = getattr(module, method_name)
            setattr(module, f"planhook_{method_name}", method)
            setattr(module, method_name, f_replace)


def unhook_plan_building():
    """
    After building the plan we unhook the methods such that
    we call the "real" methods in the actual workers
    """

    for module, replace_dict in crypten_plan_hook.items():
        for method_name in replace_dict:
            method = getattr(module, f"planhook_{method_name}")
            setattr(module, method_name, method)
            delattr(module, f"planhook_{method_name}")


def hook_crypten():
    """Hook the load function from crypten"""
    from syft.frameworks.crypten import load, load_model

    setattr(crypten, "load", load)
    setattr(crypten, "load_model", load_model)

    crypten_funcs_overload = ["cat"]
    CryptenHook().hook_crypten(crypten_funcs_overload)


def hook_crypten_module():
    """Overloading crypten.nn.Module with PySyft functionality, the primary module
    responsible for core ML functionality such as Neural network layers and
    loss functions.
    It is important to note that all the operations are actually in-place.
    """
    import crypten

    def _check_encrypted(model):
        """Raise an exception if the model is encrypted"""
        if model.encrypted:
            raise RuntimeError("Crypten model must be unencrypted to run PySyft operations")

    crypten.nn.Module._check_encrypted = _check_encrypted

    def module_is_missing_grad(model):
        """Checks if all the parameters in the model have been assigned a gradient"""
        for p in model.parameters():
            if p.grad is None:
                return True
        return False

    def create_grad_objects(model):
        """Assigns gradient to model parameters if not assigned"""
        for p in model.parameters():
            if p.requires_grad:  # check if the object requires a grad object
                o = p.sum()
                o.backward()
                if p.grad is not None:
                    p.grad -= p.grad

    def module_send_(nn_self, *dest, force_send=False, **kwargs):
        """Overloads crypten.nn instances so that they could be sent to other workers"""
        nn_self._check_encrypted()

        if module_is_missing_grad(nn_self):
            create_grad_objects(nn_self)

        for p in nn_self.parameters():
            p.send_(*dest, **kwargs)

        if isinstance(nn_self.forward, Plan):
            nn_self.forward.send(*dest, force=force_send)

        return nn_self

    crypten.nn.Module.send = module_send_
    crypten.nn.Module.send_ = module_send_

    def module_move_(nn_self, destination):
        """Overloads crypten.nn instances so that they could be moved to other workers"""
        nn_self._check_encrypted()
        params = list(nn_self.parameters())
        for p in params:
            p.move(destination)

    crypten.nn.Module.move = module_move_

    def module_get_(nn_self):
        """Overloads crypten.nn instances with get method so that parameters could be sent back to
        owner"""
        nn_self._check_encrypted()
        for p in nn_self.parameters():
            p.get_()

        if isinstance(nn_self.forward, Plan):
            nn_self.forward.get()

        return nn_self

    crypten.nn.Module.get_ = module_get_
    crypten.nn.Module.get = module_get_

    def module_share_(nn_self, *args, **kwargs):
        """Overloads share for crypten.nn.Module."""
        # TODO: add .data and .grad to syft tensors
        nn_self._check_encrypted()
        if module_is_missing_grad(nn_self):
            create_grad_objects(nn_self)

        for p in nn_self.parameters():
            p.share_(*args, **kwargs)

        return nn_self

    crypten.nn.Module.share_ = module_share_
    crypten.nn.Module.share = module_share_

    def module_fix_precision_(nn_self, *args, **kwargs):
        """Overloads fix_precision for crypten.nn.Module."""
        nn_self._check_encrypted()
        if module_is_missing_grad(nn_self):
            create_grad_objects(nn_self)

        for p in nn_self.parameters():
            p.fix_precision_(*args, **kwargs)

        return nn_self

    crypten.nn.Module.fix_precision_ = module_fix_precision_
    crypten.nn.Module.fix_precision = module_fix_precision_
    crypten.nn.Module.fix_prec = module_fix_precision_

    def module_float_precision_(nn_self):
        """Overloads float_precision for crypten.nn.Module, convert fix_precision
        parameters to normal float parameters"""
        # TODO: add .data and .grad to syft tensors
        # if module_is_missing_grad(nn_self):
        #    create_grad_objects(nn_self)
        nn_self._check_encrypted()
        for p in nn_self.parameters():
            p.float_precision_()

        return nn_self

    crypten.nn.Module.float_precision_ = module_float_precision_
    crypten.nn.Module.float_precision = module_float_precision_
    crypten.nn.Module.float_prec = module_float_precision_

    def module_copy(nn_self):
        """Returns a copy of a crypten.nn.Module"""
        nn_self._check_encrypted()
        return copy.deepcopy(nn_self)

    crypten.nn.Module.copy = module_copy

    @property
    def owner(nn_self):
        """Return the owner of the module"""
        nn_self._check_encrypted()
        for p in nn_self.parameters():
            return p.owner

    crypten.nn.Module.owner = owner

    @property
    def location(nn_self):
        """Get the location of the module"""
        nn_self._check_encrypted()
        try:
            for p in nn_self.parameters():
                return p.location
        except AttributeError:
            raise AttributeError(
                "Module has no attribute location, did you already send it to some location?"
            )

    crypten.nn.Module.location = location
