from functools import wraps
import copy

import syft
from syft.execution.plan import Plan

import crypten
import torch as th

methods_to_hook = ["load"]


def hook_plan_building():
    """When builing the plan we should not call directly specific
    methods from CrypTen and as such we return here some "dummy" responses
    only to build the plan.
    """

    f = lambda *args, **kwargs: crypten.cryptensor(th.zeros([]))
    for method_name in methods_to_hook:
        method = getattr(crypten, method_name)
        setattr(crypten, f"native_{method_name}", method)
        setattr(crypten, method_name, f)


def unhook_plan_building():
    """After building the plan we unhook the methods such that
    we call the "real" methods in the actual workers
    """
    for method_name in methods_to_hook:
        method = getattr(crypten, f"native_{method_name}")
        setattr(crypten, method_name, method)


def hook_crypten():
    from syft.frameworks.crypten import load as crypten_load

    setattr(crypten, "load", crypten_load)


def hook_crypten_module():
    """Overloading crypten.nn.Module with PySyft functionality, the primary module
        responsible for core ML functionality such as Neural network layers and
        loss functions.
        It is important to note that all the operations are actually in-place.
    """
    import crypten

    def _check_encrypted(model):
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
        nn_self._check_encrypted()
        params = list(nn_self.parameters())
        for p in params:
            p.move(destination)

    crypten.nn.Module.move = module_move_

    def module_get_(nn_self):
        """Overloads crypten.nn instances with get method so that parameters could be sent back to owner"""
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
        nn_self._check_encrypted()
        for p in nn_self.parameters():
            return p.owner

    crypten.nn.Module.owner = owner

    @property
    def location(nn_self):
        nn_self._check_encrypted()
        try:
            for p in nn_self.parameters():
                return p.location
        except AttributeError:
            raise AttributeError(
                "Module has no attribute location, did you already send it to some location?"
            )

    crypten.nn.Module.location = location
