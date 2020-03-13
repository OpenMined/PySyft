"""maintains an optimizer for each worker"""

from collections import defaultdict


class FLOptimier:
    """Creates a remote optimizer object
    which manage an optimizer for each worker"""

    def __init__(self, optimizer_class, **kwargs):
        """
        Args:
            optimizer_class: class of the pytorch optimizer
            kwargs: arguments to be forwarded to the optimizer class
        """
        self.optimizer_class = optimizer_class
        self.opt_dict = defaultdict()
        self.kwargs = kwargs

    def get_optimizer(self, model):
        """ adds an optimizer for the worker and returns the optimizer

        Args:
            model: model belonging to a worker
        """
        if hasattr(model, "location"):
            opt = self.opt_dict.setdefault(
                model.location, self.optimizer_class(model.parameters(), **self.kwargs)
            )
            return opt
        opt = self.opt_dict.setdefault(
            "central", self.optimizer_class(model.parameters(), **self.kwargs)
        )
        return opt
