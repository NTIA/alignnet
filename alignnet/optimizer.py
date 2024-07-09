import torch


class OptimizerWrapper:
    def __init__(self, class_name, lr, **kwargs):
        self._optimizer = eval(class_name)
        self.kwargs = kwargs
        self.kwargs["lr"] = lr

    def optimizer(self, params):
        return self._optimizer(params, **self.kwargs)
