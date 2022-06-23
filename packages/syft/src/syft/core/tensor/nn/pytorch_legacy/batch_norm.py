# third party
import numpy as np
from torch import Tensor
from torch import nn

# relative
from ..autodp.phi_tensor import PhiTensor


def simple_batchnorm(image: np.ndarray, scaler=1, eps: float = 1e-5) -> np.ndarray:
    # Assumes Beta parameter = 0
    return (image - image.mean()) / np.sqrt(image.var() + eps) * scaler


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.func = nn.BatchNorm2d(
            num_features=self.num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
        )

    def forward(self, image: PhiTensor):
        data = self.func(Tensor(image.child))
        minv, maxv = image.min_vals.data, image.max_vals.data
        public_avg = 0.5 * (maxv + minv)
        # Assumption: max and min are equally distant from public_avg -> var = 0.25 * (max - min)**2
        public_var = 0.25 * (maxv - minv) ** 2

        return PhiTensor(
            child=data.detach().numpy(),
            data_subjects=image.data_subjects,
            min_vals=(minv - public_avg) / np.sqrt(public_var + self.eps),
            max_vals=(maxv - public_avg) / np.sqrt(public_var + self.eps),
        )

    def parameters(self):
        return self.func.parameters()


def np_BatchNorm2d(input_tensor: PhiTensor, scaler=1, eps: float = 1e-5) -> PhiTensor:
    # TODO: Think this through a lil' more- output max is when var is highest, output min is when var is lowest
    # var = 0 when all data = min or max, var = public_var when all data is equally distributed b/w min/max
    # Odd vs even shape effects; var = max when all data EXCEPT 1 is max, but 1 is min
    image = input_tensor.child
    minv, maxv = input_tensor.min_vals.data, input_tensor.max_vals.data
    public_avg = 0.5 * (maxv + minv)
    # Assumption: max and min are equally distant from public_avg -> var = 0.25 * (max - min)**2
    public_var = 0.25 * (maxv - minv) ** 2
    return PhiTensor(
        child=(image - image.mean()) / np.sqrt(image.var() + eps) * scaler,
        min_vals=(minv - public_avg) / np.sqrt(public_var + eps) * scaler,
        max_vals=(maxv - public_avg) / np.sqrt(public_var + eps) * scaler,
        data_subjects=input_tensor.data_subjects,
    )
