import torch
import torch.nn as nn
from typing import Optional
import math


def kld_gaussian(mu, log_sigma, nu=0.0, rho=1.0):
    device = mu.device
    nu = torch.as_tensor(nu, device=device)
    rho = torch.as_tensor(rho, device=device)
    delta_variance = 2.0 * (log_sigma - torch.log(rho))
    variance_term = torch.sum(torch.exp(delta_variance) - delta_variance)
    mean_term = torch.sum((mu - nu) ** 2 / rho)
    return 0.5 * (mean_term + variance_term - 1.0)


def rand_epanechnikov_trig(shape, device, dtype=torch.float32):
    xi = torch.rand(shape,
                    dtype=dtype,
                    device=device)
    xi = 2 * torch.sin(torch.asin(2 * xi - 1) / 3)
    return xi


class VariationalNormalEpanechnikovDropout(nn.Module):
    def __init__(self, input_size, eps=1e-8):
        super().__init__()
        self.input_size = input_size
        self.eps = eps

        self.log_sigma = nn.Parameter(torch.Tensor(input_size))
        self.log_sigma.data.fill_(-1.0)
        self._mean = None
        self._const = 0.5*math.log(90.0*math.pi) - 7./6.
        self._shift = 0.5*math.log(5.0)

    def forward(self, vector, mask=None):
        epsilon = rand_epanechnikov_trig(vector.size(), device=vector.device)
        if mask is not None:
            epsilon = epsilon * mask[..., None]
        variance = torch.exp(self.log_sigma)
        if self.training:
            self._mean = vector
        vector = vector + variance * epsilon
        return vector

    def kld(self, nu=0.0, rho=1.0):
        log_sigma = self.log_sigma - self._shift
        normal_kld = kld_gaussian(self._mean, log_sigma, nu=nu, rho=rho)
        return self._const + normal_kld
