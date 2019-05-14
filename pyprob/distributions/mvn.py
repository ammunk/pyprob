import torch

from . import Distribution
from .. import util


class MultivariateNormal(Distribution):
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None):
        loc = util.to_tensor(loc)
        if covariance_matrix:
            covariance_matrix = util.to_tensor(covariance_matrix)
        if precision_matrix:
            precision_matrix = util.to_tensor(precision_matrix)
        if scale_tril:
            scale_tril = util.to_tensor(scale_tril)
        super().__init__(name='Normal', address_suffix='Normal', torch_dist=torch.distributions.MultivariateNormal(loc, covariance_matrix, precision_matrix, scale_tril))

    def __repr__(self):
        return 'Normal(mean:{}, covariance_matrix:{})'.format(self.mean, self.covariance_matrix)

    def cdf(self, value):
        return self._torch_dist.cdf(value)

    def icdf(self, value):
        return self._torch_dist.icdf(value)
