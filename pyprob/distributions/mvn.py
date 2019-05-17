import torch

from . import Distribution
from .. import util


class MultivariateNormal(Distribution):
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None):
        loc = util.to_tensor(loc)
        if covariance_matrix is not None:
            covariance_matrix = util.to_tensor(covariance_matrix)
        elif precision_matrix is not None:
            precision_matrix = util.to_tensor(precision_matrix)
        elif scale_tril is not None:
            scale_tril = util.to_tensor(scale_tril)
            if isinstance(scale_tril, torch.sparse.FloatTensor):
                scale_tril = scale_tril.to_dense()
        super().__init__(name='Normal', address_suffix='Normal', torch_dist=torch.distributions.MultivariateNormal(loc, covariance_matrix, precision_matrix, scale_tril))

    def __repr__(self):
        return 'Normal(mean:{}, scale_tril:{})'.format(self.mean, self._torch_dist.scale_tril)

    def cdf(self, value):
        return self._torch_dist.cdf(value)

    def icdf(self, value):
        return self._torch_dist.icdf(value)
