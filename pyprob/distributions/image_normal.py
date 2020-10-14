import torch

from . import Distribution
from .. import util


class ImageNormal(Distribution):
    def __init__(self, mean_image, var):
        self.mean_image = util.to_tensor(mean_image)
        self.var = var * torch.ones_like(self.mean_image)
        try:
            self.dist = torch.distributions.Normal(self.mean_image, self.var)
        except:
            import pdb; pdb.set_trace()
        super().__init__(name='ImageNormal', address_suffix='ImageNormal')

    def sample(self):
        return torch.FloatTensor(self.mean_image)

    def log_prob(self, value, sum=False):
        lp = self.dist.log_prob(util.to_tensor(value))
        return torch.sum(lp) if sum else lp

    def get_input_parameters(self):
        return {'mean_image': self.mean_image, 'var': self.var}
