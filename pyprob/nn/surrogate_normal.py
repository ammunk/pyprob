import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Distribution, Normal

class SurrogateNormal(nn.Module):
    # only support 1 d distributions
    def __init__(self, input_shape, output_shape, num_layers=2, batch_norm=True):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._output_dim = util.prod(output_shape)
        self._output_shape = torch.Size([-1]) + output_shape
        self._ff = EmbeddingFeedForward(input_shape=input_shape,
                                        output_shape=torch.Size([self._output_dim * 2]), num_layers=num_layers,
                                        activation=torch.relu, activation_last=None, batch_norm=batch_norm)
        self._total_train_iterations = 0

        # address transform
        self._transform_mean = lambda dists: torch.stack([d.mean for d in dists])
        self._transform_stddev = lambda dists: torch.stack([d.stddev for d in dists])

        self.dist_type = Normal(loc=0, scale=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self._ff(x)
        self.means = x[:, :self._output_dim].view(self._output_shape)
        self.stddevs = torch.exp(x[:, self._output_dim:]).view(self._output_shape)

        # if we only have one dimensional parameters, squeeze to make them scalars
        if self.means.shape == torch.Size([1]):
            self.means = self.means.squeeze()
            self.stddevs = self.stddevs.squeeze()

        return Normal(self.means, self.stddevs)

    def loss(self, distributions):
        simulator_means = self._transform_mean(distributions)
        simulator_stddevs = self._transform_stddev(distributions)
        p_normal = Normal(simulator_means, simulator_stddevs)
        q_normal = Normal(self.means, self.stddevs)

        return Distribution.kl_divergence(p_normal, q_normal)

    # def old_loss(self, variable_dists):
    #     simulator_means = self._transform_mean(variable_dists)
    #     simulator_stddevs = self._transform_stddev(variable_dists)
    #     inv_stddevs_sqr = torch.reciprocal(torch.pow(self.stddevs,2))
    #     e_sqr = torch.pow(simulator_means,2) + torch.pow(simulator_stddevs,2)
    #     expected_nlog_norm = 0.5*inv_stddevs_sqr*(e_sqr - 2*simulator_means*self.means
    #                                               + torch.pow(self.means,2))
    #     expected_nlog_norm += torch.log(self.stddevs)
    #     return expected_nlog_norm
