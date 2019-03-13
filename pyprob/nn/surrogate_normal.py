import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Normal

class SurrogateNormal(nn.Module):
    # only support 1 d distributions
    def __init__(self, input_shape, output_shape, num_layers=2):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._output_dim = util.prod(output_shape)
        self._output_shape = torch.Size([-1]) + output_shape
        self._ff = EmbeddingFeedForward(input_shape=input_shape,
                                        output_shape=torch.Size([self._output_dim * 2]), num_layers=num_layers,
                                        activation=torch.relu, activation_last=None)
        self._total_train_iterations = 0

        # address transform
        self._transform_mean = Compose([Lambda(lambda dists: [d.mean for d in dists]),
                                        torch.Tensor,
                                        ])
        self._transform_stddev = Compose([Lambda(lambda dists: [d.stddev for d in dists]),
                                          torch.Tensor,
                                         ])

        self.dist_type = Normal(loc=0,scale=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self._ff(x)
        self.means = x[:, :self._output_dim].view(self._output_shape)
        self.stddevs = torch.exp(x[:, self._output_dim:]).view(self._output_shape)

        return Normal(self.means, self.stddevs)

    def loss(self, variable_dists):
        simulator_means = self._transform_mean(variable_dists)
        simulator_stddevs = self._transform_stddev(variable_dists)
        inv_stddevs_sqr = torch.reciprocal(torch.pow(self.stddevs,2))
        e_sqr = torch.pow(simulator_means,2) + torch.pow(simulator_stddevs,2)
        expected_nlog_norm = 0.5*inv_stddevs_sqr*(e_sqr - 2*simulator_means*self.means
                                                  + torch.pow(self.means,2))
        expected_nlog_norm += torch.log(self.stddevs)
        return expected_nlog_norm
