import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Distribution, Uniform

class SurrogateUniform(nn.Module):
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
        self._transform_low = lambda dists: torch.stack([d.low for d in dists])
        self._transform_high = lambda dists: torch.stack([d.high for d in dists])

        self.dist_type = Uniform(low=0,high=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self._ff(x)
        self.low = x[:, :self._output_dim].view(self._output_shape)
        self.high = torch.exp(x[:, self._output_dim:]).view(self._output_shape)

        return Uniform(self.low, self.high)

    def loss(self, distributions):
        simulator_lows = self._transform_low(distributions)
        simulator_highs = self._transform_high(distributions)
        p_normal = Uniform(simulator_lows, simulator_highs)
        q_normal = Uniform(self.low, self.high)

        return Distribution.kl_divergence(p_normal, q_normal)
