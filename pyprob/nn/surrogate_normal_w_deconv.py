import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Distribution, Normal

class SurrogateNormalConvTranspose2d(nn.Module):
    def __init__(self, input_shape, mean_shape, var_shape, num_layers=2):
        super().__init__()
        input_shape = util.to_size(input_shape)
        H_input = util.prod(input_shape)
        W_input = 1
        self._mean_output_dim = util.prod(mean_shape)
        self._var_output_dim = util.prod(var_shape)
        self._mean_output_shape = torch.Size([-1]) + mean_shape
        self._var_output_shape = torch.Size([-1]) + var_shape
        self._ff = EmbeddingFeedForward(input_shape=input_shape,
                                        output_shape=torch.Size([H_input + self._var_output_dim]), num_layers=num_layers,
                                        hidden_dim = H_input + self._var_output_dim,
                                        activation=torch.relu, activation_last=None)
        H = mean_shape[0]
        W = mean_shape[1]
        self.deconv = nn.Sequential([
            nn.ConvTranspose2d(1,1, kernel_size=(50, 3)),
            nn.ConvTranspose2d(1,1, kernel_size=(50, 8)),
            nn.ConvTranspose2d(1,1, kernel_size=(100, 10)),
            nn.ConvTranspose2d(1,1, kernel_size=(H-(H_input+200 + 3)+1, W-(W_input+3+8+10+3)+1))
        ])
        self._total_train_iterations = 0

        self.dist_type = Normal(loc=0, scale=1)

    def _transform_mean(self, dists):
        return torch.stack([d.mean for d in dists])

    def _transform_stddev(self, dists):
        return  torch.stack([d.stddev for d in dists])

    def forward(self, x):
        batch_size = x.size(0)
        x = self._ff(x)
        tmp = x[:, :H_input].view(batch_size, 1, -1 ,1)
        self.means = self.deconv(tmp)

        self.stddevs = torch.exp(x[:, self._mean_output_dim:]).view(self._var_output_shape)

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
