import torch
import torch.nn as nn

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Normal, Mixture


class ProposalNormalNormalMixture(nn.Module):
    def __init__(self, input_shape, output_shape, num_layers=2, mixture_components=10):
        super().__init__()
        # Currently only supports event_shape=torch.Size([]) for the mixture components
        self._mixture_components = mixture_components
        self._D = util.prod(output_shape)
        self._K = mixture_components
        if self._D > 1:
            hidden_dim = int(self._D/100)
        else:
            hidden_dim = None
        input_shape = util.to_size(input_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape,
                                        output_shape=torch.Size([(self._D*2 + 1)*self._K]),
                                        hidden_dim=hidden_dim,
                                        num_layers=num_layers, activation=torch.relu,
                                        activation_last=None)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        means = x[:, :self._K*self._D].view(batch_size, -1)
        stddevs = x[:, self._K*self._D:self._K*self._D*2].view(batch_size, -1)
        coeffs = x[:, 2*self._K*self._D:].view(batch_size, -1)
        stddevs = torch.exp(stddevs)
        coeffs = torch.softmax(coeffs, dim=1)
        # the squeeze is when a trace is a mix of surrogate and original samples
        # the dimensions might be slightly off
        prior_means = torch.stack([v.distribution.mean.squeeze() for v in prior_variables]).view(batch_size, -1)
        prior_stddevs = torch.stack([v.distribution.stddev.squeeze() for v in prior_variables]).view(batch_size, -1)
        prior_means = prior_means.repeat(1, self._K)
        prior_stddevs = prior_stddevs.repeat(1, self._K)
        means = prior_means + (means * prior_stddevs)
        stddevs = stddevs * prior_stddevs
        means = means.view(batch_size, -1)
        stddevs = stddevs.view(batch_size, -1)
        distributions = [Normal(means[:, i*self._D:(i+1)*self._D].view(batch_size,-1),
                                stddevs[:, i*self._D:(i+1)*self._D].view(batch_size,-1)) for i in range(self._K)]
        return Mixture(distributions, coeffs)
