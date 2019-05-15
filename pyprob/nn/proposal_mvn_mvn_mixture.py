import torch
import torch.nn as nn

from . import EmbeddingFeedForward
from .. import util
from ..distributions import MultivariateNormal, Mixture

class ProposalMVNMVNMixture(nn.Module):
    def __init__(self, input_shape, output_shape, num_layers=2, mixture_components=10):
        super().__init__()
        # Currently only supports event_shape=torch.Size([]) for the mixture components
        self._K = mixture_components
        input_shape = util.to_size(input_shape)
        self._D = util.prod(output_shape)
        self.scale_tril_dim = int(self._D*(self._D+1)/2)
        self._ff = EmbeddingFeedForward(input_shape=input_shape,
                                        output_shape=torch.Size([(self._D+self.scale_tril_dim+1)*self._K]),
                                        num_layers=num_layers,
                                        activation=torch.relu,
                                        activation_last=None)
        self._total_train_iterations = 0
        self.diag_mask = torch.eye(self._D).byte()

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        # extract prior info
        prior_means = torch.stack([v.distribution.mean for v in prior_variables]).view(batch_size, -1)
        prior_stddevs = torch.stack([torch.sqrt(v.distribution.variance) for v in prior_variables]).view(batch_size, -1)

        prior_means = prior_means.expand_as(means)
        prior_stddevs = prior_stddevs.expand_as(means)

        # forward
        x = self._ff(x)
        means = x[:, :self._D*self._K].view(batch_size, -1)
        # scale and move mean
        means = prior_means + (means * prior_stddevs)

        scale_tril_slice = (self._D + self._scale_tril_dim)*self._K
        scale_tril = x[:, self._D*self._K:scale_tril_slice].view(batch_size, self._K, self._D, self._D)
        coeffs = x[:, scale_tril_slice:].view(batch_size, -1)

        #scale stddevs (i.e. diagonal of lower triangular matrix)
        scale_tril = scale_tril.diagonal(dim1=-2, dim2=-1).exp_().mul_(prior_stddevs)

        coeffs = torch.softmax(coeffs, dim=1)

        means = means.view(batch_size, -1)
        distributions = [MultivariateNormal(means[:, k:k+self._D], scale_tril=Ls[:,k,:,:]) for k in range(self._K)]
        return Mixture(distributions, coeffs)
