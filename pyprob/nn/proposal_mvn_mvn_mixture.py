import torch
import torch.nn as nn

from . import EmbeddingFeedForward
from .. import util
from ..distributions import MultivariateNormal, Mixture

class ProposalMVNMVNMixture(nn.Module):
    def __init__(self, input_shape, output_shape, num_layers=2, rank=2, mixture_components=10):
        super().__init__()
        # Currently only supports event_shape=torch.Size([]) for the mixture components
        self._K = mixture_components
        input_shape = util.to_size(input_shape)
        self._D = util.prod(output_shape)
        if rank:
            if rank > self._D or rank < 1:
                raise ValueError('Rank outside range. Rank has to satisfy: 0 < rank <= D')
            else:
                self._scale_tril_dim = int(rank*(rank +1)/2 + rank*(self._D-rank)) - rank + self._D
                self._low_rank = True
        else:
            self._low_rank = False
            self._scale_tril_dim = int(self._D*(self._D+1)/2)
        self._ff = EmbeddingFeedForward(input_shape=input_shape,
                                        output_shape=torch.Size([(self._D+self._scale_tril_dim+1)*self._K]),
                                        num_layers=num_layers,
                                        hidden_dim=int(self._D/100),
                                        activation=torch.relu,
                                        activation_last=None)
        self._total_train_iterations = 0
        self.eye = torch.eye(self._D)
        low_rank_upper = torch.LongTensor([[d, d]   for d in range(self._D)]
                                          + [[r, c] for r in range(1,rank)
                                                    for c in range(r)])
        low_rank_lower = torch.LongTensor([[r, c] for r in range(rank,self._D)
                                                  for c in range(rank)])

        self._sparse_index = torch.cat([low_rank_upper, low_rank_lower], dim=0)

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        # forward
        x = self._ff(x)
        means = x[:, :self._D*self._K].view(batch_size, self._K, -1)

        # extract prior info
        prior_means = torch.stack([v.distribution.mean for v in prior_variables]).view(batch_size, -1)
        prior_stddevs = torch.stack([torch.sqrt(v.distribution.variance) for v in prior_variables]).view(batch_size, -1)


        # scale and move mean
        means = prior_means.unsqueeze(1) + (means * prior_stddevs.unsqueeze(1))

        if self._low_rank:
            scale_trils = [0]*self._K
            jump = self._scale_tril_dim
            for k in range(self._K):
                scale_tril_slice = jump*(k+1) + self._D*self._K
                low_rank_tril = x[:, self._D*self._K + jump*k: scale_tril_slice].view(batch_size,-1)
                #scale stddevs (i.e. diagonal of lower triangular matrix)
                low_rank_tril[:, :self._D].exp_().mul_(prior_stddevs)
                sparse_index = torch.cat([torch.arange(batch_size).unsqueeze(1).repeat_interleave(self._sparse_index.shape[0],dim=0),
                                          self._sparse_index.repeat(batch_size, 1)], dim=1)
                scale_tril = torch.sparse.FloatTensor(sparse_index.t(),
                                                      low_rank_tril.flatten(),
                                                      torch.Size([batch_size, self._D, self._D]))
                scale_trils[k] = scale_tril
            coeffs = x[:, jump*self._K+self._D*self._K:].view(batch_size, -1)

            # scale_tril_slice = self._K*self._D + (self._scale_tril_dim)*self._K
            # low_rank_tril = x[:, self._D*self._K: scale_tril_slice].view(batch_size,-1)
            # low_rank_tril[:, :self._D*self._K].exp_().mul_(prior_stddevs.repeat_interleave(self._K,dim=1))
            # sparse_index = torch.cat([torch.arange(batch_size).unsqueeze(1).repeat_interleave(self._sparse_index.shape[0],dim=0),
            #                           self._sparse_index.repeat(batch_size, 1)], dim=1)
            # scale_tril = torch.sparse.FloatTensor(sparse_index.t(),
            #                                       low_rank_tril.flatten(),
            #                                      torch.Size([batch_size, self._K,self._D, self._D]))
        else:
            scale_tril_slice = (self._D + self._scale_tril_dim)*self._K
            scale_tril = x[:, self._D*self._K:scale_tril_slice].view(batch_size, self._K, self._D, self._D).view(batch_size,-1)
            #scale stddevs (i.e. diagonal of lower triangular matrix)
            scale_tril.diagonal(dim1=-2, dim2=-1).exp_().mul_(prior_stddevs)

            coeffs = x[:, scale_tril_slice:].view(batch_size, -1)


        coeffs = torch.softmax(coeffs, dim=1)

        if self._low_rank:
            distributions = [MultivariateNormal(means[:, k, :], scale_tril=scale_trils[k]) for k in range(self._K)]
        else:
            distributions = [MultivariateNormal(means[:, k, :], scale_tril=scale_tril[:,k,:,:]) for k in range(self._K)]
        return Mixture(distributions, coeffs)
