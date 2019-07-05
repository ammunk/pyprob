import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Distribution, Categorical

class SurrogateCategorical(nn.Module):
    # only support 1 d distributions
    def __init__(self, input_shape, num_categories, constants={}, num_layers=2,
                 hidden_dim=None):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape,
                                        output_shape=torch.Size([num_categories]),
                                        num_layers=num_layers,
                                        activation=torch.relu,
                                        activation_last=None,
                                        hidden_dim=hidden_dim)
        self._total_train_iterations = 0

        self.dist_type = Categorical(probs=torch.Tensor([1]))

    def forward(self, x):
        batch_size = x.size(0)
        x = self._ff(x)
        self.probs = torch.softmax(x, dim=1).view(batch_size, -1) + util._epsilon

        return Categorical(self.probs)

    def loss(self, p_categorical):
        q_categorical = Categorical(self.probs)

        return Distribution.kl_divergence(p_categorical, q_categorical)
