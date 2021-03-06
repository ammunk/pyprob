from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np

from . import EmbeddingFeedForward
from .. import util


class ScaledDotProductAttention(nn.Module):
    """ based on https://github.com/jadore801120/attention-is-all-you-need-pytorch
        by Yu-Hsiang Huang
    """
    def __init__(self, temperature=1, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        output = torch.bmm(attn, v)
        return output, attn


class ZeroInputLinear(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output = nn.Parameter(torch.Tensor(output_dim))
        self.output.to(device=util.device)
        self.output.data.normal_()

    def forward(self, input=None):
        """
        input could be e.g. an empty tensor an empty Tensor, but is not used
        """
        return self.output


class PrevSamplesEmbedder(nn.Module):
    def __init__(self,
                 input_dim,
                 sample_embedding_dim=16,
                 n_queries=4,
                 key_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.sample_embedding_dim = sample_embedding_dim
        self.key_dim = key_dim
        self.n_queries = n_queries
        self.store_att_weights = False

        # initialise neural nets
        self.dpa = ScaledDotProductAttention()
        self._query_layers = nn.ModuleDict()
        self._key_layers = nn.ModuleDict()
        self._value_layers = nn.ModuleDict()


    def _add_address(self, address, value_shape, *embedder_args, **embedder_kwargs):
        """
        Add a new query layer and key layer for a previously unseen address.
        Should only be used in `polymorph`
        """
        if self.input_dim > 0:
            self._query_layers[address] = nn.Linear(
                self.input_dim,
                self.key_dim*self.n_queries
            )
        else:
            self._query_layers[address] = ZeroInputLinear(
                self.key_dim*self.n_queries
            )
        self._key_layers[address] = \
            EmbeddingFeedForward(value_shape, self.key_dim, hidden_dim=50,
                                 *embedder_args, **embedder_kwargs)
        self._value_layers[address] = \
            EmbeddingFeedForward(value_shape,
                                 self.sample_embedding_dim, hidden_dim=50,
                                 *embedder_args, **embedder_kwargs)
        self._query_layers[address].to(device=util._device)
        self._key_layers.to(device=util._device)
        self._value_layers.to(device=util._device)

    def enable_weight_storing(self):
        self.store_att_weights = True
        self.attention_weights = []

    def disable_weight_storing(self):
        self.store_att_weights = False
        self.attention_weights = []

    def init_for_trace(self):
        """
        Initialises state for beginning of a new inference trace.
         - removes all sampled values
        """
        self.empty = True
        self.keys = torch.Tensor([]).to(device=util._device)    # -1 x n_keys x key_dim
        self.values = torch.Tensor([]).to(device=util._device)  # -1 x n_keys x sample_embedding_dim
        if self.store_att_weights:
            if not self.attention_weights:
                self.attention_weights = [OrderedDict()]
            else:
                self.attention_weights.append(OrderedDict())

    def add_value(self, address, values):
        """
        Add an embedding of a sampled value. This will be stored and can be
        accessed later when __forward__ is run.
        """
        self.empty = False
        batch_size = 1 if len(values.shape) == 0 else values.size(0)
        keys = self._key_layers[address](values.view(batch_size, -1)).view(batch_size,
                                                                           1, -1)
        values = self._value_layers[address](values.view(batch_size, -1)).view(batch_size,
                                                                               1, -1)
        self.keys = torch.cat((self.keys, keys), dim=1)
        self.values = torch.cat((self.values, values), dim=1)

    def store_attention_weights(self, address, attn):
        self.attention_weights[-1][address] = attn

    def forward(self, address, queries, batch_size=None):
        """
        Returns weighted sum of sample embeddings in `Attention Is All You
        Need` style (https://arxiv.org/pdf/1706.03762.pdf).
        Queries:    f_address(input)
                     - n_queries x key_dim
        Keys:       f_address(sample_embedding) for each (address, embedding)
                    pair
                     - n_keys x key_dim
        Values:     weighted sum of embeddings
                     - n_keys x sample_embedding_dim
        """
        if self.empty:
            return torch.zeros(batch_size,
                               self.n_queries*self.sample_embedding_dim).to(device=util._device)
        if len(queries) == 0:
            queries = [torch.Tensor([0])] * self.n_queries
        queries = self._query_layers[address](queries).view(-1, self.n_queries, self.key_dim)
        output, attn = self.dpa(queries, self.keys, self.values)

        if self.store_att_weights:
            self.store_attention_weights(address, attn)

        return output.view(-1, self.n_queries*self.sample_embedding_dim)

    def get_output_dim(self):
        return self.n_queries*self.sample_embedding_dim
