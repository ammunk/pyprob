import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Categorical

class SurrogateAddressTransition(nn.Module):

    def __init__(self, input_shape, next_address=None, num_layers=2, first_address=False, last_address=False):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._output_shape = torch.Size([2])
        self._first_address = first_address
        self._last_address = last_address
        if next_address:
            self._addresses = [next_address]
            self._address_to_class = {next_address: torch.Tensor([0])}
        else:
            self._addresses = ["__end"]
            self._address_to_class = {"__end": torch.Tensor([0])}
        # one class placeholder
        self._ff = EmbeddingFeedForward(input_shape=input_shape,
                                        output_shape=self._output_shape,
                                        num_layers=num_layers,
                                        activation=torch.relu,
                                        activation_last=None)
        self._softmax = nn.Softmax(dim=1)

        self._total_train_iterations = 0
        self._n_classes = 1

        self._update_transforms()


    def forward(self, x):
        if self._first_address:
            self._categorical = AddressCategorical(probs=[1], n_classes=self._n_classes, transform=self._transform_to_address)
        elif self._last_address:
            self._categorical = Categorical(probs=[1])
            self._categorical = AddressCategorical(probs=[1], n_classes=self._n_classes, transform=self._transform_to_address)
        else:
            batch_size = x.size(0)
            x = self._ff(x)
            self.probs = self._softmax(x)
            self._categorical = AddressCategorical(probs=self.probs, n_classes=self._n_classes, transform=self._transform_to_address)

        return self._categorical

    def add_address_transition(self, new_address):
        pass

    def loss(self, next_addresses):
        classes = self._transform_to_class(next_addresses).to(device=util._device)
        loss = -self._categorical.log_prob(classes)
        return loss

    def _update_transforms(self):
        # address transform
        self._transform_to_class = Compose([Lambda(lambda next_addresses: [self._address_to_class[next_address]
                                                                           for next_address in next_addresses]),
                                            torch.Tensor,])
        self._transform_to_address = Lambda(lambda address_class: self._addresses[address_class])


class AddressCategorical(Categorical):
    def __init__(self, probs=None, n_classes=0, transform=None):
        super().__init__(probs=probs)
        self._transform_to_address = transform
        self._n_classes = n_classes

    def sample(self):
        c = super().sample()
        if c == self._n_classes:
            return "unknown"
        else:
            return self._transform_to_address(c)
