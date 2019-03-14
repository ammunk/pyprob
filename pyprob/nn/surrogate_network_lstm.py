import torch
import torch.nn as nn
from termcolor import colored

from . import EmbeddingFeedForward, InferenceNetwork, SurrogateAddressTransition, SurrogateNormal
from .. import util, state
from ..distributions import Normal, Uniform, Categorical, Poisson
from ..trace import Variable, Trace


class SurrogateNetworkLSTM(InferenceNetwork):
    def __init__(self, lstm_dim=512, lstm_depth=1, sample_embedding_dim=4,
                 address_embedding_dim=64, distribution_type_embedding_dim=8,
                 *args, **kwargs):
        super().__init__(network_type='SurrogateNetworkLSTM', *args, **kwargs)
        self._layers_sample_embedding = nn.ModuleDict()
        self._layers_address_embedding = nn.ParameterDict()
        self._layers_distribution_type_embedding = nn.ParameterDict()
        self._layers_lstm = None
        self._lstm_input_dim = None
        self._lstm_dim = lstm_dim
        self._lstm_depth = lstm_depth
        self._infer_lstm_state = None
        self._sample_embedding_dim = sample_embedding_dim
        self._address_embedding_dim = address_embedding_dim
        self._distribution_type_embedding_dim = distribution_type_embedding_dim
        self._tagged_addresses = []
        self._address_base = {}

        # Surrogate attributes
        self._layers_address_transitions = nn.ModuleDict()
        self._layers_surrogate_distributions = nn.ModuleDict()

    def _init_layers(self):
        self._lstm_input_dim = self._sample_embedding_dim + 2 * (self._address_embedding_dim + self._distribution_type_embedding_dim)
        self._layers_lstm = nn.LSTM(self._lstm_input_dim, self._lstm_dim, self._lstm_depth)
        self._layers_lstm.to(device=util._device)

    def _init_layers_observe_embedding(self, observe_embedding, example_trace):
        pass

    def _embed_observe(self):
        raise NotImplementedError()

    def _pre_generate_layers(self, dataset):
        raise NotImplementedError()

    def _polymorph(self, batch):
        layers_changed = False
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]

            # for surrogate modeling we loop through all variables
            old_address = "__init__"
            for variable in example_trace.variables:
                address = variable.address
                distribution = variable.distribution

                if address not in self._layers_address_embedding:
                    emb = nn.Parameter(util.to_tensor(torch.zeros(self._address_embedding_dim).normal_()))
                    self._layers_address_embedding[address] = emb

                if distribution.name not in self._layers_distribution_type_embedding:
                    emb = nn.Parameter(util.to_tensor(torch.zeros(self._distribution_type_embedding_dim).normal_()))
                    self._layers_distribution_type_embedding[distribution.name] = emb

                if old_address not in self._layers_address_transitions:
                    if not old_address == "__init__":
                        self._layers_address_transitions[old_address] = SurrogateAddressTransition(self._lstm_dim + self._sample_embedding_dim, address)
                    else:
                        self._layers_address_transitions[old_address] = SurrogateAddressTransition(self._lstm_dim + self._sample_embedding_dim, address, first_address=True)
                        layers_changed = True
                else:
                    if address not in self._layers_address_transitions[old_address]._address_to_class:
                        self._layers_address_transitions[old_address].add_address_transition(address)
                        layers_changed = True
                if address not in self._layers_surrogate_distributions:
                    variable_shape = variable.value.shape
                    if isinstance(distribution, Normal):
                        surrogate_distribution = SurrogateNormal(self._lstm_dim, variable_shape)
                        sample_embedding_layer = EmbeddingFeedForward(variable.value.shape, self._sample_embedding_dim, num_layers=1)
                    if isinstance(distribution, Uniform):
                        surrogate_distribution = SurrogateUniform(self._lstm_dim, variable_shape)
                    if isinstance(distribution, Poisson):
                        surrogate_distribution = SurrogatePoisson(self._lstm_dim, variable_shape)
                    if isinstance(distribution, Categorical):
                        surrogate_distribution = SurrogateCategorical(self._lstm_dim, distribution.num_categories)
                        sample_embedding_layer = EmbeddingFeedForward(variable.value.shape,
                                                                      self._sample_embedding_dim,
                                                                      input_is_one_hot_index=True,
                                                                      input_one_hot_dim=distribution.num_categories,
                                                                      num_layers=1)

                    surrogate_distribution.to(device=util._device)
                    sample_embedding_layer.to(device=util._device)
                    self._layers_sample_embedding[address] = sample_embedding_layer
                    self._layers_surrogate_distributions[address] = surrogate_distribution
                    layers_changed = True
                    print('New layers, address: {}, distribution: {}'.format(util.truncate_str(address), distribution.name))

                old_address = address
            # add final address transition that ends the trace
            if address not in self._layers_address_transitions:
                self._layers_address_transitions[address] = SurrogateAddressTransition(self._lstm_dim + self._sample_embedding_dim,
                                                                                       None, last_address=True)

        if layers_changed:
            num_params = sum(p.numel() for p in self.parameters())
            print('Total addresses: {:,}, distribution types: {:,}, parameters: {:,}'.format(len(self._layers_address_embedding), len(self._layers_distribution_type_embedding), num_params))
            self._history_num_params.append(num_params)
            self._history_num_params_trace.append(self._total_train_traces)
        return layers_changed

    def run_lstm_step(self, variable, prev_variable=None):
        success = True
        if prev_variable is None:
            # First time step
            prev_sample_embedding = util.to_tensor(torch.zeros(1, self._sample_embedding_dim))
            prev_address_embedding = util.to_tensor(torch.zeros(self._address_embedding_dim))
            prev_distribution_type_embedding = util.to_tensor(torch.zeros(self._distribution_type_embedding_dim))
            h0 = util.to_tensor(torch.zeros(self._lstm_depth, 1, self._lstm_dim))
            c0 = util.to_tensor(torch.zeros(self._lstm_depth, 1, self._lstm_dim))
            self._lstm_state = (h0, c0)
        else:
            prev_address = prev_variable.address
            prev_distribution = prev_variable.distribution
            prev_value = prev_variable.value
            if prev_value.dim() == 0:
                prev_value = prev_value.unsqueeze(0)
            if prev_address in self._layers_address_embedding:
                prev_sample_embedding = self._layers_sample_embedding[prev_address](prev_value.float())
                prev_address_embedding = self._layers_address_embedding[prev_address]
                prev_distribution_type_embedding = self._layers_distribution_type_embedding[prev_distribution.name]
            else:
                print('Warning: address of previous variable unknown by inference network: {}'.format(prev_address))
                success = False

        current_address = variable.address
        current_distribution = variable.distribution
        if current_address in self._layers_address_embedding:
            current_address_embedding = self._layers_address_embedding[current_address]
            current_distribution_type_embedding = self._layers_distribution_type_embedding[current_distribution.name]
        else:
            print('Warning: address of current variable unknown by inference network: {}'.format(current_address))
            success = False

        if success:
            t = torch.cat([prev_sample_embedding[0],
                           prev_distribution_type_embedding,
                           prev_address_embedding,
                           current_distribution_type_embedding,
                           current_address_embedding]).unsqueeze(0)
            lstm_input = t.unsqueeze(0)
            lstm_output, self._lstm_state = self._layers_lstm(lstm_input, self._lstm_state)
            return success, lstm_output
        else:
            return success, _

    def _loss(self, batch):
        batch_loss = 0
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]
            sub_batch_length = len(sub_batch)
            sub_batch_loss = 0.
            # print('sub_batch_length', sub_batch_length, 'example_trace_length_controlled', example_trace.length_controlled, '  ')

            # Construct LSTM input sequence for the whole trace length of sub_batch
            lstm_input = []
            for time_step in range(example_trace.length):
                current_variable = example_trace.variables[time_step]
                current_address = current_variable.address
                self._address_base[current_address] = "__".join(current_address.split("__")[:-1])

                if current_address not in self._layers_address_embedding and current_address not in self._layers_surrogate_distributions:
                    print(colored('Address unknown by inference network: {}'.format(current_address), 'red', attrs=['bold']))
                    return False, 0
                current_distribution = current_variable.distribution
                current_address_embedding = self._layers_address_embedding[current_address]
                current_distribution_type_embedding = self._layers_distribution_type_embedding[current_distribution.name]

                if time_step == 0:
                    prev_sample_embedding = util.to_tensor(torch.zeros(sub_batch_length, self._sample_embedding_dim))
                    prev_address_embedding = util.to_tensor(torch.zeros(self._address_embedding_dim))
                    prev_distribution_type_embedding = util.to_tensor(torch.zeros(self._distribution_type_embedding_dim))
                else:
                    prev_variable = example_trace.variables_controlled[time_step - 1]
                    prev_address = prev_variable.address
                    if prev_address not in self._layers_address_embedding:
                        print(colored('Address unknown by inference network: {}'.format(prev_address), 'red', attrs=['bold']))
                        return False, 0
                    prev_distribution = prev_variable.distribution
                    smp = util.to_tensor(torch.stack([trace.variables_controlled[time_step - 1].value.float() for trace in sub_batch]))
                    prev_sample_embedding = self._layers_sample_embedding[prev_address](smp)
                    prev_address_embedding = self._layers_address_embedding[prev_address]
                    prev_distribution_type_embedding = self._layers_distribution_type_embedding[prev_distribution.name]

                lstm_input_time_step = []
                for b in range(sub_batch_length):
                    t = torch.cat([prev_sample_embedding[b],
                                   prev_distribution_type_embedding,
                                   prev_address_embedding,
                                   current_distribution_type_embedding,
                                   current_address_embedding])
                    lstm_input_time_step.append(t)
                lstm_input.append(torch.stack(lstm_input_time_step))

            # Execute LSTM in a single operation on the whole input sequence
            lstm_input = torch.stack(lstm_input)
            h0 = util.to_tensor(torch.zeros(self._lstm_depth, sub_batch_length, self._lstm_dim))
            c0 = util.to_tensor(torch.zeros(self._lstm_depth, sub_batch_length, self._lstm_dim))
            lstm_output, _ = self._layers_lstm(lstm_input, (h0, c0))

            # surrogate loss
            surrogate_loss = 0.
            trace_length = example_trace.length
            for time_step in range(trace_length):
                current_variable = example_trace.variables[time_step]
                address = current_variable.address
                if current_variable.tagged:
                    self._tagged_addresses.append(address)
                proposal_input = lstm_output[time_step]
                next_adresses, variable_dist = zip(*[(trace.variables[time_step+1].address
                                                        if time_step < trace_length - 1 else "__end__",
                                                        trace.variables[time_step].distribution)
                                                        for trace in sub_batch])
                address_transition_layer = self._layers_address_transitions[address]
                surrogate_distribution_layer = self._layers_surrogate_distributions[address]

                # only consider loss and training if we are not at the end of trace
                # TODO MAKE SURE TRACE ENDS ARE UNIQUE!
                if time_step < trace_length - 1:
                    smp = util.to_tensor(torch.stack([trace.variables[time_step].value.float() for trace in sub_batch]))
                    sample_embedding = self._layers_sample_embedding[address](smp)
                    address_transition_input = torch.cat([proposal_input, sample_embedding], dim=1)
                    _ = address_transition_layer(address_transition_input)
                    surrogate_loss += torch.sum(address_transition_layer.loss(next_adresses))

                _ = surrogate_distribution_layer(proposal_input)
                surrogate_loss += torch.sum(surrogate_distribution_layer.loss(variable_dist))

            batch_loss += sub_batch_loss + surrogate_loss
        return True, batch_loss / batch.size

    def get_surrogate_forward(self, original_forward=lambda x: x):
        self._original_forward = original_forward
        return self._surrogate_forward

    def _surrogate_forward(self, *args, **kwargs):
        """
        Rewrite the forward function otherwise specified by the user.

        This forward function uses the surrogate model as joint distribution

        """
        # sample initial address
        address = self._layers_address_transitions["__init__"](None).sample()
        prev_variable = None
        while address != "__end__":
            surrogate_dist = self._layers_surrogate_distributions[address]
            current_variable = Variable(distribution=surrogate_dist.dist_type,
                                        address=self._address_base[address],
                                        value=None)
            _, lstm_output = self.run_lstm_step(current_variable, prev_variable)
            address_dist = self._layers_address_transitions[address]
            lstm_output = lstm_output.squeeze(0) # squeeze the sequence dimension
            dist = surrogate_dist(lstm_output)
            # TODO DEAL WITH REUSE???
            value = state.sample(distribution=dist, address=self._address_base[address])
            if address in self._tagged_addresses:
                state.tag(value, address=self._address_base[address])
            prev_variable = Variable(distribution=surrogate_dist.dist_type,
                                     address=self._address_base[address], value=value)

            smp = util.to_tensor(value)
            sample_embedding = self._layers_sample_embedding[address](smp)
            address_transition_input = torch.cat([lstm_output, sample_embedding], dim=1)
            a_dist = address_dist(address_transition_input)
            address = a_dist.sample()

            if address == "unknown":
                # if an address is unknown default to the simulator
                # by resetting the _current_trace
                state._current_trace = Trace()
                self._original_forward(*args, **kwargs)
                break

        return None

