import torch
import time
import sys
import os
import math
import random
from termcolor import colored
import numpy as np

from .distributions import Empirical
from . import util, state, TraceMode, PriorInflation, InferenceEngine, \
    InferenceNetwork, Optimizer, LearningRateScheduler, AddressDictionary, \
    ImportanceWeighting
from .nn import InferenceNetwork as InferenceNetworkBase
from .nn import SurrogateNetworkLSTM as SurrogateNetworkBase
from .nn import OnlineDataset, OfflineDataset, InferenceNetworkFeedForward, \
    InferenceNetworkLSTM, SurrogateNetworkLSTM
from .remote import ModelServer


class Model():
    def __init__(self, name='Unnamed pyprob model', address_dict_file_name=None):
        super().__init__()
        self.name = name
        self._inference_network = None
        self._surrogate_network = None
        self._surrogate_forward = None
        self._original_forward = self.forward
        if address_dict_file_name is None:
            self._address_dictionary = None
        else:
            self._address_dictionary = AddressDictionary(address_dict_file_name)

    def forward(self):
        raise NotImplementedError()

    def _estimate_z_inv(self, trace, rs_address, num_estimate_samples, *args, **kwargs):
        if num_estimate_samples < 1:
            return 1

        rejection_length_list = []
        for _ in range(num_estimate_samples):
            try:
                partial_trace = state.RSPartialTrace(trace, rs_address)
                state._begin_trace(rs_partial_trace=partial_trace)
                self.forward(*args, **kwargs)
            except state.RejectionEndException as e:
                rejection_length_list.append(e.length)
        assert len(rejection_length_list) == num_estimate_samples, f'When estimating Z, all function calls should throw an exception ({len(rejection_length_list)} != {num_estimate_samples})'
        return np.mean([x for x in rejection_length_list])

    def _estimate_z(self, trace, rs_address, num_estimate_samples, *args, **kwargs):
        if num_estimate_samples < 1:
            return 1

        total_samples = 0
        accepted_samples = 0
        while total_samples < num_estimate_samples:
            try:
                partial_trace = state.RSPartialTrace(trace, rs_address)
                state._begin_trace(rs_partial_trace=partial_trace)
                self.forward(*args, **kwargs)
            except state.RejectionEndException as e:
                if total_samples + e.length <= num_estimate_samples:
                    total_samples += e.length
                    accepted_samples += 1
                else: # Before accepting in this rejection sampling run, we reach the required number of samples
                    total_samples = num_estimate_samples
        estimate = accepted_samples / total_samples
        return estimate

    def _trace_generator(self, trace_mode=TraceMode.PRIOR,
                         prior_inflation=PriorInflation.DISABLED,
                         inference_engine=InferenceEngine.IMPORTANCE_SAMPLING,
                         inference_network=None, observe=None, metropolis_hastings_trace=None,
                         likelihood_importance=1.,
                         proposal=None, importance_weighting=ImportanceWeighting.IW2, num_z_estimate_samples=None, num_z_inv_estimate_samples=None,
                         *args, **kwargs):
        while True:
            state._init_traces(func=self.forward, trace_mode=trace_mode,
                            prior_inflation=prior_inflation, inference_engine=inference_engine,
                            inference_network=inference_network, observe=observe,
                            metropolis_hastings_trace=metropolis_hastings_trace,
                            address_dictionary=self._address_dictionary,
                            likelihood_importance=likelihood_importance,
                            importance_weighting=importance_weighting)
            state._begin_trace()
            result = self.forward(*args, **kwargs)
            trace = state._end_trace(result)

            ## Fix trace weights
            if trace_mode == TraceMode.POSTERIOR and importance_weighting == ImportanceWeighting.IW1 and (inference_engine==InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK or proposal is not None):
                for rs_address, rs_entry in trace.rs_entries_dict_address.items():
                    pass
                    state._init_traces(trace_mode=trace_mode, func=self.forward, prior_inflation=prior_inflation,
                                       inference_engine=inference_engine, inference_network=inference_network,
                                       observe=observe, metropolis_hastings_trace=metropolis_hastings_trace,
                                       address_dictionary=self._address_dictionary, likelihood_importance=likelihood_importance,
                                       importance_weighting=importance_weighting)
                    z_q_estimate = self._estimate_z(trace, rs_address, num_z_estimate_samples,
                                                    *args, **kwargs)

                    state._init_traces(trace_mode=TraceMode.PRIOR, func=self.forward, prior_inflation=prior_inflation,
                                       inference_engine=inference_engine, inference_network=inference_network,
                                       observe=observe, metropolis_hastings_trace=metropolis_hastings_trace,
                                       address_dictionary=self._address_dictionary, likelihood_importance=likelihood_importance,
                                       importance_weighting=importance_weighting)
                    z_p_inv_estimate = self._estimate_z_inv(trace, rs_address, num_z_inv_estimate_samples,
                                                            *args, **kwargs)

                    rs_entry.log_importance_weight = float(np.log(z_q_estimate * z_p_inv_estimate))
                    rs_entry.log_prob = float(np.log(z_p_inv_estimate))
                    # trace.refresh_weights_and_dictionaries should be called here, but since it will be called after discarding rejected samples, we leave it for then.

            if trace_mode == TraceMode.PRIOR_FOR_INFERENCE_NETWORK or\
                (trace_mode == TraceMode.POSTERIOR and inference_engine in [InferenceEngine.IMPORTANCE_SAMPLING, InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK] and importance_weighting in [ImportanceWeighting.IW1, ImportanceWeighting.IW0]):
                trace.discard_rejected()
            yield trace

    def _traces(self, num_traces=10, trace_mode=TraceMode.PRIOR,
                prior_inflation=PriorInflation.DISABLED,
                inference_engine=InferenceEngine.IMPORTANCE_SAMPLING,
                inference_network=None, map_func=None, silent=False,
                observe=None, file_name=None, likelihood_importance=1., *args,
                **kwargs):

        generator = self._trace_generator(trace_mode=trace_mode,
                                          prior_inflation=prior_inflation,
                                          inference_engine=inference_engine,
                                          inference_network=inference_network, observe=observe,
                                          likelihood_importance=likelihood_importance,
                                          *args, **kwargs)

        traces = Empirical(file_name=file_name)

        if map_func is None:
            map_func = lambda trace: trace

        time_start = time.time()

        if (util._verbosity > 1) and not silent:
            len_str_num_traces = len(str(num_traces))
            print('Time spent  | Time remain.| Progress             | {} | Traces/sec'.format('Trace'.ljust(len_str_num_traces * 2 + 1)))
            prev_duration = 0

        for i in range(num_traces):
            if (util._verbosity > 1) and not silent:
                duration = time.time() - time_start
                if (duration - prev_duration > util._print_refresh_rate) or (i == num_traces - 1):
                    prev_duration = duration
                    traces_per_second = (i + 1) / duration
                    print('{} | {} | {} | {}/{} | {:,.2f}       '.format(util.days_hours_mins_secs_str(duration),
                                                                         util.days_hours_mins_secs_str((num_traces - i) / traces_per_second),
                                                                         util.progress_bar(i+1, num_traces),
                                                                         str(i+1).rjust(len_str_num_traces),
                                                                         num_traces, traces_per_second), end='\r')
                    sys.stdout.flush()

            trace = next(generator)

            if trace_mode == TraceMode.PRIOR:
                log_weight = 1.
            else:
                log_weight = trace.log_importance_weight

            traces.add(map_func(trace), log_weight)

        if (util._verbosity > 1) and not silent:
            print()
        traces.finalize()
        return traces

    def get_trace(self, *args, **kwargs):
        return next(self._trace_generator(*args, **kwargs))

    def joint(self, num_traces=10,
              prior_inflation=PriorInflation.DISABLED, map_func=None, file_name=None,
              likelihood_importance=1., surrogate=False, *args, **kwargs):

        if surrogate and self._surrogate_network:
            self._surrogate_network.eval()
            self._surrogate_network.to(device=torch.device('cpu'))
            self.forward = self._surrogate_forward
        elif surrogate and not self._surrogate_network:
            raise RuntimeError("Surrogate model not trained")
        else:
            self.forward = self._original_forward

        with torch.no_grad():
            prior = self._traces(num_traces=num_traces, trace_mode=TraceMode.PRIOR,
                                prior_inflation=prior_inflation, map_func=map_func,
                                file_name=file_name,
                                likelihood_importance=likelihood_importance, *args, **kwargs)

            prior.rename('Prior, traces: {:,}'.format(prior.length))

            # prior.add_metadata(op='prior', num_traces=num_traces,
            #                 prior_inflation=str(prior_inflation),
            #                 likelihood_importance=likelihood_importance)
        return prior

    def posterior(self, num_traces=10,
                  inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, initial_trace=None,
                  map_func=None, observe=None, file_name=None, thinning_steps=None,
                  likelihood_importance=1., surrogate=False, *args, **kwargs):

        if surrogate and self._surrogate_network:
            self._surrogate_network.eval()
            self._surrogate_network.to(device=torch.device('cpu'))
            self.forward = self._surrogate_forward
        elif surrogate and not self._surrogate_network:
            raise RuntimeError("Surrogate model not trained")
        else:
            self.forward = self._original_forward

        with torch.no_grad():
            if inference_engine == InferenceEngine.IMPORTANCE_SAMPLING:
                posterior = self._traces(num_traces=num_traces,
                                        trace_mode=TraceMode.POSTERIOR,
                                        inference_engine=inference_engine,
                                        inference_network=None, map_func=map_func,
                                        observe=observe, file_name=file_name,
                                        likelihood_importance=likelihood_importance,
                                        *args, **kwargs)

                posterior.rename('Posterior, IS, traces: {:,}, ESS: {:,.2f}'.format(posterior.length, posterior.effective_sample_size))

                # posterior.add_metadata(op='posterior', num_traces=num_traces,
                #                     inference_engine=str(inference_engine),
                #                     effective_sample_size=posterior.effective_sample_size,
                #                     likelihood_importance=likelihood_importance)

            elif inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
                if self._inference_network is None:
                    raise RuntimeError('Cannot run inference engine IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK because no inference network for this model is available. Use learn_inference_network or load_inference_network first.')
                else:
                    self._inference_network.to(device=torch.device('cpu'))

                with torch.no_grad():
                    posterior = self._traces(num_traces=num_traces,
                                            trace_mode=TraceMode.POSTERIOR,
                                            inference_engine=inference_engine,
                                            inference_network=self._inference_network,
                                            map_func=map_func,
                                            observe=observe, file_name=file_name,
                                            likelihood_importance=likelihood_importance,
                                            *args, **kwargs)

                posterior.rename('Posterior, IC, traces: {:,}, train. traces: {:,}, ESS: {:,.2f}'.format(posterior.length, self._inference_network._total_train_traces, posterior.effective_sample_size))
                # posterior.add_metadata(op='posterior', num_traces=num_traces,
                #                     inference_engine=str(inference_engine),
                #                     effective_sample_size=posterior.effective_sample_size,
                #                     likelihood_importance=likelihood_importance,
                #                    train_traces=self._inference_network._total_train_traces)
            else:  # inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS

                posterior = Empirical(file_name=file_name)
                if map_func is None:
                    map_func = lambda trace: trace
                if initial_trace is None:
                    current_trace = next(self._trace_generator(trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, observe=observe, *args, **kwargs))
                else:
                    current_trace = initial_trace

                time_start = time.time()
                traces_accepted = 0
                samples_reused = 0
                samples_all = 0
                if thinning_steps is None:
                    thinning_steps = 1

                if util._verbosity > 1:
                    len_str_num_traces = len(str(num_traces))
                    print('Time spent  | Time remain.| Progress             | {} | Accepted|Smp reuse| Traces/sec'.format('Trace'.ljust(len_str_num_traces * 2 + 1)))
                    prev_duration = 0

                for i in range(num_traces):
                    if util._verbosity > 1:
                        duration = time.time() - time_start
                        if (duration - prev_duration > util._print_refresh_rate) or (i == num_traces - 1):
                            prev_duration = duration
                            traces_per_second = (i + 1) / duration
                            print('{} | {} | {} | {}/{} | {} | {} | {:,.2f}       '.format(util.days_hours_mins_secs_str(duration), util.days_hours_mins_secs_str((num_traces - i) / traces_per_second), util.progress_bar(i+1, num_traces), str(i+1).rjust(len_str_num_traces), num_traces, '{:,.2f}%'.format(100 * (traces_accepted / (i + 1))).rjust(7), '{:,.2f}%'.format(100 * samples_reused / max(1, samples_all)).rjust(7), traces_per_second), end='\r')

                            sys.stdout.flush()

                    candidate_trace = next(self._trace_generator(trace_mode=TraceMode.POSTERIOR,
                                            inference_engine=inference_engine,
                                            metropolis_hastings_trace=current_trace,
                                            observe=observe,
                                            *args, **kwargs))

                    current_controlled_len = len([v for v in current_trace.variables if v.control])
                    candidate_controlled = [v for v in candidate_trace.variables if v.control]

                    log_acceptance_ratio = (math.log(current_controlled_len)
                                            - math.log(len(candidate_controlled))
                                            + candidate_trace.log_prob_observed
                                            - current_trace.log_prob_observed)

                    for variable in candidate_controlled:
                        if variable.reused:
                            log_acceptance_ratio += torch.sum(variable.log_prob)
                            log_acceptance_ratio -= torch.sum(current_trace.variables_dict_address[variable.address].log_prob)
                            samples_reused += 1

                    samples_all += len(candidate_controlled)

                    if state._metropolis_hastings_site_transition_log_prob is None:
                        print(colored('Warning: trace did not hit the Metropolis Hastings site, ensure that the model is deterministic except pyprob.sample calls', 'red', attrs=['bold']))
                    else:
                        log_acceptance_ratio += torch.sum(state._metropolis_hastings_site_transition_log_prob)

                    # print(log_acceptance_ratio)
                    if math.log(random.random()) < float(log_acceptance_ratio):
                        traces_accepted += 1
                        current_trace = candidate_trace
                    # do thinning
                    if i % thinning_steps == 0:
                        posterior.add(map_func(current_trace))

                if util._verbosity > 1:
                    print()

                posterior.finalize()
                posterior.rename('Posterior, {}, traces: {:,}{}, accepted: {:,.2f}%, sample reuse: {:,.2f}%'.format('LMH' if inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS else 'RMH', posterior.length, '' if thinning_steps == 1 else ' (thinning steps: {:,})'.format(thinning_steps), 100 * (traces_accepted / num_traces), 100 * samples_reused / samples_all))
                # posterior.add_metadata(op='posterior', num_traces=num_traces,
                #                        inference_engine=str(inference_engine),
                #                        likelihood_importance=likelihood_importance,
                #                        thinning_steps=thinning_steps,
                #                        num_traces_accepted=traces_accepted,
                #                        num_samples_reuised=samples_reused,
                #                        num_samples=samples_all)
        return posterior

    def posterior_return(self, num_traces=10, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, initial_trace=None, observe=None, file_name=None, thinning_steps=None, *args, **kwargs):
        return self.posterior(num_traces=num_traces, inference_engine=inference_engine, initial_trace=initial_trace, map_func=lambda trace: trace.result, observe=observe, file_name=file_name, thinning_steps=thinning_steps, *args, **kwargs)

    def joint_return(self, num_traces=10,
                     prior_inflation=PriorInflation.DISABLED, file_name=None,
                     likelihood_importance=1., surrogate=False, *args, **kwargs):
        return self.joint(num_traces=num_traces, prior_inflation=prior_inflation, map_func=lambda trace: trace.result, file_name=file_name, likelihood_importance=likelihood_importance, surrogate=surrogate, *args, **kwargs)

    def reset_inference_network(self):
        self._inference_network = None

    def reset_surrogate_network(self):
        self.forward = self._original_forward
        self._surrogate_network = None

    def learn_inference_network(self, num_traces, num_traces_end=1e9,
                                inference_network=InferenceNetwork.FEEDFORWARD,
                                prior_inflation=PriorInflation.DISABLED,
                                prev_sample_attention=False,
                                prev_sample_attention_kwargs={},
                                dataset_dir=None, dataset_valid_dir=None,
                                observe_embeddings={}, batch_size=64,
                                valid_size=None, valid_every=None,
                                optimizer_type=Optimizer.ADAM,
                                learning_rate_init=0.001,
                                learning_rate_end=1e-6,
                                learning_rate_scheduler_type=LearningRateScheduler.NONE,
                                momentum=0.9, weight_decay=0.,
                                save_file_name_prefix=None, save_every_sec=600,
                                pre_generate_layers=True,
                                distributed_backend=None,
                                distributed_params_sync_every_iter=10000,
                                distributed_num_buckets=None, num_workers=0,
                                stop_with_bad_loss=True, log_file_name=None,
                                lstm_dim=512, lstm_depth=1,
                                address_embedding_dim=64,
                                sample_embedding_dim=4,
                                variable_embeddings={},
                                distribution_type_embedding_dim=8,
                                proposal_mixture_components=10, surrogate=False,
                                sacred_run=None):

        state._set_observed_from_inf(list(observe_embeddings.keys()))

        if surrogate and self._surrogate_forward:
            self._surrogate_network.eval()
            # make surrogate run on CPU in order to train inference network with GPU
            self._surrogate_network.to(device=torch.device('cpu'))
            self.forward = self._surrogate_forward
        elif surrogate and not self._surrogate_forward:
            raise RuntimeError('The model has no trained surrogate network.')
        else:
            self.forward = self._original_forward

        if dataset_dir is None:
            dataset = OnlineDataset(model=self, prior_inflation=prior_inflation,
                                    variables_observed_inf_training=state._variables_observed_inf_training)
        else:
            dataset = OfflineDataset(dataset_dir, state._variables_observed_inf_training)

        if dataset_valid_dir is None:
            dataset_valid = None
        else:
            dataset_valid = OfflineDataset(dataset_valid_dir, state._variables_observed_inf_training)

        if self._inference_network is None:
            print('Creating new inference network...')
            if inference_network == InferenceNetwork.FEEDFORWARD:
                self._inference_network = InferenceNetworkFeedForward(model=self,
                                                                      observe_embeddings=observe_embeddings,
                                                                      prev_sample_attention=prev_sample_attention,
                                                                      prev_sample_attention_kwargs=prev_sample_attention_kwargs,
                                                                      proposal_mixture_components=proposal_mixture_components)
            elif inference_network == InferenceNetwork.LSTM:
                self._inference_network = InferenceNetworkLSTM(model=self,
                                                               prev_sample_attention=prev_sample_attention,
                                                               prev_sample_attention_kwargs=prev_sample_attention_kwargs,
                                                               observe_embeddings=observe_embeddings, lstm_dim=lstm_dim,
                                                               sample_embedding_dim=sample_embedding_dim,
                                                               address_embedding_dim=address_embedding_dim,
                                                               distribution_type_embedding_dim=distribution_type_embedding_dim,
                                                               lstm_depth=lstm_depth, variable_embeddings=variable_embeddings,
                                                               proposal_mixture_components=proposal_mixture_components)
            else:
                raise ValueError('Unknown inference_network: {}'.format(inference_network))
            if pre_generate_layers:
                if dataset_valid_dir is not None:
                    self._inference_network._pre_generate_layers(dataset_valid,
                                                                 save_file_name_prefix=save_file_name_prefix)
                if dataset_dir is not None:
                    self._inference_network._pre_generate_layers(dataset,
                                                                 save_file_name_prefix=save_file_name_prefix)
        else:
            print('Continuing to train existing inference network...')
            print('Total number of parameters: {:,}'.format(self._inference_network._history_num_params[-1]))

        self._inference_network.to(device=util._device)
        self._inference_network.optimize(num_traces=num_traces, dataset=dataset,
                                         dataset_valid=dataset_valid,
                                         num_traces_end=num_traces_end,
                                         batch_size=batch_size, valid_every=valid_every,
                                         optimizer_type=optimizer_type,
                                         learning_rate_init=learning_rate_init,
                                         learning_rate_end=learning_rate_end,
                                         learning_rate_scheduler_type=learning_rate_scheduler_type,
                                         momentum=momentum, weight_decay=weight_decay,
                                         save_file_name_prefix=save_file_name_prefix,
                                         save_every_sec=save_every_sec,
                                         distributed_backend=distributed_backend,
                                         distributed_params_sync_every_iter=distributed_params_sync_every_iter,
                                         distributed_num_buckets=distributed_num_buckets,
                                         num_workers=num_workers,
                                         stop_with_bad_loss=stop_with_bad_loss,
                                         log_file_name=log_file_name, sacred_run=sacred_run)


    def learn_surrogate_inference_network(self, num_traces, num_traces_end=1e9,
                                          prior_inflation=PriorInflation.DISABLED,
                                          dataset_dir=None,
                                          dataset_valid_dir=None, batch_size=64,
                                          valid_size=None, valid_every=None,
                                          optimizer_type=Optimizer.ADAM,
                                          learning_rate_init=0.001,
                                          learning_rate_end=1e-6,
                                          learning_rate_scheduler_type=LearningRateScheduler.NONE,
                                          momentum=0.9, weight_decay=0.,
                                          save_file_name_prefix=None,
                                          save_every_sec=600,
                                          pre_generate_layers=False,
                                          distributed_backend=None,
                                          distributed_params_sync_every_iter=10000,
                                          distributed_num_buckets=10,
                                          num_workers=0,
                                          stop_with_bad_loss=True,
                                          lstm_dim=512, lstm_depth=1,
                                          variable_embeddings={},
                                          address_embedding_dim=64,
                                          sample_embedding_dim=4,
                                          distribution_type_embedding_dim=8,
                                          log_file_name=None, ic=False, sacred_run=None):

        if dataset_dir is None:
            dataset = OnlineDataset(model=self, prior_inflation=prior_inflation)
        else:
            dataset = OfflineDataset(dataset_dir)

        if dataset_valid_dir is None:
            dataset_valid = None
        else:
            dataset_valid = OfflineDataset(dataset_valid_dir)

        if self._surrogate_network is  None:
            print('Creating new surrogate network...')
            self._surrogate_network = SurrogateNetworkLSTM(model=self,
                                                           lstm_dim=lstm_dim,
                                                           lstm_depth=lstm_depth,
                                                           sample_embedding_dim=sample_embedding_dim,
                                                           address_embedding_dim=address_embedding_dim,
                                                           distribution_type_embedding_dim=distribution_type_embedding_dim,
                                                           variable_embeddings=variable_embeddings)
            if pre_generate_layers:
                if dataset_valid_dir is not None:
                    self._inference_network._pre_generate_layers(dataset_valid,
                                                                 save_file_name_prefix=save_file_name_prefix)
                if dataset_dir is not None:
                    self._inference_network._pre_generate_layers(dataset,
                                                                 save_file_name_prefix=save_file_name_prefix)
        else:
            print('Continuing to train existing surrogate network...')
            print('Total number of parameters: {:,}'.format(self._surrogate_network._history_num_params[-1]))

        self._surrogate_network.to(device=util._device)
        self._surrogate_network.optimize(num_traces=num_traces, dataset=dataset,
                                         dataset_valid=dataset_valid,
                                         num_traces_end=num_traces_end,
                                         batch_size=batch_size,
                                         valid_every=valid_every,
                                         optimizer_type=optimizer_type,
                                         learning_rate_init=learning_rate_init,
                                         learning_rate_end=learning_rate_end,
                                         learning_rate_scheduler_type=learning_rate_scheduler_type,
                                         momentum=momentum,
                                         weight_decay=weight_decay,
                                         save_file_name_prefix=save_file_name_prefix,
                                         save_every_sec=save_every_sec,
                                         distributed_backend=distributed_backend,
                                         distributed_params_sync_every_iter=distributed_params_sync_every_iter,
                                         distributed_num_buckets=distributed_num_buckets,
                                         num_workers=num_workers,
                                         stop_with_bad_loss=stop_with_bad_loss,
                                         log_file_name=log_file_name, sacred_run=sacred_run)

        self._original_forward = self.forward
        self._surrogate_forward = self._surrogate_network.get_surrogate_forward(self._original_forward)
        self.forward = self._surrogate_forward
        print('Finished training surrogate model.')

    def get_surrogate_network(self):
        if self._surrogate_network:
            return self._surrogate_network
        else:
            raise RuntimeError('The model has no trained surrogate network.')

    def save_surrogate_network(self, file_name):
        if self._surrogate_network is None:
            raise RuntimeError('The model has no trained surrogate network.')
        self._surrogate_network._save(file_name)

    def load_surrogate_network(self, file_name):
        self._surrogate_network = SurrogateNetworkBase._load(file_name)
        self._surrogate_forward = self._surrogate_network.get_surrogate_forward(self._original_forward)
        # The following is due to a temporary hack related with https://github.com/pytorch/pytorch/issues/9981 and can be deprecated by using dill as pickler with torch > 0.4.1
        self._surrogate_network._model = self

    def get_inference_network(self):
        if self._inference_network:
            return self._inference_network
        else:
            raise RuntimeError("The model has no trained inference network.")

    def save_inference_network(self, file_name):
        if self._inference_network is None:
            raise RuntimeError('The model has no trained inference network.')
        self._inference_network._save(file_name)

    def load_inference_network(self, file_name, load_rng_state=False):
        self._inference_network = InferenceNetworkBase._load(file_name, load_rng_state)
        # The following is due to a temporary hack related with https://github.com/pytorch/pytorch/issues/9981 and can be deprecated by using dill as pickler with torch > 0.4.1
        self._inference_network._model = self

    def save_dataset(self, dataset_dir, num_traces, num_traces_per_file,
                     prior_inflation=PriorInflation.DISABLED, surrogate=False,
                     *args, **kwargs):

        if surrogate and self._surrogate_forward:
            self._surrogate_network.eval()
            # make surrogate run on CPU in order to train inference network with GPU
            self._surrogate_network.to(device=torch.device('cpu'))
            self.forward = self._surrogate_forward
        elif surrogate and not self._surrogate_forward:
            raise RuntimeError('The model has no trained surrogate network.')
        else:
            self.forward = self._original_forward

        if not os.path.exists(dataset_dir):
            print('Directory does not exist, creating: {}'.format(dataset_dir))
            os.makedirs(dataset_dir)
        dataset = OnlineDataset(self, None, prior_inflation=prior_inflation)
        dataset.save_dataset(dataset_dir=dataset_dir, num_traces=num_traces,
                             num_traces_per_file=num_traces_per_file, *args, **kwargs)

class RemoteModel(Model):
    def __init__(self, server_address='tcp://127.0.0.1:5555', before_forward_func=None, after_forward_func=None, *args, **kwargs):
        self._server_address = server_address
        self._model_server = None
        self._before_forward_func = before_forward_func  # Optional mthod to run before each forward call of the remote model (simulator)
        self._after_forward_func = after_forward_func  # Optional method to run after each forward call of the remote model (simulator)
        super().__init__(*args, **kwargs)

    def close(self):
        if self._model_server is not None:
            self._model_server.close()
        super().close()

    def forward(self):
        if self._model_server is None:
            self._model_server = ModelServer(self._server_address)
            self.name = '{} running on {}'.format(self._model_server.model_name, self._model_server.system_name)

        if self._before_forward_func is not None:
            self._before_forward_func()
        ret = self._model_server.forward()  # Calls the forward run of the remove model (simulator)
        if self._after_forward_func is not None:
            self._after_forward_func()
        return ret
