# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import torch.distributed as dist

from dynast.search.evolutionary import (
    EvolutionaryManager,
    EvolutionaryManyObjective,
    EvolutionaryMultiObjective,
    EvolutionarySingleObjective,
)
from dynast.search.tactic.linas import LINAS
from dynast.supernetwork.image_classification.ofa.ofa_interface import OFARunner
from dynast.supernetwork.machine_translation.transformer_interface import TransformerLTRunner
from dynast.supernetwork.supernetwork_registry import *
from dynast.utils import log, split_list
from dynast.utils.distributed import get_distributed_vars, get_worker_results_path, is_main_process


class LINASDistributed(LINAS):
    def __init__(
        self,
        supernet: str,
        optimization_metrics: list,
        measurements: list,
        num_evals: int,
        results_path: str,
        dataset_path: str = None,
        verbose: bool = False,
        search_algo: str = 'nsga2',
        population: int = 50,
        seed: int = 42,
        batch_size: int = 1,
        supernet_ckpt_path: str = None,
        device: str = 'cpu',
        test_fraction: float = 1.0,
        dataloader_workers: int = 4,
        **kwargs,
    ):
        self.main_results_path = results_path
        LOCAL_RANK, WORLD_RANK, WORLD_SIZE, DIST_METHOD = get_distributed_vars()
        results_path = get_worker_results_path(results_path, WORLD_RANK)

        super().__init__(
            dataset_path=dataset_path,
            supernet=supernet,
            optimization_metrics=optimization_metrics,
            measurements=measurements,
            num_evals=num_evals,
            results_path=results_path,
            seed=seed,
            population=population,
            batch_size=batch_size,
            verbose=verbose,
            search_algo=search_algo,
            supernet_ckpt_path=supernet_ckpt_path,
            device=device,
            test_fraction=test_fraction,
            dataloader_workers=dataloader_workers,
        )

    def search(self):
        """Runs the LINAS search"""

        self._init_search()

        LOCAL_RANK, WORLD_RANK, WORLD_SIZE, DIST_METHOD = get_distributed_vars()

        # START - Initial population
        if is_main_process():
            log.info('Creating data')
            # Randomly sample search space for initial population
            latest_population = [self.supernet_manager.random_sample() for _ in range(self.population)]
            data = split_list(latest_population, WORLD_SIZE)
        else:
            data = [None for _ in range(WORLD_SIZE)]

        output_list = [None]
        dist.scatter_object_list(output_list, data, src=0)

        latest_population = output_list[0]

        # END - Initial population

        # Start Lightweight Iterative Neural Architecture Search (LINAS)
        num_loops = round(self.num_evals / self.population)
        for loop in range(num_loops):
            log.info('Starting LINAS loop {} of {}.'.format(loop + 1, num_loops))

            # High-Fidelity Validation measurements
            for _, individual in enumerate(latest_population):
                log.info(
                    f'Evaluating subnetwork {_+1}/{len(latest_population)} [{self.population}] in loop {loop+1} of {num_loops}'
                )
                results = self.validation_interface.eval_subnet(individual)

            # This will act as `dist.barrier()`; plus, we get results paths from all workers.
            data = {
                'from': WORLD_RANK,
                'results_path': self.results_path,
            }

            outputs = [None for _ in range(WORLD_SIZE)]
            dist.all_gather_object(outputs, data)
            # Inner-loop Low-Fidelity Predictor Runner, need to re-instantiate every loop

            if is_main_process():
                worker_results_paths = [o['results_path'] for o in outputs]
                combined_csv = pd.concat([pd.read_csv(f) for f in worker_results_paths])
                combined_csv.to_csv(self.main_results_path, index=False)
                log.info(f'Saving combined results to {self.main_results_path}')

                self.train_predictors(results_path=self.main_results_path)

                if self.supernet in [
                    'ofa_resnet50',
                    'ofa_mbv3_d234_e346_k357_w1.0',
                    'ofa_mbv3_d234_e346_k357_w1.2',
                    'ofa_proxyless_d234_e346_k357_w1.3',
                ]:
                    runner_predict = OFARunner(
                        supernet=self.supernet,
                        latency_predictor=self.predictor_dict['latency'],
                        macs_predictor=self.predictor_dict['macs'],
                        params_predictor=self.predictor_dict['params'],
                        acc_predictor=self.predictor_dict['accuracy_top1'],
                        dataset_path=self.dataset_path,
                        device=self.device,
                        dataloader_workers=self.dataloader_workers,
                        test_fraction=self.test_fraction,
                    )
                elif self.supernet == 'transformer_lt_wmt_en_de':
                    runner_predict = TransformerLTRunner(
                        supernet=self.supernet,
                        latency_predictor=self.predictor_dict['latency'],
                        macs_predictor=self.predictor_dict['macs'],
                        params_predictor=self.predictor_dict['params'],
                        acc_predictor=self.predictor_dict['bleu'],
                        dataset_path=self.dataset_path,
                        checkpoint_path=self.supernet_ckpt_path,
                    )

                # Setup validation interface
                prediction_interface = EVALUATION_INTERFACE[self.supernet](
                    evaluator=runner_predict,
                    manager=self.supernet_manager,
                    optimization_metrics=self.optimization_metrics,
                    measurements=self.measurements,
                    csv_path=None,
                    predictor_mode=True,
                )

                if self.num_objectives == 1:
                    problem = EvolutionarySingleObjective(
                        evaluation_interface=prediction_interface,
                        param_count=self.supernet_manager.param_count,
                        param_upperbound=self.supernet_manager.param_upperbound,
                    )
                    if self.search_algo == 'cmaes':
                        search_manager = EvolutionaryManager(
                            algorithm='cmaes',
                            seed=self.seed,
                            n_obj=self.num_objectives,
                            verbose=self.verbose,
                        )
                        search_manager.configure_cmaes(num_evals=LINAS_INNERLOOP_EVALS[self.supernet])
                    else:
                        search_manager = EvolutionaryManager(
                            algorithm='ga',
                            seed=self.seed,
                            n_obj=self.num_objectives,
                            verbose=self.verbose,
                        )
                        search_manager.configure_ga(
                            population=self.population,
                            num_evals=LINAS_INNERLOOP_EVALS[self.supernet],
                        )
                elif self.num_objectives == 2:
                    problem = EvolutionaryMultiObjective(
                        evaluation_interface=prediction_interface,
                        param_count=self.supernet_manager.param_count,
                        param_upperbound=self.supernet_manager.param_upperbound,
                    )
                    if self.search_algo == 'age':
                        search_manager = EvolutionaryManager(
                            algorithm='age',
                            seed=self.seed,
                            n_obj=self.num_objectives,
                            verbose=self.verbose,
                        )
                        search_manager.configure_age(
                            population=self.population, num_evals=LINAS_INNERLOOP_EVALS[self.supernet]
                        )
                    else:
                        search_manager = EvolutionaryManager(
                            algorithm='nsga2',
                            seed=self.seed,
                            n_obj=self.num_objectives,
                            verbose=self.verbose,
                        )
                        search_manager.configure_nsga2(
                            population=self.population, num_evals=LINAS_INNERLOOP_EVALS[self.supernet]
                        )
                elif self.num_objectives == 3:
                    problem = EvolutionaryManyObjective(
                        evaluation_interface=prediction_interface,
                        param_count=self.supernet_manager.param_count,
                        param_upperbound=self.supernet_manager.param_upperbound,
                    )
                    if self.search_algo == 'ctaea':
                        search_manager = EvolutionaryManager(
                            algorithm='ctaea',
                            seed=self.seed,
                            n_obj=self.num_objectives,
                            verbose=self.verbose,
                        )
                        search_manager.configure_ctaea(num_evals=LINAS_INNERLOOP_EVALS[self.supernet])
                    elif self.search_algo == 'moead':
                        search_manager = EvolutionaryManager(
                            algorithm='moead',
                            seed=self.seed,
                            n_obj=self.num_objectives,
                            verbose=self.verbose,
                        )
                        search_manager.configure_moead(num_evals=LINAS_INNERLOOP_EVALS[self.supernet])
                    else:
                        search_manager = EvolutionaryManager(
                            algorithm='unsga3',
                            seed=self.seed,
                            n_obj=self.num_objectives,
                            verbose=self.verbose,
                        )
                        search_manager.configure_unsga3(
                            population=self.population, num_evals=LINAS_INNERLOOP_EVALS[self.supernet]
                        )
                else:
                    log.error('Number of objectives not supported. Update optimization_metrics!')

                results = search_manager.run_search(problem)

                latest_population = results.pop.get('X')

                data = split_list(latest_population, WORLD_SIZE)
            else:
                data = [None for _ in range(WORLD_SIZE)]

            output_list = [None]
            dist.scatter_object_list(output_list, data, src=0)

            latest_population = output_list[0]

        log.info("Validated model architectures in file: {}".format(self.results_path))

        output = list()
        for individual in latest_population:
            output.append(self.supernet_manager.translate2param(individual))
        return output
