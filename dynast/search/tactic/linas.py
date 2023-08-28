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

from dynast.predictors.predictor_manager import PredictorManager
from dynast.search.evolutionary import (
    EvolutionaryManager,
    EvolutionaryManyObjective,
    EvolutionaryMultiObjective,
    EvolutionarySingleObjective,
)
from dynast.search.tactic.base import NASBaseConfig
from dynast.supernetwork.image_classification.bootstrapnas.bootstrapnas_encoding import BootstrapNASEncoding
from dynast.supernetwork.image_classification.bootstrapnas.bootstrapnas_interface import BootstrapNASRunner
from dynast.supernetwork.image_classification.ofa.ofa_interface import OFARunner
from dynast.supernetwork.machine_translation.transformer_interface import TransformerLTRunner
from dynast.supernetwork.supernetwork_registry import *
from dynast.supernetwork.text_classification.bert_interface import BertSST2Runner
from dynast.utils import log


class LINAS(NASBaseConfig):
    """The LINAS algorithm is a bi-objective optimization approach that explores the sub-networks
    optimization space by iteratively training predictors and using evolutionary algorithms to
    suggest new canditates.
    """

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
        batch_size: int = 128,
        eval_batch_size: int = 128,
        supernet_ckpt_path: str = None,
        device: str = 'cpu',
        test_fraction: float = 1.0,
        dataloader_workers: int = 4,
        metric_eval_fns: dict = None,
        **kwargs,
    ):
        """Params:

        - dataset_path - (str) Path to the dataset needed by the supernetwork runner (e.g., ImageNet is used by OFA)
        - supernet - (str) Super-network name
        - optimization_metrics - (list) List of user-defined optimization metrics that are associated with the supernetwork.
        - measurements - (list) List of metrics that will be measure during sub-network evaluation.
        - num_evals - (int) Number of evaluations to perform during search.
        - search_algo - (str) LINAS low-fidelity search algorithm for the inner optimization loop.
        - population - (int) Population size for each iteration.
        - seed - (int) Random seed.
        - batch_size - (int) Batch size for latency measurement, has a significant impact on latency.
        """
        # TODO(macsz) Update docstring above.
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
            eval_batch_size=eval_batch_size,
            verbose=verbose,
            search_algo=search_algo,
            supernet_ckpt_path=supernet_ckpt_path,
            device=device,
            test_fraction=test_fraction,
            dataloader_workers=dataloader_workers,
            metric_eval_fns=metric_eval_fns,
            **kwargs,
        )

    def train_predictors(self, results_path: str = None):
        """Handles the training of predictors for the LINAS inner-loop based on the
        user-defined optimization metrics.

        If `results_path` is not set, the `self.results_path` will be used instead.
        """

        # Store predictor objects by objective name in a dictionary
        self.predictor_dict = dict()

        # Create/train a predictor for each objective
        for objective in SUPERNET_METRICS[self.supernet]:
            log.debug(
                f'objective: {objective}; optimization metrics: {self.optimization_metrics}; supernet metrics: {SUPERNET_METRICS[self.supernet]};'
            )

            if objective in self.optimization_metrics:
                objective_predictor = PredictorManager(
                    objective_name=objective,
                    results_path=results_path if results_path else self.results_path,
                    supernet_manager=self.supernet_manager,
                    column_names=SUPERNET_METRICS[self.supernet],
                )
                log.info(f'Training {objective} predictor.')
                predictor = objective_predictor.train_predictor()
                log.info(f'Updated self.predictor_dict[{objective}].')
                self.predictor_dict[objective] = predictor
            else:
                self.predictor_dict[objective] = None

    def search(self):
        """Runs the LINAS search"""

        self._init_search()

        # Randomly sample search space for initial population
        latest_population = [self.supernet_manager.random_sample() for _ in range(self.population)]

        # Start Lightweight Iterative Neural Architecture Search (LINAS)
        num_loops = round(self.num_evals / self.population)
        for loop in range(num_loops):
            log.info('Starting LINAS loop {} of {}.'.format(loop + 1, num_loops))

            # High-Fidelity Validation measurements
            for _, individual in enumerate(latest_population):
                log.info(f'Evaluating subnetwork {_+1}/{self.population} in loop {loop+1} of {num_loops}')
                self.validation_interface.eval_subnet(individual)

            # Inner-loop Low-Fidelity Predictor Runner, need to re-instantiate every loop
            self.train_predictors()

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

            elif self.supernet == 'bert_base_sst2':
                runner_predict = BertSST2Runner(
                    supernet=self.supernet,
                    latency_predictor=self.predictor_dict['latency'],
                    macs_predictor=self.predictor_dict['macs'],
                    params_predictor=self.predictor_dict['params'],
                    acc_predictor=self.predictor_dict['accuracy_sst2'],
                    dataset_path=self.dataset_path,
                    checkpoint_path=self.supernet_ckpt_path,
                    device=self.device,
                )
            elif 'bootstrapnas' in self.supernet:
                runner_predict = BootstrapNASRunner(
                    bootstrapnas_supernetwork=self.bootstrapnas_supernetwork,
                    supernet=self.supernet,
                    latency_predictor=self.predictor_dict['latency'],
                    macs_predictor=self.predictor_dict['macs'],
                    params_predictor=self.predictor_dict['params'],
                    acc_predictor=self.predictor_dict['accuracy_top1'],
                    dataset_path=self.dataset_path,
                    batch_size=self.batch_size,
                    device=self.device,
                )
            else:
                raise NotImplementedError
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

        log.info("Validated model architectures in file: {}".format(self.results_path))

        output = list()
        for individual in latest_population:
            param_individual = self.supernet_manager.translate2param(individual)
            if 'bootstrapnas' in self.supernet:
                param_individual = BootstrapNASEncoding.convert_subnet_config_to_bootstrapnas(param_individual)
            output.append(param_individual)

        return output
