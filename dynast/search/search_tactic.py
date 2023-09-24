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


import os

import pandas as pd
import torch.distributed as dist

from dynast.predictors.predictor_manager import PredictorManager
from dynast.search.evolutionary import (
    EvolutionaryManager,
    EvolutionaryManyObjective,
    EvolutionaryMultiObjective,
    EvolutionarySingleObjective,
)
from dynast.supernetwork.image_classification.bootstrapnas.bootstrapnas_encoding import BootstrapNASEncoding
from dynast.supernetwork.image_classification.bootstrapnas.bootstrapnas_interface import BootstrapNASRunner
from dynast.supernetwork.image_classification.ofa.ofa_interface import OFARunner
from dynast.supernetwork.image_classification.vit.vit_interface import ViTRunner
from dynast.supernetwork.machine_translation.transformer_interface import TransformerLTRunner
from dynast.supernetwork.supernetwork_registry import *
from dynast.supernetwork.text_classification.bert_interface import BertSST2Runner
from dynast.utils import LazyImport, log, split_list
from dynast.utils.distributed import get_distributed_vars, get_worker_results_path, is_main_process
from dynast.utils.exceptions import InvalidMetricsException, InvalidSupernetException

QuantizedOFARunner = LazyImport(
    'dynast.supernetwork.image_classification.ofa_quantization.quantization_interface.QuantizedOFARunner'
)


class NASBaseConfig:
    """Base class that supports the various search tactics such as LINAS and Evolutionary"""

    def __init__(
        self,
        dataset_path: str = None,
        supernet: str = 'ofa_mbv3_d234_e346_k357_w1.0',
        optimization_metrics: list = ['latency', 'accuracy_top1'],
        measurements: list = ['latency', 'macs', 'params', 'accuracy_top1'],
        num_evals: int = 1000,
        results_path: str = 'results.csv',
        seed: int = 42,
        population: int = 50,
        batch_size: int = 128,
        eval_batch_size: int = 128,
        verbose: bool = False,
        search_algo: str = 'nsga2',
        supernet_ckpt_path: str = None,
        device: str = 'cpu',
        test_fraction: float = 1.0,
        mp_calibration_samples: int = 100,
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
        - mp_calibration_samples - (int) How many samples to use to calibrate the mixed precision model.
        """
        # TODO(macsz) Update docstring above.

        self.dataset_path = dataset_path
        self.supernet = supernet
        self.optimization_metrics = optimization_metrics
        self.measurements = measurements
        self.num_evals = num_evals
        self.results_path = results_path
        self.seed = seed
        self.population = population
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.verbose = verbose
        self.search_algo = search_algo
        self.supernet_ckpt_path = supernet_ckpt_path
        self.device = device
        self.mp_calibration_samples = mp_calibration_samples
        self.dataloader_workers = dataloader_workers
        self.test_fraction = test_fraction
        self.metric_eval_fns = metric_eval_fns

        self.bootstrapnas_supernetwork = kwargs.get('bootstrapnas_supernetwork', None)

        self.verify_measurement_types()
        self.format_csv_header()
        self.init_supernet()

        if kwargs:
            log.debug('Passed unused parameters: {}'.format(kwargs))

    def verify_measurement_types(self):
        # Remove duplicates
        self.optimization_metrics = list(set(self.optimization_metrics))
        self.num_objectives = len(self.optimization_metrics)
        self.measurements = list(set(self.measurements))

        # Check that measurements counts are correct
        if self.num_objectives > 3 or self.num_objectives < 1:
            log.error('Incorrect number of optimization objectives specified. Must be 1, 2, or 3.')

        if self.supernet in SUPERNET_METRICS:
            valid_metrics = SUPERNET_METRICS[self.supernet]
            for metric in self.optimization_metrics:
                if metric not in valid_metrics:
                    raise InvalidMetricsException(self.supernet, metric)
                self.measurements.append(metric)

            for metric in self.measurements:
                if metric not in valid_metrics:
                    raise InvalidMetricsException(self.supernet, metric)

        else:
            raise InvalidSupernetException(self.supernet)

        self.measurements = list(set(self.measurements))
        self.optimization_metrics = list(set(self.optimization_metrics))

    def format_csv_header(self):
        self.csv_header = get_csv_header(self.supernet)

        log.info(f'Results csv file header ordering will be: {self.csv_header}')

    def init_supernet(self):
        # Initializes the super-network manager
        if self.bootstrapnas_supernetwork:
            param_dict = self.bootstrapnas_supernetwork.get_search_space()
        else:
            param_dict = SUPERNET_PARAMETERS[self.supernet]
        self.supernet_manager = SUPERNET_ENCODING[self.supernet](param_dict=param_dict, seed=self.seed)

    def _init_search(self):
        if self.supernet in [
            'ofa_resnet50',
            'ofa_mbv3_d234_e346_k357_w1.0',
            'ofa_mbv3_d234_e346_k357_w1.2',
            'ofa_proxyless_d234_e346_k357_w1.3',
        ]:
            self.runner_validate = OFARunner(
                supernet=self.supernet,
                dataset_path=self.dataset_path,
                batch_size=self.batch_size,
                eval_batch_size=self.eval_batch_size,
                device=self.device,
                dataloader_workers=self.dataloader_workers,
                test_fraction=self.test_fraction,
            )
        elif self.supernet == 'transformer_lt_wmt_en_de':
            # TODO(macsz) Add `test_fraction`
            # TODO(macsz) Add `eval_batch_size`
            self.runner_validate = TransformerLTRunner(
                supernet=self.supernet,
                dataset_path=self.dataset_path,
                batch_size=self.batch_size,
                checkpoint_path=self.supernet_ckpt_path,
            )
        elif self.supernet == 'bert_base_sst2':
            # TODO(macsz) Add `test_fraction`
            # TODO(macsz) Add `eval_batch_size`
            self.runner_validate = BertSST2Runner(
                supernet=self.supernet,
                dataset_path=self.dataset_path,
                batch_size=self.batch_size,
                checkpoint_path=self.supernet_ckpt_path,
                device=self.device,
            )
        elif self.supernet == 'vit_base_imagenet':
            self.runner_validate = ViTRunner(
                supernet=self.supernet,
                dataset_path=self.dataset_path,
                batch_size=self.batch_size,
                eval_batch_size=self.eval_batch_size,
                checkpoint_path=self.supernet_ckpt_path,
                device=self.device,
                test_fraction=self.test_fraction,
            )
        elif 'bootstrapnas' in self.supernet:
            self.runner_validate = BootstrapNASRunner(
                bootstrapnas_supernetwork=self.bootstrapnas_supernetwork,
                supernet=self.supernet,
                dataset_path=self.dataset_path,
                batch_size=self.batch_size,
                eval_batch_size=self.eval_batch_size,
                device=self.device,
                metric_eval_fns=self.metric_eval_fns,
            )
        elif self.supernet == 'inc_quantization_ofa_resnet50':
            self.runner_validate = QuantizedOFARunner(
                supernet=self.supernet,
                dataset_path=self.dataset_path,
                batch_size=self.batch_size,
                eval_batch_size=self.eval_batch_size,
                device=self.device,
                dataloader_workers=self.dataloader_workers,
                test_fraction=self.test_fraction,
                mp_calibration_samples=self.mp_calibration_samples,
            )
        else:
            log.error(f'Missing interface and runner for supernet: {self.supernet}!')
            raise NotImplementedError

        # Setup validation interface
        self.validation_interface = EVALUATION_INTERFACE[self.supernet](
            evaluator=self.runner_validate,
            manager=self.supernet_manager,
            optimization_metrics=self.optimization_metrics,
            measurements=self.measurements,
            csv_path=self.results_path,
        )

        # Clear csv file if one exists
        self.validation_interface.format_csv(self.csv_header)

    def get_best_configs(self, sort_by: str = None, ascending: bool = False, limit: int = None):
        """Returns the best sub-networks.

        Number of returned networks is controlled by the `limit` parameter. If it's not set, then
        `self.population` is used instead.
        """
        limit = self.population if limit is None else limit
        df = pd.read_csv(self.results_path).tail(limit)

        if self.csv_header is not None:
            df.columns = self.csv_header

        if sort_by is not None:
            df = df.sort_values(by=sort_by, ascending=ascending)

        if 'bootstrapnas' in self.supernet:
            df['Sub-network'] = df['Sub-network'].apply(BootstrapNASEncoding.convert_subnet_config_to_bootstrapnas)
        return df


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
        mp_calibration_samples: int = 100,
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
            mp_calibration_samples=mp_calibration_samples,
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
            elif self.supernet == 'vit_base_imagenet':
                runner_predict = ViTRunner(
                    supernet=self.supernet,
                    latency_predictor=self.predictor_dict['latency'],
                    macs_predictor=self.predictor_dict['macs'],
                    params_predictor=self.predictor_dict['params'],
                    acc_predictor=self.predictor_dict['accuracy_top1'],
                    dataset_path=self.dataset_path,
                    checkpoint_path=self.supernet_ckpt_path,
                    batch_size=self.batch_size,
                    device=self.device,
                )

            elif self.supernet == 'inc_quantization_ofa_resnet50':
                runner_predict = QuantizedOFARunner(
                    supernet=self.supernet,
                    latency_predictor=self.predictor_dict['latency'],
                    model_size_predictor=self.predictor_dict['model_size'],
                    params_predictor=self.predictor_dict['params'],
                    acc_predictor=self.predictor_dict['accuracy_top1'],
                    dataset_path=self.dataset_path,
                    device=self.device,
                    dataloader_workers=self.dataloader_workers,
                    test_fraction=self.test_fraction,
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


class Evolutionary(NASBaseConfig):
    def __init__(
        self,
        supernet,
        optimization_metrics,
        measurements,
        num_evals,
        results_path,
        dataset_path: str = None,
        seed=42,
        population=50,
        batch_size: int = 128,
        eval_batch_size: int = 128,
        verbose=False,
        search_algo='nsga2',
        supernet_ckpt_path=None,
        test_fraction: float = 1.0,
        mp_calibration_samples: int = 100,
        dataloader_workers: int = 4,
        device: str = 'cpu',
        **kwargs,
    ):
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
            mp_calibration_samples=mp_calibration_samples,
            dataloader_workers=dataloader_workers,
            **kwargs,
        )

    def search(self):
        self._init_search()

        # Following sets up the algorithm based on number of objectives
        # Could be refractored at the expense of readability
        if self.num_objectives == 1:
            problem = EvolutionarySingleObjective(
                evaluation_interface=self.validation_interface,
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
                search_manager.configure_cmaes(num_evals=self.num_evals)
            else:
                search_manager = EvolutionaryManager(
                    algorithm='ga',
                    seed=self.seed,
                    n_obj=self.num_objectives,
                    verbose=self.verbose,
                )
                search_manager.configure_ga(population=self.population, num_evals=self.num_evals)
        elif self.num_objectives == 2:
            problem = EvolutionaryMultiObjective(
                evaluation_interface=self.validation_interface,
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
                search_manager.configure_age(population=self.population, num_evals=self.num_evals)
            else:
                search_manager = EvolutionaryManager(
                    algorithm='nsga2',
                    seed=self.seed,
                    n_obj=self.num_objectives,
                    verbose=self.verbose,
                )
                search_manager.configure_nsga2(population=self.population, num_evals=self.num_evals)
        elif self.num_objectives == 3:
            problem = EvolutionaryManyObjective(
                evaluation_interface=self.validation_interface,
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
                search_manager.configure_ctaea(num_evals=self.num_evals)
            elif self.search_algo == 'moead':
                search_manager = EvolutionaryManager(
                    algorithm='moead',
                    seed=self.seed,
                    n_obj=self.num_objectives,
                    verbose=self.verbose,
                )
                search_manager.configure_moead(num_evals=self.num_evals)
            else:
                search_manager = EvolutionaryManager(
                    algorithm='unsga3',
                    seed=self.seed,
                    n_obj=self.num_objectives,
                    verbose=self.verbose,
                )
                search_manager.configure_unsga3(population=self.population, num_evals=self.num_evals)
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


class RandomSearch(NASBaseConfig):
    def __init__(
        self,
        supernet,
        optimization_metrics,
        measurements,
        num_evals,
        results_path,
        dataset_path: str = None,
        seed=42,
        population=50,
        batch_size: int = 128,
        eval_batch_size: int = 128,
        verbose=False,
        search_algo='nsga2',
        supernet_ckpt_path: str = None,
        device: str = 'cpu',
        test_fraction: float = 1.0,
        mp_calibration_samples: int = 100,
        dataloader_workers: int = 4,
        metric_eval_fns: dict = None,
        **kwargs,
    ):
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
            mp_calibration_samples=mp_calibration_samples,
            dataloader_workers=dataloader_workers,
            metric_eval_fns=metric_eval_fns,
            **kwargs,
        )

    def search(self):
        self._init_search()

        # Randomly sample search space for initial population
        latest_population = [self.supernet_manager.random_sample() for _ in range(self.population)]

        # High-Fidelity Validation measurements
        for _, individual in enumerate(latest_population):
            log.info(f'Evaluating subnetwork {_+1}/{self.population}')
            self.validation_interface.eval_subnet(individual)

        output = list()
        for individual in latest_population:
            param_individual = self.supernet_manager.translate2param(individual)
            if 'bootstrapnas' in self.supernet:
                param_individual = BootstrapNASEncoding.convert_subnet_config_to_bootstrapnas(param_individual)
            output.append(param_individual)

        return output


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
        batch_size: int = 128,
        eval_batch_size: int = 128,
        supernet_ckpt_path: str = None,
        device: str = 'cpu',
        test_fraction: float = 1.0,
        mp_calibration_samples: int = 100,
        dataloader_workers: int = 4,
        **kwargs,
    ):
        self.main_results_path = results_path
        LOCAL_RANK, WORLD_RANK, WORLD_SIZE, DIST_METHOD = get_distributed_vars()
        results_path = get_worker_results_path(results_path, WORLD_RANK)

        if device == 'cuda':
            device = f'cuda:{LOCAL_RANK}'

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
            mp_calibration_samples=mp_calibration_samples,
            dataloader_workers=dataloader_workers,
            **kwargs,
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

            # On every external loop start we have to clear worker's results. Results from all workers are later merged into a single
            # file and this will help to keep gathered results in order in which configurations were evaluated.
            self.validation_interface.format_csv(self.csv_header)

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
                # First read main CSV file (if exists), and the append all latest population results from workers.
                csv_paths = (
                    [self.main_results_path] + worker_results_paths
                    if os.path.exists(self.main_results_path)
                    else worker_results_paths
                )
                combined_csv = pd.concat([pd.read_csv(f) for f in csv_paths])
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
                elif self.supernet == 'vit_base_imagenet':
                    runner_predict = ViTRunner(
                        supernet=self.supernet,
                        latency_predictor=self.predictor_dict['latency'],
                        macs_predictor=self.predictor_dict['macs'],
                        params_predictor=self.predictor_dict['params'],
                        acc_predictor=self.predictor_dict['accuracy_top1'],
                        dataset_path=self.dataset_path,
                        checkpoint_path=self.supernet_ckpt_path,
                        batch_size=self.batch_size,
                        device=self.device,
                    )

                elif self.supernet == 'inc_quantization_ofa_resnet50':
                    runner_predict = QuantizedOFARunner(
                        supernet=self.supernet,
                        latency_predictor=self.predictor_dict['latency'],
                        model_size_predictor=self.predictor_dict['model_size'],
                        params_predictor=self.predictor_dict['params'],
                        acc_predictor=self.predictor_dict['accuracy_top1'],
                        dataset_path=self.dataset_path,
                        device=self.device,
                        dataloader_workers=self.dataloader_workers,
                        test_fraction=self.test_fraction,
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


class RandomSearchDistributed(RandomSearch):
    def __init__(
        self,
        supernet,
        optimization_metrics,
        measurements,
        num_evals,
        results_path,
        dataset_path: str = None,
        seed=42,
        population=50,
        batch_size: int = 128,
        eval_batch_size: int = 128,
        verbose=False,
        search_algo='nsga2',
        supernet_ckpt_path: str = None,
        device: str = 'cpu',
        test_fraction: float = 1.0,
        mp_calibration_samples: int = 100,
        dataloader_workers: int = 4,
        **kwargs,
    ):
        self.main_results_path = results_path
        LOCAL_RANK, WORLD_RANK, WORLD_SIZE, DIST_METHOD = get_distributed_vars()
        results_path = get_worker_results_path(results_path, WORLD_RANK)

        if device == 'cuda':
            device = f'cuda:{LOCAL_RANK}'

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
            mp_calibration_samples=mp_calibration_samples,
            dataloader_workers=dataloader_workers,
            **kwargs,
        )

    def search(self):
        self._init_search()

        LOCAL_RANK, WORLD_RANK, WORLD_SIZE, DIST_METHOD = get_distributed_vars()

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

        # High-Fidelity Validation measurements
        for _, individual in enumerate(latest_population):
            log.info(f'Evaluating subnetwork {_+1}/{len(latest_population)} [{self.population}]')
            self.validation_interface.eval_subnet(individual)

        output = list()
        for individual in latest_population:
            output.append(self.supernet_manager.translate2param(individual))

        data = {
            'output': output,
            'from': WORLD_RANK,
            'results_path': self.results_path,
        }

        outputs = [None for _ in range(WORLD_SIZE)]
        dist.all_gather_object(outputs, data)

        if is_main_process():
            output = [o['output'] for o in outputs]

            worker_results_paths = [o['results_path'] for o in outputs]
            combined_csv = pd.concat([pd.read_csv(f) for f in worker_results_paths])
            combined_csv.to_csv(self.main_results_path, index=False)
            log.info(f'Saving combined results to {self.main_results_path}')
        return output
