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

from dynast.supernetwork.image_classification.bootstrapnas.bootstrapnas_encoding import BootstrapNASEncoding
from dynast.supernetwork.image_classification.bootstrapnas.bootstrapnas_interface import BootstrapNASRunner
from dynast.supernetwork.image_classification.ofa.ofa_interface import OFARunner
from dynast.supernetwork.machine_translation.transformer_interface import TransformerLTRunner
from dynast.supernetwork.supernetwork_registry import *
from dynast.supernetwork.text_classification.bert_interface import BertSST2Runner
from dynast.utils import log


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
        self.num_objectives = len(self.optimization_metrics)  # TODO(macsz) Can be a getter
        self.measurements = list(set(self.measurements))

        # Check that measurements counts are correct
        if self.num_objectives > 3 or self.num_objectives < 1:
            log.error('Incorrect number of optimization objectives specified. Must be 1, 2, or 3.')

        # Verify that supernetwork and metrics are valid
        if self.supernet in SUPERNET_TYPE['image_classification']:
            valid_metrics = ['accuracy_top1', 'macs', 'latency', 'params']
            for metric in self.optimization_metrics:
                if metric not in valid_metrics:
                    log.error(f'Invalid metric(s) specified: {metric}. Choose from {valid_metrics}')
                elif metric in valid_metrics and metric not in self.measurements:
                    self.measurements.append(metric)

            for metric in self.measurements:
                if metric not in valid_metrics:
                    self.measurements.remove(metric)

        elif self.supernet in SUPERNET_TYPE['machine_translation']:
            # TODO(macsz,sharathns93) Fix
            pass

        elif self.supernet in SUPERNET_TYPE['text_classification']:
            # TODO(macsz,sharathns93) Fix
            pass

        elif self.supernet in SUPERNET_TYPE['recommendation']:
            pass

        else:
            log.error(f'Invalid supernet specified. Choose from the following: {SUPERNET_TYPE}')

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
