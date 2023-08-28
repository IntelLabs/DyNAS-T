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

from dynast.search.tactic.base import NASBaseConfig
from dynast.supernetwork.image_classification.bootstrapnas.bootstrapnas_encoding import BootstrapNASEncoding
from dynast.supernetwork.supernetwork_registry import *
from dynast.utils import log


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
        batch_size=1,
        verbose=False,
        search_algo='nsga2',
        supernet_ckpt_path: str = None,
        device: str = 'cpu',
        test_fraction: float = 1.0,
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
            verbose=verbose,
            search_algo=search_algo,
            supernet_ckpt_path=supernet_ckpt_path,
            device=device,
            test_fraction=test_fraction,
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
