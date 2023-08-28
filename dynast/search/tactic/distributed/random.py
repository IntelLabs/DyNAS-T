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

from dynast.search.tactic.random import RandomSearch
from dynast.supernetwork.supernetwork_registry import *
from dynast.utils import log, split_list
from dynast.utils.distributed import get_distributed_vars, get_worker_results_path, is_main_process


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
            eval_batch_size=eval_batch_size,
            verbose=verbose,
            search_algo=search_algo,
            supernet_ckpt_path=supernet_ckpt_path,
            test_fraction=test_fraction,
            dataloader_workers=dataloader_workers,
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
