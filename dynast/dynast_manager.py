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

import logging
import sys
from typing import Dict, List

from dynast.search.search_tactic import LINAS, Evolutionary, LINASDistributed, RandomSearch, RandomSearchDistributed
from dynast.utils import check_kwargs_deprecated, log, set_logger
from dynast.utils.distributed import get_distributed_vars, init_distributed


class DyNAS:
    '''
    The DyNAS class manages the user input configuration parameters, the search tactics, and
    the supernetwork configuration. Since the search tactics vary widely, they are instantiate
    with this class based on the user input.
    '''

    def __new__(
        self,
        supernet: str,
        results_path: str,
        optimization_metrics: List[str],
        measurements: List[str],
        search_tactic: str = 'linas',
        num_evals: int = 250,
        dataset_path: str = None,
        **kwargs,
    ):
        kwargs.update(
            {
                'supernet': supernet,
                'results_path': results_path,
                'optimization_metrics': optimization_metrics,
                'measurements': measurements,
                'search_tactic': search_tactic,
                'num_evals': num_evals,
                'dataset_path': dataset_path,
            }
        )

        log_level = logging.INFO
        if kwargs.get('verbose'):
            log_level = logging.DEBUG
        set_logger(level=log_level)

        LOCAL_RANK, WORLD_RANK, WORLD_SIZE, DIST_METHOD = get_distributed_vars()
        if DIST_METHOD:
            backend = kwargs.get('backend', 'gloo')
            init_distributed(backend, WORLD_RANK, WORLD_SIZE)
            seed = kwargs.get('seed', None)
            if seed:
                kwargs['seed'] = seed + WORLD_RANK

        DyNAS._set_eval_batch_size(kwargs)

        log.info('=' * 40)
        log.info('Starting Dynamic NAS Toolkit (DyNAS-T)')
        log.info('=' * 40)

        kwargs = check_kwargs_deprecated(**kwargs)

        if len(kwargs['optimization_metrics']) > 3:
            log.error('Number of optimization_metrics is out of range. 1-3 supported.')
            sys.exit()

        log.info('-' * 40)
        log.info('DyNAS Parameter Inputs:')
        for key, value in kwargs.items():
            log.info(f'{key}: {value}')
        log.info('-' * 40)

        if kwargs.get('distributed', False):
            # LINAS bi-level evolutionary algorithm search distributed to multiple workers
            if search_tactic == 'linas':
                log.info('Initializing DyNAS LINAS (distributed) algorithm object.')
                return LINASDistributed(**kwargs)

                # Uniform random sampling of the architectural space distributed to multiple workers
            elif search_tactic == 'random':
                log.info('Initializing DyNAS random (distributed) search algorithm object.')
                return RandomSearchDistributed(**kwargs)

        # LINAS bi-level evolutionary algorithm search
        if search_tactic == 'linas':
            log.info('Initializing DyNAS LINAS algorithm object.')
            return LINAS(**kwargs)

        # Standard evolutionary algorithm search
        elif search_tactic == 'evolutionary':
            log.info('Initializing DyNAS evoluationary algorithm object.')
            return Evolutionary(**kwargs)

        # Uniform random sampling of the architectural space
        elif search_tactic == 'random':
            log.info('Initializing DyNAS random search algorithm object.')
            return RandomSearch(**kwargs)

        else:
            error_message = (
                "Invalid `--search_tactic` parameter `{}` (options: 'linas', 'evolutionary', 'random').".format(
                    search_tactic
                )
            )  # TODO(macsz) Un-hardcode options.
            log.error(error_message)
            raise NotImplementedError(error_message)

    @staticmethod
    def _set_eval_batch_size(kwargs: Dict) -> None:
        """Set eval_batch_size to batch_size if not specified."""
        if not kwargs.get('eval_batch_size'):
            kwargs['eval_batch_size'] = kwargs.get('batch_size', 128)
