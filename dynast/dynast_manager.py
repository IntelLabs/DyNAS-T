# INTEL CONFIDENTIAL
# Copyright 2022 Intel Corporation. All rights reserved.

# This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
# express license under which they were provided to you ("License"). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without
# Intel's prior written permission.

# This software and the related documents are provided as is, with no express or implied warranties, other than those
# that are expressly stated in the License.

# This software is subject to the terms and conditions entered into between the parties.

import sys

from dynast.search.search_tactic import LINAS, Evolutionary, RandomSearch
from dynast.utils import log


class DyNAS:
    '''
    The DyNAS class manages the user input configuration parameters, the search tactics, and
    the supernetwork configuration. Since the search tactics vary widely, they are instantiate
    with this class based on the user input.
    '''

    def __new__(self, **kwargs):

        log.info('=' * 40)
        log.info('Starting Dynamic NAS Toolkit (DyNAS-T)')
        log.info('=' * 40)

        # Required arguments for the DyNAS class
        # TODO(macsz) If these are common for all created classes, then we can move it out of kwargs
        REQUIRED_KWARGS = [
            'supernet',
            'optimization_metrics',
            'measurements',
            'search_tactic',
            'num_evals',
            'results_path',
            'dataset_path',
        ]

        # Validity checks
        for argument in REQUIRED_KWARGS:
            if argument not in kwargs:
                log.error(f"Missing `--{argument}` parameter.")
                sys.exit("Missing argument, see log file for info.")

        if len(kwargs['optimization_metrics']) > 3:
            log.error('Number of optimization_metrics is out of range. 1-3 supported.')
            sys.exit()

        log.info('-' * 40)
        log.info('DyNAS Parameter Inputs:')
        for key, value in kwargs.items():
            log.info(f'{key}: {value}')
        log.info('-' * 40)

        # LINAS bi-level evolutionary algorithm search
        if kwargs['search_tactic'] == 'linas':
            log.info('Initializing DyNAS LINAS algorithm object.')
            return LINAS(**kwargs)

        # Standard evolutionary algorithm search
        elif kwargs['search_tactic'] == 'evolutionary':
            log.info('Initializing DyNAS evoluationary algorithm object.')
            return Evolutionary(**kwargs)

        # Uniform random sampling of the architectural space
        elif kwargs['search_tactic'] == 'random':
            log.info('Initializing DyNAS random search algorithm object.')
            return RandomSearch(**kwargs)

        else:
            log.error("Invalid `--search_tactic` parameter (options: 'linas', 'evolutionary', 'random').")
