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

import csv
import random
import string

from dynast.utils import LazyImport, log
from dynast.utils.distributed import get_distributed_vars

set_workspace = LazyImport("neural_compressor.set_workspace")


class EvaluationInterface:
    """
    The interface class update is required to be updated for each unique SuperNetwork
    framework as it controls how evaluation calls are made from DyNAS-T

    Args:
        evaluator : class
            The 'runner' that performs the validation or prediction
        manager : class
            The DyNAS-T manager that translates between PyMoo and the parameter dict
        csv_path : string
            (Optional) The csv file that get written to during the subnetwork search
    """

    def __init__(
        self,
        evaluator,
        manager,
        optimization_metrics,
        measurements,
        csv_path,
        predictor_mode,
        mixed_precision: bool = False,
    ):
        self.evaluator = evaluator
        self.manager = manager
        self.optimization_metrics = optimization_metrics
        self.measurements = measurements
        self.predictor_mode = predictor_mode
        self.csv_path = csv_path
        self.mixed_precision = mixed_precision

    def format_csv(self, csv_header):
        if self.csv_path:
            f = open(self.csv_path, "w")
            writer = csv.writer(f)
            result = csv_header
            writer.writerow(result)
            f.close()
        log.info(f'(Re)Formatted results file: {self.csv_path}')
        log.info(f'csv file header: {csv_header}')

    def _set_workspace(self):
        LOCAL_RANK, WORLD_RANK, WORLD_SIZE, DIST_METHOD = get_distributed_vars()
        WORLD_RANK = WORLD_RANK if WORLD_RANK is not None else 0
        workspace_name = f"/tmp/dynast_nc_workspace_{WORLD_RANK}_{''.join(random.choices(string.ascii_letters, k=6))}"
        set_workspace(workspace_name)
        log.debug(f'Setting Neural Compressor workspace: {workspace_name}')
