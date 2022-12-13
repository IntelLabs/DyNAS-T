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

from dynast.utils import log


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

    def __init__(self, evaluator, manager, optimization_metrics, measurements, csv_path, predictor_mode):
        self.evaluator = evaluator
        self.manager = manager
        self.optimization_metrics = optimization_metrics
        self.measurements = measurements
        self.predictor_mode = predictor_mode
        self.csv_path = csv_path

    def format_csv(self, csv_header):
        if self.csv_path:
            f = open(self.csv_path, "w")
            writer = csv.writer(f)
            result = csv_header
            writer.writerow(result)
            f.close()
        log.info(f'(Re)Formatted results file: {self.csv_path}')
        log.info(f'csv file header: {csv_header}')
