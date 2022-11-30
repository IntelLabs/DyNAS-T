# INTEL CONFIDENTIAL
# Copyright 2022 Intel Corporation. All rights reserved.

# This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
# express license under which they were provided to you ("License"). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without
# Intel's prior written permission.

# This software and the related documents are provided as is, with no express or implied warranties, other than those
# that are expressly stated in the License.

# This software is subject to the terms and conditions entered into between the parties.

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
