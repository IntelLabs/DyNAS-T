# INTEL CONFIDENTIAL
# Copyright 2022 Intel Corporation. All rights reserved.

# This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
# express license under which they were provided to you ("License"). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without
# Intel's prior written permission.

# This software and the related documents are provided as is, with no express or implied warranties, other than those
# that are expressly stated in the License.

# This software is subject to the terms and conditions entered into between the parties.

from dynast.predictors.dynamic_predictor import Predictor
from dynast.utils import log


class PredictorManager:
    def __init__(
        self,
        results_path: str,
        supernet_manager,
        objective_name: str,
        column_names: list,
        config_name: str = 'subnet',
        predictor_type: str = 'dynamic',
    ):
        self.results_path = results_path
        self.supernet_manager = supernet_manager
        self.objective_name = objective_name
        self.config_name = config_name
        self.column_names = column_names
        self.predictor_type = predictor_type

    def train_predictor(self):
        if self.predictor_type == 'dynamic':
            log.info('Building dynamic predictor for {}'.format(self.objective_name))

            names = ['subnet', 'date'] + self.column_names

            df = self.supernet_manager.import_csv(
                self.results_path, config=self.config_name, objective=self.objective_name, column_names=names
            )
            features, labels = self.supernet_manager.create_training_set(df, config=self.config_name)
            predictor = Predictor()
            predictor.train(features, labels.ravel())
            return predictor
        elif self.predictor_type == 'mlp':
            log.error('MLP predictor not implemented yet.')
        else:
            log.error('Invalid Predictor Selected.')
