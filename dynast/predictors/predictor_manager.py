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
