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


import copy
import csv
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
from addict import Dict
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import SubnetConfig
from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import resume_compression_from_state
from nncf.torch.checkpoint_loading import load_state
from nncf.torch.model_creation import create_nncf_network

from dynast.measure.latency import auto_steps
from dynast.predictors.dynamic_predictor import Predictor
from dynast.search.evaluation_interface import EvaluationInterface
from dynast.utils import log
from dynast.utils.datasets import CIFAR10
from dynast.utils.nn import get_macs, measure_latency, reset_bn, validate_classification


class BootstrapNAS:
    def __init__(self, model: torch.nn.Module, nncf_config: Dict):
        nncf_network = create_nncf_network(model, nncf_config)

        compression_state = torch.load(nncf_config.supernet_path, map_location=torch.device(nncf_config.device))
        self._model, self._elasticity_ctrl = resume_compression_from_state(nncf_network, compression_state)
        model_weights = torch.load(nncf_config.supernet_weights, map_location=torch.device(nncf_config.device))

        load_state(model, model_weights, is_resume=True)

    def get_search_space(self):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        active_handlers = {
            dim: m_handler._handlers[dim] for dim in m_handler._handlers if m_handler._is_handler_enabled_map[dim]
        }
        space = {}
        for handler_id, handler in active_handlers.items():
            space[handler_id.value] = handler.get_search_space()
        return space

    def eval_subnet(self, config, eval_fn, **kwargs):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        m_handler.activate_subnet_for_config(
            m_handler.get_config_from_pymoo(config)
            # config
        )
        print(kwargs)
        return eval_fn(self._model, **kwargs)

    def get_active_subnet(self):
        return self._model

    def get_active_config(self):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        return m_handler.get_active_config()

    def get_random_config(self):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        return m_handler.get_random_config()

    def get_minimum_config(self):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        return m_handler.get_minimum_config()

    def get_maximum_config(self):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        return m_handler.get_maximum_config()

    def get_available_elasticity_dims(self):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        return m_handler.get_available_elasticity_dims()

    def activate_subnet_for_config(self, config):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        m_handler.activate_subnet_for_config(config)

    def get_config_from_pymoo(self, x: List) -> SubnetConfig:
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        return m_handler.get_config_from_pymoo(x)


