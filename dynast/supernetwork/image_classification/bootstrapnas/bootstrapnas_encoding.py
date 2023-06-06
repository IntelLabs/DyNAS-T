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


import json
from collections import OrderedDict
from typing import List, Union

import numpy as np

from dynast.search.encoding import EncodingBase
from dynast.utils import LazyImport, log

nncf = LazyImport("nncf")


class BootstrapNASEncoding(EncodingBase):
    def __init__(self, param_dict: dict, verbose: bool = False, seed: int = 0):
        param_dict = BootstrapNASEncoding._bnas_to_dynast(param_dict)

        super().__init__(param_dict, verbose, seed)

    @staticmethod
    def _bnas_to_dynast(bootstrapnas_supernet_parameters: dict) -> dict:
        """Translate the BootstrapNAS parameter dictionary to DyNAS-T
        paramenter dictionary format.
        Args:
        -----
        * `bootstrapnas_supernet_parameters`: BootstrapNAS parameter dictionary.
        Returns:
        --------
        * `supernet_parameters`: DyNAS-T format of supernet parameters.
        """
        supernet_parameters = dict()

        for key, value in bootstrapnas_supernet_parameters.items():
            if key == "width":
                assert type(value) == dict
                for i, arr in enumerate(value):
                    supernet_parameters[f"width_{i}"] = {
                        "count": 1,
                        "vars": value[i],
                    }
            elif key == "depth":
                assert type(value) == list
                supernet_parameters["depth"] = {"count": 1, "vars": value}
            # TODO:(daniel-codes) need kernel example
            else:
                log.error("Unknown key name in BNAS parameter dictionary.")
                raise KeyError("Unknown key name in BNAS parameter dictionary.")

        return supernet_parameters

    def translate2pymoo(self, parameters: dict) -> List[int]:
        '''Translate a single parameter dict to pymoo vector'''

        output = list()

        mapper_counter = 0
        for key, value in self.param_dict.items():
            param_counter = 0
            if len(value['vars']) >= 2:
                for i in range(value['count']):
                    output.append(self.inv_mapper[mapper_counter][parameters[key][param_counter]])
                    mapper_counter += 1
                    param_counter += 1
            else:
                mapper_counter += 1
        return output

    def process_param_dict(self) -> None:
        '''Builds a parameter mapping arrays and an upper-bound vector for PyMoo.'''
        parameter_count = 0
        parameter_bound = list()
        parameter_upperbound = list()
        parameter_mapper = list()

        for parameter, options in self.param_dict.items():
            # How many variables should be searched for
            if len(options['vars']) > 1:
                parameter_count += options['count']
                parameter_bound.append(options['count'])

            # How many variables for each parameter
            for i in range(options['count']):
                if len(options['vars']) > 1:
                    parameter_upperbound.append(len(options['vars']) - 1)
                single_mapping = dict()
                for idx, value in enumerate(options['vars']):
                    if type(value) == int or type(value) == float:
                        single_mapping[idx] = value
                    else:
                        single_mapping[idx] = str(value)

                parameter_mapper.append(single_mapping)

        if self.verbose:
            log.info('Problem definition variables: {}'.format(parameter_count))
            log.info('Variable Upper Bound array: {}'.format(parameter_upperbound))
            log.info('Mapping dictionary created of length: {}'.format(len(parameter_mapper)))
            log.info('Parameter Bound: {}'.format(parameter_bound))

        return parameter_mapper, parameter_upperbound, parameter_count

    def translate2param(self, pymoo_vector: List[int]) -> dict:
        '''Translate a PyMoo 1-D parameter vector back to the elastic parameter dictionary format'''
        output = dict()

        # Assign (and map) each vector element to the appropriate parameter dictionary key
        counter = 0
        pymoo_vector_counter = 0
        for key, value in self.param_dict.items():
            output[key] = list()
            if len(value['vars']) >= 2:
                for i in range(value['count']):
                    # Note: following round term was added to support CMA-ES
                    output[key].append(self.mapper[counter][round(pymoo_vector[pymoo_vector_counter])])
                    counter += 1
                    pymoo_vector_counter += 1
            else:
                output[key].append(self.mapper[counter][0])
                counter += 1

        # Insure correct vector mapping occurred
        assert counter == len(self.mapper)

        return output

    def reconstruct_pymoo_vector(self, pymoo_vector: List[int]) -> List[int]:
        if len(self.mapper) == len(pymoo_vector):
            return pymoo_vector

        full_pymoo_vector = []
        pymoo_vector_ind = 0

        for key, value in self.param_dict.items():
            for i in range(value['count']):
                if len(value['vars']) == 1:
                    full_pymoo_vector.append(0)
                else:
                    full_pymoo_vector.append(pymoo_vector[pymoo_vector_ind])
                    pymoo_vector_ind += 1
        return full_pymoo_vector

    def partial_pymoo_vector(self, pymoo_vector: List[int]) -> List[int]:
        if len(self.param_upperbound) == len(pymoo_vector):
            return pymoo_vector

        partial_pymoo_vector = []
        pymoo_vector_ind = 0

        for key, value in self.param_dict.items():
            for i in range(value['count']):
                if len(value['vars']) > 1:
                    partial_pymoo_vector.append(pymoo_vector[pymoo_vector_ind])
                pymoo_vector_ind += 1

        return partial_pymoo_vector

    def onehot_generic(self, in_array: List[int]) -> np.ndarray:
        '''This is a generic approach to one-hot vectorization for predictor training and testing.
        It does not account for unused parameter mapping (e.g. block depth).
        For unused parameter mapping, the end user will need to provide a custom solution.

        input_array - the pymoo individual 1-D vector
        mapper - the map for elastic parameters of the supernetwork
        '''
        # Insure compatible array and mapper
        in_array = self.partial_pymoo_vector(in_array)
        short_mapper = [d for d in self.mapper if len(d) > 1]
        assert len(in_array) == len(short_mapper)

        onehot = list()

        # This function converts a pymoo input vector to a one-hot feature vector
        for i in range(len(short_mapper)):
            segment = [0 for _ in range(len(short_mapper[i]))]
            segment[in_array[i]] = 1
            onehot.extend(segment)

        return np.array(onehot)

    @staticmethod
    def convert_subnet_config_to_bootstrapnas(subnet_config: Union[dict, str]) -> OrderedDict:
        ElasticityDim = nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim.ElasticityDim
        if isinstance(subnet_config, str):
            subnet_config = json.loads(subnet_config.replace('\'', '\"'))
        output = OrderedDict()
        for key, value in subnet_config.items():
            if 'width' in key:
                if output.get(ElasticityDim.WIDTH) is None:
                    output[ElasticityDim.WIDTH] = {}
                idx = len(output[ElasticityDim.WIDTH])
                output[ElasticityDim.WIDTH][idx] = value[0]
            elif 'depth' in key:
                output[ElasticityDim.DEPTH] = json.loads(value[0])
            else:
                raise Exception("Unknown key name in BNAS parameter dictionary.")
        return output
