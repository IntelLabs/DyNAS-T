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


import ast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dynast.supernetwork.text_classification.bert_encoding import BertSST2Encoding
from dynast.utils import log


class ViTEncoding(BertSST2Encoding):
    def __init__(self, param_dict: dict, verbose: bool = False, seed: int = 0):
        super().__init__(param_dict, verbose, seed)

    def onehot_custom(self, subnet_cfg, provide_onehot=True, max_layers=12):
        features = []
        num_layers = subnet_cfg['num_layers'][0]

        attn_head_list = subnet_cfg['num_attention_heads'][:num_layers] + [0] * (max_layers - num_layers)
        intermediate_size_list = subnet_cfg['vit_intermediate_sizes'][:num_layers] + [0] * (max_layers - num_layers)
        features = [num_layers] + attn_head_list + intermediate_size_list

        if provide_onehot == True:
            examples = np.array([features])
            one_hot_count = 0
            # TODO(macsz,sharathns93) `self.unique_values` might be uninitialized
            # if `create_training_set` hasn't been called prior to this
            unique_values = self.unique_values

            for unique in unique_values:
                one_hot_count += len(unique.tolist())

            one_hot_examples = np.zeros((examples.shape[0], one_hot_count))
            for e, example in enumerate(examples):
                offset = 0
                for f in range(len(example)):
                    index = np.where(unique_values[f] == example[f])[0] + offset
                    one_hot_examples[e, index] = 1.0
                    offset += len(unique_values[f])
            return one_hot_examples

        else:
            return features

    def import_csv(self, filepath, config, objective, column_names=None, drop_duplicates=True):
        '''
        Import a csv file generated from a supernetwork search for the purpose
        of training a predictor.

        filepath - path of the csv to be imported.
        config - the subnetwork configuration
        objective - target/label for the subnet configuration (e.g. accuracy, latency)
        column_names - a list of column names for the dataframe
        df - the output dataframe that contains the original config dict, pymoo, and 1-hot
             equivalent vector for training.
        '''

        max_layers = 12  # 16 # TODO: Needs to be moved as an argument
        if column_names == None:
            df = pd.read_csv(filepath)
        else:
            df = pd.read_csv(filepath)
            df.columns = column_names
        df = df[[config, objective]]
        # Old corner case coverage
        df[config] = df[config].replace({'null': 'None'}, regex=True)

        if drop_duplicates:
            df.drop_duplicates(subset=[config], inplace=True)
            df.reset_index(drop=True, inplace=True)

        convert_to_dict = list()
        convert_to_pymoo = list()
        convert_to_onehot = list()

        for i in range(len(df)):
            # Elastic Param Config format
            config_as_dict = ast.literal_eval(df[config].iloc[i])
            convert_to_dict.append(config_as_dict)
            # PyMoo 1-D vector format
            config_as_pymoo = self.translate2pymoo(config_as_dict)
            convert_to_pymoo.append(config_as_pymoo)
            # Onehot predictor format
            config_as_onehot = self.onehot_custom(config_as_dict, provide_onehot=False, max_layers=max_layers)
            convert_to_onehot.append(config_as_onehot)
        df[config] = convert_to_dict
        df['config_pymoo'] = convert_to_pymoo
        df['config_onehot'] = convert_to_onehot
        return df
