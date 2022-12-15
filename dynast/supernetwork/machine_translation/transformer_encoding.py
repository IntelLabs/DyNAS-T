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

from dynast.search.encoding import EncodingBase
from dynast.utils import log


class TransformerLTEncoding(EncodingBase):
    def __init__(self, param_dict: dict, verbose: bool = False, seed: int = 0):
        super().__init__(param_dict, verbose, seed)

    def onehot_custom(self, subnet_cfg, provide_onehot=True, encode_layer_num_int=6):

        features = []
        features.extend(subnet_cfg['encoder_embed_dim'])

        # Encoder FFN Embed Dim
        encoder_ffn_embed_dim = subnet_cfg['encoder_ffn_embed_dim']

        if encode_layer_num_int < 6:
            encoder_ffn_embed_dim.extend([0] * (6 - encode_layer_num_int))
        features.extend(encoder_ffn_embed_dim)

        # Encoder Self-Attn Heads

        encoder_self_attention_heads = subnet_cfg['encoder_self_attention_heads'][:encode_layer_num_int]

        if encode_layer_num_int < 6:
            encoder_self_attention_heads.extend([0] * (6 - encode_layer_num_int))
        features.extend(encoder_self_attention_heads)

        features.extend(subnet_cfg['decoder_embed_dim'])

        decoder_layer_num = subnet_cfg['decoder_layer_num']
        decoder_layer_num_int = decoder_layer_num[0]
        features.extend(decoder_layer_num)

        # Decoder FFN Embed Dim
        decoder_ffn_embed_dim = subnet_cfg['decoder_ffn_embed_dim'][:decoder_layer_num_int]

        if decoder_layer_num_int < 6:
            decoder_ffn_embed_dim.extend([0] * (6 - decoder_layer_num_int))
        features.extend(decoder_ffn_embed_dim)

        # Decoder Attn Heads
        decoder_self_attention_heads = subnet_cfg['decoder_self_attention_heads'][:decoder_layer_num_int]

        if decoder_layer_num_int < 6:
            decoder_self_attention_heads.extend([0] * (6 - decoder_layer_num_int))
        features.extend(decoder_self_attention_heads)

        # Decoder ENDE HEADS

        decoder_ende_attention_heads = subnet_cfg['decoder_ende_attention_heads'][:decoder_layer_num_int]

        if decoder_layer_num_int < 6:
            decoder_ende_attention_heads.extend([0] * (6 - decoder_layer_num_int))

        features.extend(decoder_ende_attention_heads)

        arbitrary_ende_attn_trans = []
        for i in range(decoder_layer_num_int):
            if subnet_cfg['decoder_arbitrary_ende_attn'][i] == -1:
                arbitrary_ende_attn_trans.append(1)
            elif subnet_cfg['decoder_arbitrary_ende_attn'][i] == 1:
                arbitrary_ende_attn_trans.append(2)
            elif subnet_cfg['decoder_arbitrary_ende_attn'][i] == 2:
                arbitrary_ende_attn_trans.append(3)

        if decoder_layer_num_int < 6:
            arbitrary_ende_attn_trans.extend([0] * (6 - decoder_layer_num_int))
        features.extend(arbitrary_ende_attn_trans)

        if provide_onehot == True:
            examples = np.array([features])
            one_hot_count = 0
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
            config_as_onehot = self.onehot_custom(config_as_dict, provide_onehot=False)
            convert_to_onehot.append(config_as_onehot)
        df[config] = convert_to_dict
        df['config_pymoo'] = convert_to_pymoo
        df['config_onehot'] = convert_to_onehot

        return df

    # @staticmethod
    def create_training_set(self, dataframe, config='subnet', train_with_all=True, split=0.33, seed=None):
        '''
        Create a sklearn compatible test/train set from an imported results csv
        after "import_csv" method is run.
        '''

        collect_rows = list()
        for i in range(len(dataframe)):
            collect_rows.append(np.asarray(dataframe['config_onehot'].iloc[i]))
        features = np.asarray(collect_rows)
        labels = dataframe.drop(columns=[config, 'config_pymoo', 'config_onehot']).values

        assert len(features) == len(labels)
        one_hot_count = 0
        unique_values = []

        for c in range(features.shape[1]):
            unique_values.append(np.unique(features[:, c]))
            one_hot_count += len(unique_values[-1])
        one_hot_examples = np.zeros((features.shape[0], one_hot_count))
        for e, example in enumerate(features):
            offset = 0
            for f in range(len(example)):
                index = np.where(unique_values[f] == example[f])[0] + offset
                one_hot_examples[e, index] = 1.0
                offset += len(unique_values[f])

        features = one_hot_examples
        self.unique_values = unique_values
        if train_with_all:
            log.info('Training set length={}'.format(len(labels)))
            return features, labels
        else:
            features_train, features_test, labels_train, labels_test = train_test_split(
                features, labels, test_size=split, random_state=seed
            )
            log.info('Test ({}) Train ({}) ratio is {}.'.format(len(labels_train), len(labels_test), split))
            return features_train, features_test, labels_train, labels_test
