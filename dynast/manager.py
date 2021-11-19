import random
import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split


class ParameterManager:
    
    def __init__(self, param_dict, verbose=False, seed=0):
        self.param_dict = param_dict
        self.verbose = verbose
        self.mapper, self.param_upperbound, self.param_count = self.process_param_dict()
        self.inv_mapper = self.inv_mapper()
        self.set_seed(seed)
        
    def process_param_dict(self):
        '''
        Builds a parameter mapping arrays and an upper-bound vector for PyMoo.    
        '''
        parameter_count = 0
        parameter_bound = list()
        parameter_upperbound = list()
        parameter_mapper = list()

        for parameter, options in self.param_dict.items():
            # How many variables should be searched for
            parameter_count += options['count']
            parameter_bound.append(options['count'])

            # How many variables for each parameter
            for i in range(options['count']):
                parameter_upperbound.append(len(options['vars'])-1)
                index_simple = [x for x in range(len(options['vars']))]
                parameter_mapper.append(dict(zip(index_simple, options['vars'])))
        
        if self.verbose:
            print('[Info] Problem definition variables: {}'.format(parameter_count))
            print('[Info] Variable Upper Bound array: {}'.format(parameter_upperbound))
            print('[Info] Mapping dictionary created of length: {}'.format(len(parameter_mapper)))
            print('[Info] Parameter Bound: {}'.format(parameter_bound))

        return parameter_mapper, parameter_upperbound, parameter_count
    
    def inv_mapper(self):
        '''
        Builds inverse of self.mapper    
        '''
        inv_parameter_mapper = list()
        
        for value in self.mapper:
            inverse_dict = {v: k for k, v in value.items()}
            inv_parameter_mapper.append(inverse_dict)
        
        return inv_parameter_mapper
        
    def onehot_generic(self, in_array):
        '''
        This is a generic approach to one-hot vectorization for predictor training 
        and testing. It does not account for unused parameter mapping (e.g. block depth).
        For unused parameter mapping, the end user will need to provide a custom solution.

        input_array - the pymoo individual 1-D vector
        mapper - the map for elastic parameters of the supernetwork
        '''
        # Insure compatible array and mapper
        assert len(in_array) == len(self.mapper)

        onehot = list()

        # This function converts a pymoo input vector to a one-hot feature vector
        for i in range(len(self.mapper)):
            segment = [0 for _ in range(len(self.mapper[i]))]
            segment[in_array[i]] = 1
            onehot.extend(segment)

        return np.array(onehot)
    
    def random_sample(self):
        '''
        Generates a random subnetwork from the possible elastic parameter range
        '''
        pymoo_vector = list()
        for i in range(len(self.mapper)):
            options = [x for x in range(len(self.mapper[i]))]
            pymoo_vector.append(random.choice(options))

        return pymoo_vector
    
    def translate2param(self, pymoo_vector):
        '''
        Translate a PyMoo 1-D parameter vector back to the elastic parameter dictionary format
        '''
        output = dict()

        # Assign (and map) each vector element to the appropriate parameter dictionary key
        counter = 0
        for key, value in self.param_dict.items():
            output[key] = list()
            for i in range(value['count']):
                output[key].append(self.mapper[counter][pymoo_vector[counter]])
                counter += 1

        # Insure correct vector mapping occurred        
        assert counter == len(self.mapper)

        return output

    def translate2pymoo(self, parameters):
        '''
        Translate a single parameter dict to pymoo vector
        '''
        output = list()
        
        mapper_counter = 0
        for key, value in self.param_dict.items():
            param_counter = 0
            for i in range(value['count']):
                output.append(self.inv_mapper[mapper_counter][parameters[key][param_counter]])
                mapper_counter += 1
                param_counter += 1
        
        return output
    
    def import_csv(self, filepath, config, objective, column_names=None):
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
            df = pd.read_csv(filepath, names=column_names)
        df = df[[config, objective]]

        # Old corner case coverage
        df[config] = df[config].replace({'null': 'None'}, regex=True)

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
            # Onehot preditor format
            config_as_onehot = self.onehot_generic(config_as_pymoo)
            convert_to_onehot.append(config_as_onehot)

        df[config] = convert_to_dict
        df['config_pymoo'] = convert_to_pymoo
        df['config_onehot'] = convert_to_onehot

        return df

    def set_seed(self, seed):
        '''
        Set the random seed for randomized subnet generation and test/train split
        '''
        self.seed = seed
        random.seed(seed)        

    @staticmethod
    def create_training_set(dataframe, train_only=True, split=0.33):
        '''
        Create a sklearn compatible test/train set from an imported results csv
        after "import_csv" method is run.        
        '''
        collect_rows = list()
        for i in range(len(dataframe)):
            collect_rows.append(np.asarray(dataframe['config_onehot'].iloc[i]))
        features = np.asarray(collect_rows)   

        labels = dataframe.drop(columns=['config', 'config_pymoo', 'config_onehot']).values

        assert len(features) == len(labels)

        if train_only:
            print('[Info] Training set length={}'.format(len(labels)))
            return features, labels
        else:
            features_train, features_test, labels_train, labels_test \
                = train_test_split(features, labels, test_size=split, random_state=self.seed)
            print('[Info] Test ({}) Train ({}) ratio is {}.'.format(len(labels_train), len(labels_test), splits))
            return features_train, features_test, labels_train, labels_test

    

            