"""
DyNAS-T Reference Search

SuperNetwork: HAT Transformer-Base

Search Tactic: Full Search (predictor training then search)

Description: Multi-Objective genetic algorithm search.
Assumes that the user has:
    * a latency predictor to predict the network, and the metadata
      with information on the hardware config
    * an AccuracyPredictor neural network that predicts the acc
The results can be saved to the `--csv_path` file.
"""
# Imports
import argparse
import csv
import json
from datetime import datetime
import numpy as np
import pandas as pd
import random
import copy
import pickle
import json
# DyNAS-T Specific Imports
from dynast.manager import ParameterManager
from dynast.evaluation_module.predictor import NCFHitRatePredictor, NCFLatencyPredictor
from dynast.search_module.search import SearchAlgoManager, ProblemMultiObjective
from dynast.analytics_module.results import ResultsManager


class NCFRunner:
    '''
    The NCFRunner is responsible for 'running' the subnetwork evaluation.
    '''
    def __init__(self, supernet, model_dir, lat_predictor, acc_predictor,max_layers=6,unique_value_path=None):

        self.supernet = supernet
        self.model_dir = model_dir
        self.acc_predictor = acc_predictor
        self.latency_predictor = lat_predictor
        self.onehot_unique = unique_value_path
        self.max_layers = max_layers

    def estimate_accuracy_hitrate(self, subnet_cfg):
        # Accuracy Predictor
        hr = self.acc_predictor.predict_single(subnet_cfg)
        return hr

    def estimate_latency(self, subnet_cfg):
        # Latency Predictor
        latency = self.latency_predictor.predict_single(subnet_cfg)
        return latency

    def convert_onehot(self,param_dict,unique_values):
        config = param_dict
        max_layers=self.max_layers

        features = []
        features.extend([config['num_factors_gmf'][0], config['num_factors_mlp_users'][0],
                         config['num_factors_mlp_items'][0],config['num_layers'][0]])
        hidden_list =  config['hidden_sizes'][:config['num_layers'][0]] + [0]* (max_layers-config['num_layers'][0])

        example = features + hidden_list
        one_hot_count = 0
        for value in unique_values:
            one_hot_count = one_hot_count + len(value)

        one_hot_examples = np.zeros((1, one_hot_count))
        offset = 0
        for f in range(len(example)):
            index = np.where(unique_values[f] == example[f])[0] + offset
            one_hot_examples[0, index] = 1.0
            offset += len(unique_values[f])
        return one_hot_examples

class UserEvaluationInterface:
    '''
    The interface class update is required to be updated for each unique SuperNetwork
    framework as it controls how evaluation calls are made from DyNAS-T

    Parameters
    ----------
    evaluator : class
        The 'runner' that performs the validation or prediction
    manager : class
        The DyNAS-T manager that translates between PyMoo and the parameter dict
    csv_path : string
        (Optional) The csv file that get written to during the subnetwork search
    '''

    def __init__(self, evaluator, manager,unique_values, csv_path=None):
        self.evaluator = evaluator
        self.manager = manager
        self.csv_path = csv_path
        self.unique_values = unique_values

    def eval_subnet(self, x):
        # PyMoo vector to Elastic Parameter Mapping
        param_dict = self.manager.translate2param(x)

        one_hot_feature = self.evaluator.convert_onehot(param_dict,self.unique_values)

        sample = {
            'num_factors_gmf': param_dict["num_factors_gmf"],
            "num_factors_mlp_users": param_dict["num_factors_mlp_users"],
            "num_factors_mlp_items":  param_dict["num_factors_mlp_items"],
           "num_layers":  param_dict["num_layers"]
            }
        # Bug Fix - deep copy prevents accidental re-mapping of sample
        subnet_sample = copy.deepcopy(sample)

        latency = self.evaluator.estimate_latency(one_hot_feature)
        hit_rate = self.evaluator.estimate_accuracy_hitrate(one_hot_feature)

        if self.csv_path:
            with open(self.csv_path, 'a') as f:
                writer = csv.writer(f)
                date = str(datetime.now())
                result = [subnet_sample, date, latency, hit_rate]
                writer.writerow(result)

        # PyMoo only minimizes objectives, thus accuracy needs to be negative
        # Requires format: subnetwork, objective x, objective y
        return sample, latency, -hit_rate

def convert_dataset(dataset_csv,dataset_type='accuracy',max_layers=6):
    with open(dataset_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        config_list = []
        acc_list = []
        ignored_header = False
        with open('temp_'+dataset_type+'.csv','w') as f:
            writer = csv.writer(f)
            if dataset_type =='accuracy':
                writer.writerow(['config','hit_rate'])
            else:
                writer.writerow(['config','latency'])
            for row in reader:
                 if not ignored_header:
                    ignored_header = True
                    continue

                 config_dict = {}
                 acc_dict = {}

                 config_dict["num_factors_gmf"] = [int(row[0])]
                 config_dict["num_factors_mlp_users"] = [int(row[1])]
                 config_dict["num_factors_mlp_items"] = [int(row[2])]
                 config_dict["num_layers"] = [int(row[3])]

                 hidden_sizes_list = []
                 i = 4
                 for _ in range(max_layers):
                      hidden_sizes_list.append(int(row[i]))
                      i += 1

                 config_dict["hidden_sizes"] = hidden_sizes_list
                 if dataset_type == 'accuracy':
                    hit_rate = float(row[i])
                    ndcg = float(row[i+1])
                 else:
                    latency = float(row[i])

                 config_list.append(config_dict)
                 if dataset_type =='accuracy':
                    writer.writerow([json.dumps(config_dict),hit_rate])
                 else:
                    writer.writerow([json.dumps(config_dict),latency])

    return 'temp_'+dataset_type+'.csv'


def to_one_hot(examples):

    one_hot_count = 0
    unique_values = []
    for c in range(examples.shape[1]):
        unique_values.append(np.unique(examples[:, c]))
        one_hot_count += len(unique_values[-1])

    one_hot_examples = np.zeros((examples.shape[0], one_hot_count))
    for e, example in enumerate(examples):
        offset = 0
        for f in range(len(example)):
            index = np.where(unique_values[f] == example[f])[0] + offset
            one_hot_examples[e, index] = 1.0
            offset += len(unique_values[f])
    return one_hot_examples,unique_values

def extract_data(filename,type):
        reader = csv.reader(open(filename), delimiter=',')
        rows = []
        for row in reader:
          rows.append(row)

         # Extract examples and labels
        data = np.array(rows[1:]).astype('float32')
        if type=='latency':
            examples, labels = data[:, 0:-1], data[:, -1]
        else:
            examples, labels = data[:, 0:-2], data[:, -2]

        return examples,labels


def main(args):

    # --------------------------------
    # NCF <-> DyNAS-T Interface Setup
    # --------------------------------

    # Define SuperNetwork Parameter Dictionary and Instantiate Manager
    supernet_parameters = {"num_factors_gmf" : {'count':1, 'vars': [8, 16,32,64, 128]},
                           "num_factors_mlp_users" : {'count':1, 'vars': [8, 16,32,64, 128]},
                           "num_factors_mlp_items": {'count':1, 'vars': [8, 16,32,64, 128]},
                           "num_layers" : {'count':1, 'vars': [1,2,3,4,5,6]},
                           "hidden_sizes" : {'count':6, 'vars':[8, 16, 32, 64, 128, 256, 512, 1024]} ,
                           }
    supernet_manager = ParameterManager(param_dict=supernet_parameters,
                                        seed=args.seed)
    supernet = args.supernet
    max_layers = max(supernet_parameters["num_layers"]["vars"])

    accuracy_file = convert_dataset(args.accuracy_csv,'accuracy',max_layers)
    latency_file = convert_dataset(args.latency_csv,'latency',max_layers)

    if args.accpred_path:
        print('[Info] Loading pre-trained accuracy predictor.')
        acc_pred = NCFHitRatePredictor()
        acc_pred.load(args.accpred_path)
    else:
        print('[Info] Building Accuracy Predictor')
        examples, labels = extract_data(args.accuracy_csv,'accuracy')
        one_hot, unique_values = to_one_hot(examples)
        acc_pred = NCFHitRatePredictor()
        acc_pred.train(one_hot, labels)

    if args.latpred_path:
       print('[Info] Loading pre-trained latency predictor.')
       lat_pred = NCFLatencyPredictor()
       lat_pred.load(args.latpred_path)
    else:
       print('[Info] Building Latency Predictor')

       examples,labels = extract_data(args.latency_csv,'latency')
       one_hot, unique_values = to_one_hot(examples)

       lat_pred = NCFLatencyPredictor()
       lat_pred.train(one_hot, labels)

    # Instatiate objective 'runner'
    runner = NCFRunner(supernet=supernet, model_dir=args.model_dir,lat_predictor=lat_pred,
                       acc_predictor=acc_pred, max_layers=max_layers)

    # --------------------------------
    # DyNAS-T Search Components
    # --------------------------------

    # Define how evaluations occur, gives option for csv file
    evaluation_interface = UserEvaluationInterface(evaluator=runner,
                                                   manager=supernet_manager,
                                                   unique_values = unique_values,
                                                   csv_path=args.csv_path)

    # Instantiate Multi-Objective Problem Class
    problem = ProblemMultiObjective(evaluation_interface=evaluation_interface,
                                    param_count=supernet_manager.param_count,
                                    param_upperbound=supernet_manager.param_upperbound)

    # Instantiate Search Manager
    search_manager = SearchAlgoManager(algorithm=args.algorithm,
                                       seed=args.seed)
    search_manager.configure_nsga2(population=args.population,
                                   num_evals=args.num_evals)

    # Run the search!
    output = search_manager.run_search(problem)

    # Process results
    results = ResultsManager(csv_path=args.csv_path,
                             manager=supernet_manager,
                             search_output=output)
    results.history_to_csv()
    results.front_to_csv()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='/store/.torch/ncf_nets')
    parser.add_argument('--supernet', default='NCF')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--latpred_path', help='path to latency predictor')
    parser.add_argument('--accpred_path', help='path to pre-trained acc predictor')
    parser.add_argument('--latency_csv', help='path to latency predictor')
    parser.add_argument('--accuracy_csv', help='path to pre-trained acc predictor')
    parser.add_argument('--unique_value_path', help='path to unique one-hot values')

    parser.add_argument('--input_csv', help='path to pre-trained acc predictor')

    # DyNAS-T Arguments
    parser.add_argument('--algorithm', default='nsga2', choices=['nsga2', 'rnsga2'],
                        help='Search algorithm, currently supports nsga2 and rnsga2')
    parser.add_argument('--num_evals', default=10000, type=int, help='number of TOTAL acc_pred, LUT samples')
    parser.add_argument('--csv_path', required=True, default=None, help='location to save results.')
    parser.add_argument('--population', default=50, type=int, help='population size for each generation')
    parser.add_argument('--verbose', action='store_true', help='Flag to control output')
    parser.add_argument('--search_tactic', default='nsga2', choices=['nsga2', 'rnsga2'],
                        help='Search tactic (e.g., full search, warm-start, concurrent.')
    args = parser.parse_args()

    print('\n'+'-'*40)
    print('DyNAS-T Multi-Objective Search Starting')
    print('-'*40)

    main(args)
