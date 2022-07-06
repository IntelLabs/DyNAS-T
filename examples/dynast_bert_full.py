"""
DyNAS-T Reference Search

SuperNetwork: Elastic BERT-BASE

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
import copy
import csv
from datetime import datetime

import numpy as np

from dynast.analytics_module.results import ResultsManager
from dynast.evaluation_module.predictor import Predictor

# DyNAS-T Specific Imports
from dynast.manager import ParameterManager
from dynast.search_module.search import ProblemMultiObjective, SearchAlgoManager
from dynast.utils import log


class BertRunner:
    '''
    The BertRunner is responsible for 'running' the subnetwork evaluation.
    '''

    def __init__(
        self,
        supernet,
        model_dir,
        lat_predictor,
        acc_predictor,
        max_layers=6,
        unique_value_path=None,
    ):

        self.supernet = supernet
        self.model_dir = model_dir
        self.acc_predictor = acc_predictor
        self.latency_predictor = lat_predictor
        self.onehot_unique = unique_value_path
        self.max_layers = max_layers

    def estimate_accuracy_mlm(self, subnet_cfg):

        mlm = self.acc_predictor.predict(subnet_cfg)

        return mlm

    def estimate_latency(self, subnet_cfg):

        # Latency Predictor
        latency = self.latency_predictor.predict(subnet_cfg)
        return latency

    def convert_onehot(self, param_dict, unique_values):
        config = param_dict
        max_layers = self.max_layers

        features = []
        features.extend(
            [
                config['bert_embedding_sizes'][0],
                config['bert_hidden_sizes'][0],
                config['num_layers'][0],
            ]
        )
        attn_head_list = config['num_attention_heads'][: config['num_layers'][0]] + [
            0
        ] * (max_layers - config['num_layers'][0])
        intermediate_size_list = config['bert_intermediate_sizes'][
            : config['num_layers'][0]
        ] + [0] * (max_layers - config['num_layers'][0])

        example = features + attn_head_list + intermediate_size_list
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

    def __init__(self, evaluator, manager, unique_values, csv_path=None):
        self.evaluator = evaluator
        self.manager = manager
        self.csv_path = csv_path
        self.unique_values = unique_values

    def eval_subnet(self, x):
        # PyMoo vector to Elastic Parameter Mapping
        param_dict = self.manager.translate2param(x)

        one_hot_feature = self.evaluator.convert_onehot(param_dict, self.unique_values)

        sample = {
            'bert_embedding_sizes': param_dict["bert_embedding_sizes"],
            "bert_hidden_sizes": param_dict["bert_hidden_sizes"],
            "num_layers": param_dict["num_layers"],
            "num_attention_heads": param_dict["num_attention_heads"],
            "bert_intermediate_sizes": param_dict["bert_intermediate_sizes"],
        }
        # Bug Fix - deep copy prevents accidental re-mapping of sample
        subnet_sample = copy.deepcopy(sample)

        latency = self.evaluator.estimate_latency(one_hot_feature)
        mlm_acc = self.evaluator.estimate_accuracy_mlm(one_hot_feature)

        if self.csv_path:
            with open(self.csv_path, 'a') as f:
                writer = csv.writer(f)
                date = str(datetime.now())
                result = [subnet_sample, date, latency, mlm_acc]
                writer.writerow(result)

        # PyMoo only minimizes objectives, thus accuracy needs to be negative
        # Requires format: subnetwork, objective x, objective y
        return sample, latency, -mlm_acc


def get_unique_vals(examples):
    one_hot_count = 0
    unique_values = []
    for c in range(examples.shape[1]):
        unique_values.append(np.unique(examples[:, c]))
        one_hot_count += len(unique_values[-1])
    return unique_values


def to_one_hot(examples, unique_values):

    one_hot_count = 0
    for value in unique_values:
        one_hot_count = one_hot_count + len(value)

    one_hot_examples = np.zeros((examples.shape[0], one_hot_count))
    for e, example in enumerate(examples):
        offset = 0
        for f in range(len(example)):
            index = np.where(unique_values[f] == example[f])[0] + offset
            one_hot_examples[e, index] = 1.0
            offset += len(unique_values[f])
    return one_hot_examples


def extract_data(filename, type):
    reader = csv.reader(open(filename), delimiter=',')
    rows = []
    for row in reader:
        rows.append(row)

    # Extract examples and labels
    data = np.array(rows[1:]).astype('float32')
    examples, labels = data[:, 0:-1], data[:, -1]

    return examples, labels


def main(args):

    # --------------------------------
    # OFA <-> DyNAS-T Interface Setup
    # --------------------------------

    # Define SuperNetwork Parameter Dictionary and Instantiate Manager

    supernet_parameters = {
        "bert_embedding_sizes": {'count': 1, 'vars': [512]},
        "bert_hidden_sizes": {'count': 1, 'vars': [768]},
        "num_layers": {'count': 1, 'vars': [4, 5, 6, 7, 8, 9, 10, 12]},
        "num_attention_heads": {'count': 12, 'vars': [2, 4, 8, 12]},
        "bert_intermediate_sizes": {'count': 12, 'vars': [512, 768, 1024, 3072]},
    }

    supernet_manager = ParameterManager(param_dict=supernet_parameters, seed=args.seed)
    # Objective 1 (Latency or MACs)
    supernet = args.supernet
    max_layers = max(supernet_parameters["num_layers"]["vars"])

    acc_examples, acc_labels = extract_data(args.accuracy_csv, 'accuracy')

    lat_examples, lat_labels = extract_data(args.latency_csv, 'latency')

    acc_unique = get_unique_vals(acc_examples)
    lat_unique = get_unique_vals(lat_examples)
    if len(acc_unique) == len(lat_unique):
        unique_values = acc_unique
    else:
        if len(acc_unique) > len(lat_unique):
            unique_values = acc_unique
        else:
            unique_values = lat_unique

    # Objective 2 (Accuracy)
    if args.accpred_path:
        log.info('Loading pre-trained accuracy predictor.')
        acc_pred = Predictor()
        acc_pred.load(args.accpred_path)
    else:
        log.info('Building Accuracy Predictor')
        one_hot = to_one_hot(acc_examples, unique_values)
        acc_pred = Predictor()
        acc_pred.train(one_hot, acc_labels)

    if args.latpred_path:
        log.info('Loading pre-trained latency predictor.')
        lat_pred = Predictor()
        lat_pred.load(args.latpred_path)
    else:
        log.info('Building Latency Predictor')
        one_hot = to_one_hot(lat_examples, unique_values)
        lat_pred = Predictor()
        lat_pred.train(one_hot, lat_labels)

    # Instatiate objective 'runner'
    runner = BertRunner(
        supernet=supernet,
        model_dir=args.model_dir,
        lat_predictor=lat_pred,
        acc_predictor=acc_pred,
        max_layers=max_layers,
    )

    # --------------------------------
    # DyNAS-T Search Components
    # --------------------------------

    # Define how evaluations occur, gives option for csv file
    evaluation_interface = UserEvaluationInterface(
        evaluator=runner,
        manager=supernet_manager,
        unique_values=unique_values,
        csv_path=args.csv_path,
    )

    # Instantiate Multi-Objective Problem Class
    problem = ProblemMultiObjective(
        evaluation_interface=evaluation_interface,
        param_count=supernet_manager.param_count,
        param_upperbound=supernet_manager.param_upperbound,
    )

    # Instantiate Search Manager
    search_manager = SearchAlgoManager(algorithm=args.algorithm, seed=args.seed)
    search_manager.configure_nsga2(population=args.population, num_evals=args.num_evals)

    # Run the search!
    output = search_manager.run_search(problem)

    # Process results
    results = ResultsManager(
        csv_path=args.csv_path, manager=supernet_manager, search_output=output
    )
    results.history_to_csv()
    results.front_to_csv()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='/store/.torch/ncf_nets')
    parser.add_argument('--supernet', default='BERT')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--latpred_path', help='path to latency predictor')
    parser.add_argument('--accpred_path', help='path to pre-trained acc predictor')
    parser.add_argument('--latency_csv', help='path to latency predictor')
    parser.add_argument('--accuracy_csv', help='path to pre-trained acc predictor')
    parser.add_argument('--unique_value_path', help='path to unique one-hot values')

    parser.add_argument('--input_csv', help='path to pre-trained acc predictor')

    # DyNAS-T Arguments
    parser.add_argument(
        '--algorithm',
        default='nsga2',
        choices=['nsga2', 'rnsga2'],
        help='Search algorithm, currently supports nsga2 and rnsga2',
    )
    parser.add_argument(
        '--num_evals',
        default=10000,
        type=int,
        help='number of TOTAL acc_pred, LUT samples',
    )
    parser.add_argument(
        '--csv_path', required=True, default=None, help='location to save results.'
    )
    parser.add_argument(
        '--population', default=50, type=int, help='population size for each generation'
    )
    parser.add_argument('--verbose', action='store_true', help='Flag to control output')
    parser.add_argument(
        '--search_tactic',
        default='nsga2',
        choices=['nsga2', 'rnsga2'],
        help='Search tactic (e.g., full search, warm-start, concurrent.',
    )
    args = parser.parse_args()

    log.info('\n' + '-' * 40)
    log.info('DyNAS-T Multi-Objective Search Starting')
    log.info('-' * 40)

    main(args)
