"""
DyNAS-T Reference Search

SuperNetwork: OFA MobileNetV3

Search Tactic: Full Search (predictor training then search)

Description: Multi-Objective genetic algorithm search.
Assumes that the user has:
    * a latency LUT to predict the network, and the metadata
      with information on the hardware config
    * an AccuracyPredictor neural network that predicts the acc
The results can be saved to the `--csv_path` file.
"""
# Imports
import argparse
import copy
import csv
import json
import pickle
from datetime import datetime

import numpy as np
# OFA Specific Imports
from ofa.tutorial.latency_table import LatencyEstimator

from dynast.analytics_module.results import ResultsManager
from dynast.evaluation_module.predictor import (MobileNetAccuracyPredictor,
                                                MobileNetMACsPredictor)  # TODO(Maciej) Change to `Predictor`
# DyNAS-T Specific Imports
from dynast.manager import ParameterManager
from dynast.search_module.search import (ProblemMultiObjective,
                                         SearchAlgoManager)


class OFARunner:
    '''
    The OFARunner is responsible for 'running' the subnetwork evaluation.
    '''
    def __init__(self, supernet, model_dir, lut, acc_predictor, macs_predictor):

        self.supernet = supernet
        self.model_dir = model_dir
        self.acc_predictor = acc_predictor
        self.macs_predictor = macs_predictor
        if isinstance(lut, dict):
            self.lut = lut
        else:
            with open(lut, 'r') as f:
                self.lut = json.load(f)
        self.latencyEstimator = LatencyEstimator(url=self.lut)
        self.width = float(supernet[-3:])

    def estimate_accuracy_custom(self, subnet_cfg):

        # Ridge Predictor - 120 vector
        top1 = self.acc_predictor.predict_single(self.onehot_custom(subnet_cfg['ks'],
            subnet_cfg['e'], subnet_cfg['d']))

        return top1

    def estimate_accuracy_top1(self, subnet_cfg):

        # Ridge Predictor - 135 vector
        top1 = self.acc_predictor.predict_single(subnet_cfg)
        print(top1)
        return top1

    def estimate_macs(self, subnet_cfg):

        # Ridge Predictor - 135 vector
        macs = self.macs_predictor.predict_single(subnet_cfg)
        return macs

    def estimate_latency(self, subnet_cfg):

        # LUT Latency Predictor
        latency = self.latencyEstimator.predict_network_latency_given_spec(subnet_cfg, width=self.width)
        return latency

    def construct_maps(self, keys):
        d = dict()
        keys = list(set(keys))
        for k in keys:
            if k not in d:
                d[k] = len(list(d.keys()))
        return d

    def onehot_custom(self, ks_list, ex_list, d_list):

        ks_map = self.construct_maps(keys=(3, 5, 7))
        ex_map = self.construct_maps(keys=(3, 4, 6))
        dp_map = self.construct_maps(keys=(2, 3, 4))

        # This function converts a network config to a feature vector (128-D).
        start = 0
        end = 4
        for d in d_list:
            for j in range(start+d, end):
                ks_list[j] = 0
                ex_list[j] = 0
            start += 4
            end += 4

        # convert to onehot
        ks_onehot = [0 for _ in range(60)]
        ex_onehot = [0 for _ in range(60)]

        for i in range(20):
            start = i * 3
            if ks_list[i] != 0:
                ks_onehot[start + ks_map[ks_list[i]]] = 1
            if ex_list[i] != 0:
                ex_onehot[start + ex_map[ex_list[i]]] = 1

        return np.array(ks_onehot + ex_onehot)


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

    def __init__(self, evaluator, manager, csv_path=None):
        self.evaluator = evaluator
        self.manager = manager
        self.csv_path = csv_path

    def eval_subnet(self, x):
        # PyMoo vector to Elastic Parameter Mapping
        param_dict = self.manager.translate2param(x)

        sample = {
            'wid': None,
            'ks': param_dict['ks'],
            'e': param_dict['e'],
            'd': param_dict['d'],
            'r': [224]
        }
        # Bug Fix - deep copy prevents accidental re-mapping of sample
        subnet_sample = copy.deepcopy(sample)

        latency = self.evaluator.estimate_latency(subnet_sample)
        #macs = self.evaluator.estimate_macs(self.manager.onehot_generic(x))
        top1 = self.evaluator.estimate_accuracy_top1q(self.manager.onehot_generic(x))

        if self.csv_path:
            with open(self.csv_path, 'a') as f:
                writer = csv.writer(f)
                date = str(datetime.now())
                result = [subnet_sample, date, latency, top1]
                writer.writerow(result)

        # PyMoo only minimizes objectives, thus accuracy needs to be negative
        # Requires format: subnetwork, objective x, objective y
        return sample, latency, -top1



def main(args):

    # --------------------------------
    # OFA <-> DyNAS-T Interface Setup
    # --------------------------------

    # Define SuperNetwork Parameter Dictionary and Instantiate Manager
    supernet_parameters = {'ks'  :  {'count' : 20, 'vars' : [3, 5, 7]},
                           'e'   :  {'count' : 20, 'vars' : [3, 4, 6]},
                           'd'   :  {'count' : 5,  'vars' : [2, 3, 4]} }
    supernet_manager = ParameterManager(param_dict=supernet_parameters,
                                        seed=args.seed)

    # Objective 1 (Latency or MACs)
    supernet = args.supernet
    print('[Info] Loading Latency LUT.')
    with open(args.lut_path, 'r') as f:
        lut = json.load(f)
    supernet = lut['metadata']['_net']
    assert supernet == args.supernet

    # MACs example
    if False:
        print('[Info] Building MACs Predictor')
        df = supernet_manager.import_csv(args.input_csv, config='config', objective='macs')
        features, labels = supernet_manager.create_training_set(df)
        macs_pred = MobileNetMACsPredictor()
        macs_pred.train(features, labels)

    # Objective 2 (Accuracy)
    if args.accpred_path:
        print('[Info] Loading pre-trained accuracy predictor.')
        with open(args.accpred_path, 'rb') as f:
            acc_pred = pickle.load(f)
    else:
        print('[Info] Building Accuracy Predictor')
        df = supernet_manager.import_csv(args.input_csv, config='config', objective='top1')
        features, labels = supernet_manager.create_training_set(df)
        acc_pred = MobileNetAccuracyPredictor()
        acc_pred.train(features, labels)

    # Instatiate objective 'runner'
    runner = OFARunner(supernet=supernet, model_dir=args.model_dir, lut=lut,
                       acc_predictor=acc_pred, macs_predictor=None)

    # --------------------------------
    # DyNAS-T Search Components
    # --------------------------------

    # Define how evaluations occur, gives option for csv file
    evaluation_interface = UserEvaluationInterface(evaluator=runner,
                                                   manager=supernet_manager,
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
    # SuperNetwork Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='/store/.torch/ofa_nets')
    parser.add_argument('--supernet', default='ofa_mbv3_d234_e346_k357_w1.0',
                        choices=['ofa_mbv3_d234_e346_k357_w1.0', 'ofa_mbv3_d234_e346_k357_w1.2'])
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--lut_path', help='path to latency look-up table (required)')
    parser.add_argument('--accpred_path', help='path to pre-trained acc predictor')
    parser.add_argument('--input_csv', help='path to pre-trained acc predictor')

    # DyNAS-T Arguments
    parser.add_argument('--algorithm', default='nsga2', choices=['nsga2', 'rnsga2'],
                        help='Search algorithm, currently supports nsga2 and rnsga2')
    parser.add_argument('--num_evals', default=100000, type=int, help='number of TOTAL acc_pred, LUT samples')
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