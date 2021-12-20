"""
DyNAS-T Reference Search

SuperNetwork: OFA MobileNetV3

Search Tactic: Concurrent Search (predictor training then search)

Description: Multi-Objective genetic algorithm search.
Assumes that the user has:
    * a latency LUT to predict the network, and the metadata
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
import copy
import pickle
import uuid
from fvcore.nn import FlopCountAnalysis, parameter_count
from handi.runner import OVRunner
import shutil
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# OFA Specific Imports
from ofa.tutorial.latency_table import LatencyEstimator
from ofa.imagenet_codebase.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_codebase.run_manager import ImagenetRunConfig
from ofa.imagenet_codebase.run_manager import RunManager
import ofa

# DyNAS-T Specific Imports
from dynast.manager import ParameterManager
from dynast.evaluation_module.predictor import MobileNetAccuracyPredictor, MobileNetLatencyPredictor
from dynast.search_module.search import SearchAlgoManager, ProblemMultiObjective
from dynast.analytics_module.results import ResultsManager





class OFARunner:
    '''
    The OFARunner is responsible for 'running' the subnetwork evaluation.
    '''
    def __init__(self, supernet, model_dir, acc_predictor, macs_predictor,
                 latency_predictor, imagenetpath):

        self.supernet = supernet
        self.model_dir = model_dir
        self.acc_predictor = acc_predictor
        self.macs_predictor = macs_predictor
        self.latency_predictor = latency_predictor
        #if isinstance(lut, dict):
        #    self.lut = lut
        #else:
        #    with open(lut, 'r') as f:
        #        self.lut = json.load(f)
        #self.latencyEstimator = LatencyEstimator(url=self.lut)
        #self.width = float(supernet[-3:])

        # Validation setup
        self.target = 'cpu'
        #self.batch_size = 1
        #self.batch_size_val = 64
        self.test_size = None
        ImagenetDataProvider.DEFAULT_PATH = imagenetpath
        self.ofa_network = ofa.model_zoo.ofa_net(supernet, pretrained=True, model_dir=model_dir)
        self.run_config = ImagenetRunConfig(test_batch_size=64, n_worker=20)

    def get_subnet(self, subnet_cfg):

        self.ofa_network.set_active_subnet(ks=subnet_cfg['ks'],
                                           e=subnet_cfg['e'],
                                           d=subnet_cfg['d'])
        self.subnet = self.ofa_network.get_active_subnet(preserve_weight=True)
        self.subnet.eval()
        return self.subnet

    def validate_top1(self, subnet_cfg, target=None, measure_latency=True):

        #import ofa.config
        if target is None:
            target = self.target
        subnet = self.get_subnet(subnet_cfg)
        #self.subnet = subnet
        folder_name = '.torch/tmp-{}'.format(uuid.uuid1().hex)
        run_manager = RunManager('{}/eval_subnet'.format(folder_name), subnet,
                                self.run_config, init=False, print_info=False)
        bn_mean, bn_var = run_manager.reset_running_statistics(net=subnet)

        # Test sampled subnet
        self.run_config.data_provider.assign_active_img_size(subnet_cfg['r'][0])
        loss, top1, top5 = run_manager.validate(net=subnet, test_size=self.test_size, no_logs=True)
        return top1

    def validate_macs(self, subnet_cfg):

        device = 'cpu'
        model = self.get_subnet(subnet_cfg)
        model = model.to(device)
        # batch, channels, res, res
        inputs = torch.randn((1, 3, 224, 224), device=device)
        rm_bn_from_net(model)
        model.eval()
        flops = FlopCountAnalysis(model, inputs).total()
        return flops/10**6

    def estimate_accuracy_top1(self, subnet_cfg):

        # Ridge Predictor - 135 vector
        top1 = self.acc_predictor.predict_single(subnet_cfg)
        return top1

    def estimate_macs(self, subnet_cfg):

        # Ridge Predictor - 135 vector
        macs = self.macs_predictor.predict_single(subnet_cfg)
        return macs

    def estimate_latency(self, subnet_cfg):

        # LUT Latency Predictor
        latency = self.latency_predictor.predict_single(subnet_cfg)
        return latency

    #def estimate_latency_lut(self, subnet_cfg):

        # LUT Latency Predictor
    #    latency = self.latencyEstimator.predict_network_latency_given_spec(subnet_cfg, width=self.width)
    #    return latency


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

    def __init__(self, evaluator, manager):
        self.evaluator = evaluator
        self.manager = manager

    def eval_subnet(self, x, validation=False, csv_path=None):
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

        if validation:
            latency_ov_quant, throughput_ov_quant, top1_ov_quant, top5_ov_quant, \
            latency_ov_fp32, throughput_ov_fp32, top1_ov_fp32, top5_ov_fp32 = self.evaluator.evaluate_quantized(
                subnet_cfg=subnet_sample,
                cores=56,
                quant_policy='DefaultQuantization',
                exp_name='test',
                test_size=100,
                test_ov_fp32=False,
            )

            latency = latency_ov_quant
            top1 = top1_ov_quant
            #top1 = self.evaluator.estimate_accuracy_top1(self.manager.onehot_generic(x))
        else:
            latency = self.evaluator.estimate_latency(self.manager.onehot_generic(x))
            top1 = self.evaluator.estimate_accuracy_top1(self.manager.onehot_generic(x))

        if csv_path:
            with open(csv_path, 'a') as f:
                writer = csv.writer(f)
                date = str(datetime.now())
                result = [subnet_sample, date, float(latency), float(top1)]
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


    runner_validator = OVRunner(
            net=args.supernet,
            path=args.dataset_path,
            model_dir=args.model_dir,
            batch_size=128,
            batch_size_val=128,
            workers=2,
            test_size=100,
        )


    supernet = args.supernet
    if args.lut_path:
        print('[Info] Loading Latency LUT.')
        with open(args.lut_path, 'r') as f:
            lut = json.load(f)
        supernet = lut['metadata']['_net']
        assert supernet == args.supernet

    #print('[Info] Building Accuracy Predictor')
    #df = supernet_manager.import_csv(args.input_csv, config='config', objective='top1')
    #features, labels = supernet_manager.create_training_set(df)
    #acc_pred_main = MobileNetAccuracyPredictor()
    #acc_pred_main.train(features, labels)

    # Instatiate objective 'runner', treating latency LUT as ground truth for latency in this example
    #runner_validator = OFARunner(supernet=supernet,
    #                   model_dir=args.model_dir,
    #                   acc_predictor=None,
    #                   macs_predictor=None,
    #                   latency_predictor=None,
    #                   imagenetpath=args.dataset_path_ov)

    # Define how evaluations occur, gives option for csv file
    validation_interface = UserEvaluationInterface(evaluator=runner_validator,
                                                   manager=supernet_manager)

    # Take initial validation measurements to start concurrent search
    # validated_population is the csv that will contain all the validated pop results

    if False:
        validated_population = 'val_set_int8_p50.csv'
        with open(validated_population, 'w') as f:
            writer = csv.writer(f)
        last_population = [supernet_manager.random_sample() for _ in range(args.population)]

    if True:
        validated_population = 'val_set_int8_p50.csv'
        #shutil.copy2('val_set_int8_start.csv',
        #            validated_population)

    # --------------------------------
    # DyNAS-T ConcurrentNAS Loop
    # --------------------------------

    num_loops = 10
    for loop in range(1, num_loops+1):
        print(f'[Info] Starting ConcurrentNAS loop {loop} of {num_loops}.')

        if loop != 1:
            for individual in last_population:
                print(individual)
                validation_interface.eval_subnet(individual, validation=True, csv_path=validated_population)

        print('[Info] Training "weak" latency predictor.')
        df = supernet_manager.import_csv(validated_population, config='config', objective='latency',
            column_names=['config','date','latency','top1'])
        features, labels = supernet_manager.create_training_set(df)
        lat_pred = MobileNetLatencyPredictor()
        lat_pred.train(features, labels)

        print('[Info] Training "weak" accuracy predictor.')
        df = supernet_manager.import_csv(validated_population, config='config', objective='top1',
            column_names=['config','date','latency','top1'])
        features, labels = supernet_manager.create_training_set(df)
        acc_pred = MobileNetAccuracyPredictor()
        acc_pred.train(features, labels)

        runner_predictor = OFARunner(supernet=supernet, model_dir=args.model_dir, macs_predictor=None,
            imagenetpath=args.dataset_path, acc_predictor=acc_pred, latency_predictor=lat_pred)

        prediction_interface = UserEvaluationInterface(evaluator=runner_predictor,
                                                       manager=supernet_manager)

        # Instantiate Multi-Objective Problem Class
        problem = ProblemMultiObjective(evaluation_interface=prediction_interface,
                                        param_count=supernet_manager.param_count,
                                        param_upperbound=supernet_manager.param_upperbound)

        # Instantiate Search Manager
        search_manager = SearchAlgoManager(algorithm=args.algorithm,
                                        seed=args.seed)
        search_manager.configure_nsga2(population=args.population,
                                    num_evals=100000)

        # Run the search!
        output = search_manager.run_search(problem)
        last_population = output.pop.get('X')

        # Process results
        results = ResultsManager(csv_path=f'./results_temp/loop{loop}c_{args.csv_path}',
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
    parser.add_argument('--dataset_path', help='The path of dataset (e.g. ImageNet)',
                        type=str, default='/datasets/imagenet-ilsvrc2012')
    parser.add_argument('--dataset_path_ov', help='The path of OV dataset (e.g. ImageNet)',
                        type=str, default='~/nosnap/dataset/pot-acc-aware-quant/ImageNet/original')


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
    print('DyNAS-T Multi-Objective Concurrent Search Starting')
    print('-'*40)

    main(args)