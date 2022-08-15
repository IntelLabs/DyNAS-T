"""
DyNAS-T Reference Search

SuperNetwork: OFA ResNet50

Search Tactic: Full Search (MACs measured, Accuracy predicted)

Description: Multi-Objective genetic algorithm search.
The results can be saved to the `--csv_path` file.
"""
# Imports
import argparse
import copy
import csv
import json
import pickle
import uuid
from datetime import datetime

import ofa
import torch
from fvcore.nn import FlopCountAnalysis
from ofa.imagenet_codebase.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_codebase.run_manager import ImagenetRunConfig, RunManager
from ofa.tutorial.flops_table import rm_bn_from_net

from dynast.analytics_module.results import ResultsManager
from dynast.evaluation_module.predictor import (MobileNetAccuracyPredictor,
                                                MobileNetMACsPredictor)  # TODO(Maciej) Change to `Predictor`
# DyNAS-T Specific Imports
from dynast.manager import ParameterManager
from dynast.search_module.search import (ProblemMultiObjective,
                                         SearchAlgoManager)
from dynast.utils import log


class OFAResNetRunner:
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
        self.target = 'cpu'
        self.test_size = None
        ImagenetDataProvider.DEFAULT_PATH = imagenetpath
        self.ofa_network = ofa.model_zoo.ofa_net(supernet, pretrained=True, model_dir=model_dir)
        self.run_config = ImagenetRunConfig(test_batch_size=64, n_worker=20)

    def get_subnet(self, subnet_cfg):

        self.ofa_network.set_active_subnet(ks=subnet_cfg['d'],
                                           e=subnet_cfg['e'],
                                           d=subnet_cfg['w'])
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
        inputs = torch.randn((1, 3, 224, 224), device=device)
        rm_bn_from_net(model)
        model.eval()
        # MACs=FLOPs in FVCore
        macs = FlopCountAnalysis(model, inputs).total()
        return macs/10**6

    def estimate_accuracy_top1(self, subnet_cfg):

        top1 = self.acc_predictor.predict_single(subnet_cfg)
        return top1

    def estimate_macs(self, subnet_cfg):

        macs = self.macs_predictor.predict_single(subnet_cfg)
        return macs

    def estimate_latency(self, subnet_cfg):

        # Latency Predictor
        latency = self.latency_predictor.predict_single(subnet_cfg)
        return latency


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
            'd': param_dict['d'],
            'e': param_dict['e'],
            'w': param_dict['w']
        }
        # Bug Fix - deep copy prevents accidental re-mapping of sample
        subnet_sample = copy.deepcopy(sample)

        #latency = self.evaluator.estimate_latency(subnet_sample)
        macs = self.evaluator.validate_macs(subnet_sample)
        #macs = self.evaluator.estimate_macs(self.manager.onehot_generic(x))
        top1 = self.evaluator.estimate_accuracy_top1(self.manager.onehot_generic(x))

        if self.csv_path:
            with open(self.csv_path, 'a') as f:
                writer = csv.writer(f)
                date = str(datetime.now())
                result = [subnet_sample, date, macs, top1]
                writer.writerow(result)

        # PyMoo only minimizes objectives, thus accuracy needs to be negative
        # Requires format: subnetwork, objective x, objective y
        return sample, macs, -top1



def main(args):

    # --------------------------------
    # OFA <-> DyNAS-T Interface Setup
    # --------------------------------

    # Define SuperNetwork Parameter Dictionary and Instantiate Manager
    supernet_parameters = {'d'  :  {'count' : 5,  'vars' : [0, 1, 2]},
                           'e'  :  {'count' : 18, 'vars' : [0.2, 0.25, 0.35]},
                           'w'  :  {'count' : 6,  'vars' : [0, 1, 2]} }
    supernet_manager = ParameterManager(param_dict=supernet_parameters,
                                        seed=args.seed)

    # Objective 1 (Latency or MACs)
    supernet = args.supernet
    if args.lut_path:
        log.info('Loading Latency LUT.')
        with open(args.lut_path, 'r') as f:
            lut = json.load(f)
        supernet = lut['metadata']['_net']
        assert supernet == args.supernet

    # MACs example
    if False:
        log.info('Building MACs Predictor')
        df = supernet_manager.import_csv(args.input_csv, config='config', objective='macs', column_names=['config','date','macs','top1'])
        features, labels = supernet_manager.create_training_set(df)
        macs_pred = MobileNetMACsPredictor()
        macs_pred.train(features, labels)

    # Objective 2 (Accuracy)
    if args.accpred_path:
        log.info('Loading pre-trained accuracy predictor.')
        with open(args.accpred_path, 'rb') as f:
            acc_pred = pickle.load(f)
    else:
        log.info('Building Accuracy Predictor')
        df = supernet_manager.import_csv(args.input_csv, config='config', objective='top1')
        features, labels = supernet_manager.create_training_set(df)
        acc_pred = MobileNetAccuracyPredictor()
        acc_pred.train(features, labels)

    # Instatiate objective 'runner'
    runner = OFAResNetRunner(supernet=supernet, model_dir=args.model_dir,
                             acc_predictor=acc_pred, macs_predictor=None,
                             latency_predictor=None, imagenetpath=args.dataset_path)

    # --------------------------------
    # DyNAS-T Search Components
    # --------------------------------

    # Define how evaluations occur, gives option for csv file
    evaluation_interface = UserEvaluationInterface(evaluator=runner,
                                                   manager=supernet_manager,
                                                   csv_path=args.csv_path)

    # Random Sample Example
    if False:
        samples = list()
        while len(samples) < 2000:
            sample = supernet_manager.random_sample()
            if sample not in samples:
                samples.append(sample)

        for individual in samples:
            evaluation_interface.eval_subnet(individual)

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
    parser.add_argument('--supernet', default='ofa_resnet50',
                        choices=['ofa_mbv3_d234_e346_k357_w1.0', 'ofa_mbv3_d234_e346_k357_w1.2'])
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--lut_path', help='path to latency look-up table (required)')
    parser.add_argument('--accpred_path', help='path to pre-trained acc predictor')
    parser.add_argument('--input_csv', help='path to pre-trained acc predictor')
    parser.add_argument('--dataset_path', help='The path of dataset (e.g. ImageNet)',
                        type=str, default='/datasets/imagenet-ilsvrc2012')

    # DyNAS-T Arguments
    parser.add_argument('--algorithm', default='nsga2', choices=['nsga2', 'rnsga2'],
                        help='Search algorithm, currently supports nsga2 and rnsga2')
    parser.add_argument('--num_evals', default=100000, type=int, help='number of TOTAL acc_pred, LUT samples')
    parser.add_argument('--csv_path', required=True, default=None, help='location to save results.')
    parser.add_argument('--population', default=50, type=int, help='population size for each generation')
    parser.add_argument('--verbose', action='store_true', help='Flag to control output')
    args = parser.parse_args()

    log.info('\n'+'-'*40)
    log.info('DyNAS-T Multi-Objective Search Starting')
    log.info('-'*40)

    main(args)