import argparse
import copy
import csv
from datetime import datetime

import torch
from tqdm import tqdm

from dynast.analytics_module.results import ResultsManager
from dynast.evaluation_module.predictor import (ResNet50AccuracyPredictor,
                                                ResNet50LatencyPredictor)
from dynast.manager import ParameterManager
from dynast.search_module.search import (ProblemMultiObjective,
                                         SearchAlgoManager)
from dynast.utils import get_hostname, log
from dynast.utils.bootstrapnas import (SUPERNET_PARAMETERS, BNASRunner,
                                       get_supernet)
from dynast.utils.cache import Cache


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

    def __init__(self, evaluator: BNASRunner, manager: ParameterManager, csv_path: str = None):
        self.evaluator = evaluator
        self.manager = manager
        self.cache = Cache('bnas_resnet50_bn4000_b128')
        self.csv_path = csv_path

    def eval_subnet(self, x, validation=False):
        # PyMoo vector to Elastic Parameter Mapping
        param_dict = self.manager.translate2param(x)

        sample = {
            'd': param_dict['d'],
            'e': param_dict['e'],
            'w': param_dict['w'],
        }

        # Bug Fix - deep copy prevents accidental re-mapping of sample
        subnet_sample = copy.deepcopy(sample)

        cache_key = Cache.key(x)

        if validation:
            if not self.cache.exists(cache_key):
                log.info('Key {} not found in the cache.'.format(cache_key))
                self.evaluator.quantize(subnet_sample)
                top1, top5, gflops, model_params = self.evaluator.validate_subnet(subnet_sample)
                latency = self.evaluator.benchmark_subnet()
                date = str(datetime.now())

                self.cache.update(key=cache_key, payload=[subnet_sample, date, float(latency), float(top1)])
            else:
                subnet_sample, date, latency, top1 = self.cache.get(key=cache_key)
                log.info('Found key {} in cache. Re-using: {}'.format(cache_key,
                         [subnet_sample, date, float(latency), float(top1)]))
            log.info('Cache hits: {}'.format(self.cache._hits))
        else:
            top1 = self.evaluator.estimate_accuracy_top1(self.manager.onehot_generic(x))
            latency = self.evaluator.estimate_latency(self.manager.onehot_generic(x))

        if self.csv_path:
            with open(self.csv_path, 'a') as f:
                writer = csv.writer(f)
                date = str(datetime.now())
                result = [subnet_sample, date, float(latency), float(top1)]
                writer.writerow(result)

        # PyMoo only minimizes objectives, thus accuracy needs to be negative
        # Requires format: subnetwork, objective x, objective y
        return sample, latency, -top1


def main(args):
    log.info('Starting BootstrapNASResnet50 Search')
    supernet = get_supernet(args.supernet_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f'Using device: {device}')

    # --------------------------------
    # DyNAS-T Search Setup
    # --------------------------------

    # Define SuperNetwork Parameter Dictionary and Instantiate Manager
    supernet_manager = ParameterManager(param_dict=SUPERNET_PARAMETERS, seed=args.seed)

    # Instatiate objective 'runner', treating latency LUT as ground truth for latency in this example
    validation_runner = BNASRunner(
        supernet=supernet,
        acc_predictor=None,
        latency_predictor=None,
        bn_samples=args.bn_samples,
        batch_size=args.batch_size,
    )

    # Concurrent Search
    validated_population = args.csv_path_val_output
    print(f'[Info] Validated population file: {args.csv_path_val_output}')

    # Define how evaluations occur, gives option for csv file
    validation_interface = UserEvaluationInterface(evaluator=validation_runner,
                                                   manager=supernet_manager,
                                                   csv_path=validated_population)

    # clear validation file
    with open(validated_population, 'w') as f:
        writer = csv.writer(f)

    # NOTE(Maciej): May include duplicates (desired behaviour in this case)
    last_population = [supernet_manager.random_sample() for _ in range(args.population)]

    # --------------------------------
    # DyNAS-T ConcurrentNAS Loop
    # --------------------------------

    num_loops = 10
    for loop in range(1, num_loops+1):
        print(f'[Info] Starting ConcurrentNAS loop {loop} of {num_loops}.')

        for individual in tqdm(last_population, desc='Population'):
            print(individual)
            validation_interface.eval_subnet(individual, validation=True)

        print('[Info] Training "weak" Latency predictor.')
        df = supernet_manager.import_csv(validated_population, config='config', objective='latency',
                                         column_names=['config', 'date', 'latency', 'top1'])
        features, labels = supernet_manager.create_training_set(df)
        lat_pred = ResNet50LatencyPredictor()
        lat_pred.train(features, labels)

        print('[Info] Training "weak" accuracy predictor.')
        df = supernet_manager.import_csv(validated_population, config='config', objective='top1',
                                         column_names=['config', 'date', 'latency', 'top1'])
        features, labels = supernet_manager.create_training_set(df)
        acc_pred = ResNet50AccuracyPredictor()
        acc_pred.train(features, labels)

        runner_predictor = BNASRunner(supernet=supernet,
                                      acc_predictor=acc_pred,
                                      latency_predictor=lat_pred,
                                      bn_samples=args.bn_samples,
                                      batch_size=args.batch_size,
                                      )

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
                                       num_evals=args.num_evals)

        # Run the search!
        output = search_manager.run_search(problem)
        last_population = output.pop.get('X')

        # Process results - use for inner loop debug
        if False:
            results = ResultsManager(csv_path=f'./results_temp/loop{loop}_{args.csv_path}',
                                     manager=supernet_manager,
                                     search_output=output)
            results.history_to_csv()
            results.front_to_csv()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--supernet_path', default='./supernet/torchvision_resnet50_supernet.pth',
                        type=str, help='Path to saved supernet')
    parser.add_argument('--csv_path_val_output', default=None, help='location to save results.')
    parser.add_argument('--bn_samples', default=4000, type=int,
                        help='Numer of samples to be used to adjust batch norm. NOTE: Will be adjusted to batch size.')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument('--verbose', action='store_true', help='Flag to control output')
    parser.add_argument('--algorithm',
                        default='nsga2', choices=['nsga2', 'rnsga2'],
                        help='GA algorithm, currently supports nsga2 and rnsga2')
    parser.add_argument('--num_evals',
                        default=20000, type=int, help='number of evaluations')
    parser.add_argument('--population',
                        default=50, type=int, help='population size for each generation')

    args = parser.parse_args()

    if not args.csv_path_val_output:
        args.csv_path_val_output = 'results/linas_{}.csv'.format(get_hostname())
        log.warning('Argument `csv_path_val_output` not set. Using default: `{}`'.format(args.csv_path_val_output))

    main(args)
