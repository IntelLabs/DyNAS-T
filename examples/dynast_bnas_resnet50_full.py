import argparse
import copy
import csv
from datetime import datetime

import torch

from dynast.analytics_module.results import ResultsManager
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

    def __init__(self, evaluator, manager, csv_path):
        self.csv_path = csv_path
        self.evaluator = evaluator
        self.manager = manager
        self.cache = Cache('bnas_resnet50_bn4000_b128')

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
    runner = BNASRunner(
        supernet=supernet,
        acc_predictor=None,
        latency_predictor=None,
        bn_samples=args.bn_samples,
        batch_size=args.batch_size,
    )

    # Define how evaluations occur, gives option for csv file
    evaluation_interface = UserEvaluationInterface(evaluator=runner, manager=supernet_manager, csv_path=args.csv_path)

    # --------------------------------
    # DyNAS-T Search Components
    # --------------------------------

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
    if False:
        results = ResultsManager(csv_path=args.csv_path,
                                 manager=supernet_manager,
                                 search_output=output)
        results.history_to_csv()
        results.front_to_csv()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--supernet_path', default='./supernet/torchvision_resnet50_supernet.pth',
                        type=str, help='Path to saved supernet')
    parser.add_argument('--csv_path', default=None, help='location to save results.')
    parser.add_argument('--bn_samples', default=4000, type=int,
                        help='Numer of samples to be used to adjust batch norm. NOTE: Will be adjusted to batch size.')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument('--verbose', action='store_true', help='Flag to control output')
    parser.add_argument('--algorithm',
                        default='nsga2', choices=['nsga2', 'rnsga2'],
                        help='GA algorithm, currently supports nsga2 and rnsga2')
    parser.add_argument('--num_evals',
                        default=5000, type=int, help='number of evaluations')
    parser.add_argument('--population',
                        default=50, type=int, help='population size for each generation')

    args = parser.parse_args()

    if not args.csv_path:
        args.csv_path = 'results/full_{}.csv'.format(get_hostname())
        log.warning('Argument `csv_path` not set. Using default: `{}`'.format(args.csv_path))

    main(args)
