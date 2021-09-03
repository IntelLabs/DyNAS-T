from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.rnsga2 import RNSGA2
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.model.problem import Problem
from pymoo.optimize import minimize

import time
import csv
import numpy as np
import autograd.numpy as anp
from datetime import datetime



class MultiObjectiveSearchManager: 
    '''
    Manages the search parameters for multi-objective search (2 objectives).

    Parameters
    ----------
    algorithm : string
        Define a multi-objective search algorithm.
    seed : int
        Seed value for Pymoo search.
    verbose : Boolean
        Verbosity option
    engine : string
        Support different engine types (e.g. pymoo, optuna, nni)
    '''

    def __init__(self, algorithm='nsga2', seed=0, verbose=False, engine='pymoo'):
        self.algorithm = algorithm
        self.seed = seed
        self.verbose = verbose
        self.engine = engine
        if self.algorithm == 'nsga2':
            if self.engine == 'pymoo':
                self.algorithm_pymoo = self.configure_pymoo_nsga2()
            elif self.engine == 'optuna':
                raise NotImplementedError
            else:
                raise NotImplementedError
        elif self.algorithm == 'rnsga2':
            print('[Warning] RNSGA2 not supported')
        else:
            print('[Warning] Invalid Algorithm')

            
    def configure_pymoo_nsga2(self, population=50, num_evals=1000):
        self.population = population
        self.num_evals = num_evals
        self.algorithm_pymoo = NSGA2(pop_size=self.population, 
            sampling=get_sampling("int_lhs"),
            crossover=get_crossover("int_sbx", prob=0.9, eta=10.0),
            mutation=get_mutation("int_pm", prob=None, eta=3.0),
            eliminate_duplicates=True)

        if num_evals % population != 0:
            print('[Warning] Number of samples not divisible by population size')


    def run_search(self, problem):

        if self.verbose:
            print('[Info] Results will be written to (csv format): {}'.format(problem.csv_path))
            print('[Info] Running Search .', end='', flush=True)
            start_time = time.time()

        if self.algorithm == 'nsga2' and self.engine == 'pymoo':    
            result = minimize(problem, self.algorithm_pymoo, 
                           ('n_gen', int(self.num_evals/self.population)), 
                           seed=self.seed, 
                           verbose=self.verbose)
        else:
            raise NotImplementedError

        if self.verbose:
            print('[Info] Search Took {} seconds.'.format(time.time()-start_time))

        return result


class PymooImageClfProblem(Problem):

    def __init__(self, latency_runner, accuracy_runner, csv_path, archManager):
        super().__init__(n_var=45, n_obj=2, n_constr=0, xl=0, xu=2, type_var=np.int)
        
        self.latency_runner = latency_runner
        self.accuracy_runner = accuracy_runner
        self.csv_path = csv_path
        self.archManager = archManager

    def _evaluate(self, x, out, *args, **kwargs):
        
        # Store results for a given generation for PyMoo
        lat_arr, top1_arr = [], []

        # Measure new individuals
        for i in range(len(x)):

            # Integer to Elastic Parameter Mapping
            ks_map =  {0:3, 1:5, 2:7}
            e_map =  {0:3, 1:4, 2:6}
            d_map =  {0:2, 1:3, 2:4}
            ks_list, e_list, d_list = [], [], []
            for j in range(20):
                ks_list.append(ks_map[int(round(x[i][j]))])
            for j in range(20):
                e_list.append(e_map[int(round(x[i][j+20]))])
            for j in range(5):
                d_list.append(d_map[int(round(x[i][j+40]))])
            
            sample = {
                'wid': None,
                'ks': ks_list,
                'e': e_list,
                'd': d_list,
                'r': [224]
            } 

            with open(self.csv_path, 'a') as f:
                writer = csv.writer(f)

                # retreive the next result and save it in the json database
                lat = self.latency_runner.estimate_latency(sample) 
                top1, top5, config = self.accuracy_runner.estimate_accuracy(sample)
                config_string = self.archManager.serialize_sample(config)
                date = str(datetime.now())

                # append to a running list
                result = [config_string, date, lat, 0, 0, top1, top5]
                writer.writerow(result)

                lat_arr.append(lat)

                # Need to treat accuracy as maximization, hence the inverse
                # 100x in denominator to prevent rounding issues
                top1_arr.append((1/top1)*100)   

        print('.', end='', flush=True)  

        # Update PyMoo with evaluation data     
        out["F"] = anp.column_stack([lat_arr, top1_arr])


