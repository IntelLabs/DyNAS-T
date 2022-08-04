import time

import autograd.numpy as anp
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.core.problem import Problem
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize

from dynast.utils import log


class SearchAlgoManager:
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
        Support different engine types (e.g. pymoo, optuna, nni, deap)
    '''

    def __init__(self, algorithm='nsga2', seed=0, verbose=False, engine='pymoo'):
        self.algorithm = algorithm
        self.seed = seed
        self.verbose = verbose
        self.engine = engine
        if self.algorithm == 'nsga2' and self.engine == 'pymoo':
            self.algorithm_def = self.configure_nsga2()
        elif self.algorithm == 'rnsga2' and self.engine == 'pymoo':
            self.algorithm_def = self.configure_rnsga2()
        else:
            log.warning('Algorithm specified not implemented.')
            raise NotImplementedError

    def configure_nsga2(self, population=50, num_evals=1000, warm_pop=None,
                        crossover_prob=0.9, crossover_eta=15.0,
                        mutation_prob=0.02, mutation_eta=20.0):
        self.population = population
        self.num_evals = num_evals

        if type(warm_pop) == 'numpy.ndarray':
            log.info('Using warm start population')
            sample_strategy = warm_pop
        else:
            sample_strategy = get_sampling("int_lhs")

        self.algorithm_def = NSGA2(pop_size=self.population,
            sampling=sample_strategy,
            crossover=get_crossover("int_sbx", prob=crossover_prob, eta=crossover_eta),
            mutation=get_mutation("int_pm", prob=mutation_prob, eta=mutation_eta),
            eliminate_duplicates=True)

        if num_evals % population != 0:
            log.warning('Number of samples not divisible by population size')

    def configure_rnsga2(self, population=50, num_evals=1000, warm_pop=None,
                         ref_points=[[0, 0]],
                         crossover_prob=0.9, crossover_eta=15.0,
                         mutation_prob=0.02, mutation_eta=20.0):
        self.population = population
        self.num_evals = num_evals

        if type(warm_pop) == 'numpy.ndarray':
            log.info('Using warm start population')
            sample_strategy = warm_pop
        else:
            sample_strategy = get_sampling("int_lhs")

        log.info('Reference points for RNSGA-II are: {}'.format(ref_points))
        reference_points = np.array(ref_points)  # lat=0, 1/acc=0

        self.algorithm_def = RNSGA2(
            ref_points=reference_points,
            pop_size=50,
            epsilon=0.01,
            normalization='front',
            extreme_points_as_reference_points=False,
            sampling=sample_strategy,
            crossover=get_crossover("int_sbx", prob=crossover_prob, eta=crossover_eta),
            mutation=get_mutation("int_pm", prob=mutation_prob, eta=mutation_eta),
            weights=np.array([0.5, 0.5]),
            eliminate_duplicates=True)

        if num_evals % population != 0:
            log.warning('Number of samples not divisible by population size')

    def run_search(
        self,
        problem: Problem,
        save_history: bool = False,
    ):

        log.info('Running Search')
        start_time = time.time()

        if self.algorithm == 'nsga2' and self.engine == 'pymoo':
            result = minimize(problem,
                              self.algorithm_def,
                              ('n_gen', int(self.num_evals/self.population)),
                              seed=self.seed,
                              save_history=save_history,
                              verbose=self.verbose)
        else:
            raise NotImplementedError

        log.info('Success! Search Took {:.3f} seconds.'.format(time.time()-start_time))

        return result

class ProblemSingleObjective(Problem):

    def __init__(self, evaluation_interface, param_count, param_upperbound):
        super().__init__(n_var=param_count, n_obj=1, n_constr=0,
                         xl=0, xu=param_upperbound, type_var=np.int)

        self.evaluation_interface = evaluation_interface

    def _evaluate(self, x, out, *args, **kwargs):

        # Store results for a given generation for PyMoo
        objective_arr = list()

        # Measure new individuals
        for i in range(len(x)):

            _, objective = self.evaluation_interface.eval_subnet(x[i])

            objective_arr.append(objective)

        print('.', end='', flush=True)

        # Update PyMoo with evaluation data
        out["F"] = anp.column_stack([objective_arr])


class ProblemMultiObjective(Problem):

    def __init__(self, evaluation_interface, param_count, param_upperbound):
        super().__init__(n_var=param_count, n_obj=3, n_constr=0,
                         xl=0, xu=param_upperbound, type_var=np.int)

        self.evaluation_interface = evaluation_interface

    def _evaluate(self, x, out, *args, **kwargs):

        # Store results for a given generation for PyMoo
        objective_x_arr, objective_y_arr = list(), list()

        # Measure new individuals
        for i in range(len(x)):

            _, objective_x, objective_y = self.evaluation_interface.eval_subnet(x[i])

            objective_x_arr.append(objective_x)
            objective_y_arr.append(objective_y)

        print('.', end='', flush=True)

        # Update PyMoo with evaluation data
        out["F"] = anp.column_stack([objective_x_arr, objective_y_arr])

class ProblemManyObjective(Problem):

    def __init__(self, evaluation_interface, param_count, param_upperbound):
        super().__init__(n_var=param_count, n_obj=3, n_constr=0,
                         xl=0, xu=param_upperbound, type_var=np.int)

        self.evaluation_interface = evaluation_interface

    def _evaluate(self, x, out, *args, **kwargs):

        # Store results for a given generation for PyMoo
        objective_x_arr, objective_y_arr, objective_z_arr = list(), list(), list()

        # Measure new individuals
        for i in range(len(x)):

            _, objective_x, objective_y, objective_z = self.evaluation_interface.eval_subnet(x[i])

            objective_x_arr.append(objective_x)
            objective_y_arr.append(objective_y)
            objective_z_arr.append(objective_z)

        print('.', end='', flush=True)

        # Update PyMoo with evaluation data
        out["F"] = anp.column_stack([objective_x_arr, objective_y_arr, objective_z_arr])
