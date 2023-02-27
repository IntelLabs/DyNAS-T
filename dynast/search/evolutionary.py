# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import autograd.numpy as anp
import numpy as np
from pymoo.algorithms.moo.age import AGEMOEA  # TODO(macsz) Add lazy imports
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from dynast.utils import log


class EvolutionaryManager:
    '''
    Manages the search parameters for the DyNAS-T single/multi-objective search.

    Parameters
    ----------
    algorithm : string
        Define a multi-objective search algorithm.
    seed : int
        Seed value for Pymoo search.
    verbose : Boolean
        Verbosity option
    engine : string
        Support different engine types (e.g. pymoo)
    n_obj: int
        Number of objectives to optimize for.
    '''

    def __init__(
        self,
        algorithm: str = 'nsga2',
        seed: int = 0,
        verbose: bool = False,
        engine: str = 'pymoo',
        n_obj: int = 2,
    ):
        self.algorithm = algorithm
        self.seed = seed
        self.verbose = verbose
        self.engine = engine
        self.n_obj = n_obj
        self.set_algorithm_def()

    def set_algorithm_def(self):
        if self.algorithm == 'cmaes' and self.n_obj == 1:
            self.algorithm_def = self.configure_cmaes()
            self.engine = 'pymoo'
        elif self.algorithm == 'ga' and self.n_obj == 1:
            self.algorithm_def = self.configure_ga()
            self.engine = 'pymoo'
        elif self.algorithm == 'nsga2' and self.n_obj == 2:
            self.algorithm_def = self.configure_nsga2()
            self.engine = 'pymoo'
        elif self.algorithm == 'rnsga2' and self.n_obj == 2:
            self.algorithm_def = self.configure_rnsga2()
            self.engine = 'pymoo'
        elif self.algorithm == 'age' and self.n_obj == 2:
            self.algorithm_def = self.configure_age()
            self.engine = 'pymoo'
        elif self.algorithm == 'ctaea' and self.n_obj == 3:
            self.algorithm_def = self.configure_ctaea()
            self.engine = 'pymoo'
        elif self.algorithm == 'moead' and self.n_obj == 3:
            self.algorithm_def = self.configure_moead()
            self.engine = 'pymoo'
        elif self.algorithm == 'unsga3' and self.n_obj == 3:
            self.algorithm_def = self.configure_unsga3()
            self.engine = 'pymoo'
        else:
            log.warning(f'Algorithm {self.algorithm} and/or number of objectives {self.n_obj} not supported.')
            raise NotImplementedError

        log.info(f'Configuring {self.algorithm} algorithm with {self.n_obj} objectives.')

    def configure_cmaes(
        self,
        num_evals: int = 1000,
        restarts: int = 10,
        restart_from_best: bool = True,
        warm_pop: np.ndarray = None,
    ):
        '''
        Refer to: https://pymoo.org/algorithms/soo/cmaes.html
        '''

        # CMA does not have "populations" in the genetic sense. Thus n_gens is set to number of evals.
        self.n_gens = num_evals

        if warm_pop:
            log.info('Using warm start population.')
            sample_strategy = warm_pop[0]  # TODO(macsz) This is not used - drop?
        else:
            sample_strategy = None

        self.algorithm_def = CMAES(
            restarts=10,
            restart_from_best=restart_from_best,
            maxfevals=int(self.n_gens) * restarts,
            x0=None,
        )

    def configure_ga(
        self,
        population=50,
        num_evals=1000,
        warm_pop=None,
        crossover_prob=0.9,
        crossover_eta=15.0,
        mutation_prob=0.02,
        mutation_eta=20.0,
    ):
        '''
        Simple Genetic Algorithm https://pymoo.org/algorithms/soo/ga.html
        '''

        self.n_gens = num_evals / population

        if type(warm_pop) == 'numpy.ndarray':
            log.info('Using warm start population.')
            sample_strategy = warm_pop
        else:
            sample_strategy = IntegerRandomSampling()

        self.algorithm_def = GA(
            pop_size=population,
            sampling=sample_strategy,
            crossover=SBX(prob=crossover_prob, eta=crossover_eta, vtype=float, repair=RoundingRepair()),
            mutation=PM(prob=mutation_prob, eta=mutation_eta, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

    def configure_nsga2(
        self,
        population=50,
        num_evals=1000,
        warm_pop=None,
        crossover_prob=0.9,
        crossover_eta=15.0,
        mutation_prob=0.02,
        mutation_eta=20.0,
    ):

        self.n_gens = num_evals / population

        if type(warm_pop) == 'numpy.ndarray':
            log.info('Using warm start population.')
            sample_strategy = warm_pop
        else:
            sample_strategy = IntegerRandomSampling()

        self.algorithm_def = NSGA2(
            pop_size=population,
            sampling=sample_strategy,
            crossover=SBX(prob=crossover_prob, eta=crossover_eta, vtype=float, repair=RoundingRepair()),
            mutation=PM(prob=mutation_prob, eta=mutation_eta, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

    def configure_unsga3(
        self,
        population=50,
        num_evals=1000,
        ref_dirs=None,
        warm_pop=None,
        crossover_prob=1.0,
        crossover_eta=30.0,
        mutation_prob=0.02,
        mutation_eta=20.0,
        n_partitions=20,
    ):
        self.n_gens = num_evals / population

        ref_dirs = get_reference_directions("energy", self.n_obj, n_partitions, seed=0)
        ref_dirs = ref_dirs.astype('float64')

        if type(warm_pop) == 'numpy.ndarray':
            log.info('Using warm start population')
            sample_strategy = warm_pop
        else:
            sample_strategy = IntegerRandomSampling()

        self.algorithm_def = UNSGA3(
            ref_dirs=ref_dirs,
            pop_size=population,
            sampling=sample_strategy,
            crossover=SBX(prob=crossover_prob, eta=crossover_eta, vtype=float, repair=RoundingRepair()),
            mutation=PM(prob=mutation_prob, eta=mutation_eta, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

    def configure_rnsga2(
        self,
        population=50,
        num_evals=1000,
        warm_pop=None,
        ref_points=[[0, 0]],
        crossover_prob=0.9,
        crossover_eta=15.0,
        mutation_prob=0.02,
        mutation_eta=20.0,
    ):
        self.engine = 'pymoo'
        self.n_gens = num_evals / population

        if type(warm_pop) == 'numpy.ndarray':
            log.info('Using warm start population')
            sample_strategy = warm_pop
        else:
            sample_strategy = IntegerRandomSampling()

        log.info('Reference points for RNSGA-II are: {}'.format(ref_points))
        reference_points = np.array(ref_points)  # lat=0, 1/acc=0

        self.algorithm_def = RNSGA2(
            ref_points=reference_points,
            pop_size=population,
            epsilon=0.01,
            normalization='front',
            extreme_points_as_reference_points=False,
            sampling=sample_strategy,
            crossover=SBX(prob=crossover_prob, eta=crossover_eta, vtype=float, repair=RoundingRepair()),
            mutation=PM(prob=mutation_prob, eta=mutation_eta, vtype=float, repair=RoundingRepair()),
            weights=np.array([0.5, 0.5]),
            eliminate_duplicates=True,
        )

    def configure_age(
        self,
        population=50,
        num_evals=1000,
        warm_pop=None,
        crossover_prob=0.9,
        crossover_eta=15.0,
        mutation_prob=0.02,
        mutation_eta=20.0,
    ):
        self.engine = 'pymoo'
        self.n_gens = num_evals / population

        if type(warm_pop) == 'numpy.ndarray':
            log.info('[Info] Using warm start population')
            sample_strategy = warm_pop
        else:
            sample_strategy = IntegerRandomSampling()

        self.algorithm_def = AGEMOEA(
            pop_size=population,
            sampling=sample_strategy,
            crossover=SBX(prob=crossover_prob, eta=crossover_eta, vtype=float, repair=RoundingRepair()),
            mutation=PM(prob=mutation_prob, eta=mutation_eta, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

    def configure_ctaea(self, warm_pop=None, num_evals=1000, ref_dirs=None, n_partitions=20):
        self.engine = 'pymoo'
        self.n_gens = num_evals / n_partitions

        if type(warm_pop) == 'numpy.ndarray':
            log.info('Using warm start population')
            sample_strategy = warm_pop
        else:
            sample_strategy = IntegerRandomSampling()

        ref_dirs = get_reference_directions("energy", self.n_obj, n_partitions, seed=0)
        ref_dirs = ref_dirs.astype('float64')

        self.algorithm_def = CTAEA(ref_dirs=ref_dirs, sampling=sample_strategy, eliminate_duplicates=True)

    def configure_moead(
        self,
        n_neighbors=20,
        num_evals=1000,
        warm_pop=None,
        ref_dirs=None,
        crossover_prob=1.0,
        crossover_eta=20.0,
        mutation_prob=None,
        mutation_eta=20.0,
        n_partitions=20,
    ):
        self.engine = 'pymoo'
        self.n_gens = num_evals / n_neighbors

        log.info('Configuring MOEA/D algorithm.')

        if type(warm_pop) == 'numpy.ndarray':
            log.info('Using warm start population')
            sample_strategy = warm_pop
        else:
            sample_strategy = IntegerRandomSampling()

        ref_dirs = get_reference_directions("energy", self.n_obj, n_partitions, seed=self.seed)
        ref_dirs = ref_dirs.astype('float64')

        self.algorithm_def = MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=n_neighbors,
            sampling=sample_strategy,
            crossover=SBX(prob=crossover_prob, eta=crossover_eta, vtype=float, repair=RoundingRepair()),
            mutation=PM(prob=mutation_prob, eta=mutation_eta, vtype=float, repair=RoundingRepair()),
        )

    def run_search(self, problem, save_history=False):
        '''
        Note: Known issue (memory leak) with the pymoo save history option v0.5.0
        '''

        log.info('Running Search ...')
        start_time = time.time()

        if self.engine == 'pymoo':
            result = minimize(
                problem,
                self.algorithm_def,
                ('n_gen', int(self.n_gens)),
                seed=int(self.seed),
                save_history=save_history,
                verbose=self.verbose,
            )
        else:
            log.info('Invalid algorithm engine configuration!')
            raise NotImplementedError

        log.info('Success! Search Took {:.3f} seconds.'.format(time.time() - start_time))

        return result


class EvolutionarySingleObjective(Problem):
    '''
    Interface between the user-defined evaluation interface and the SearchAlgoManager.

    Parameters
    ----------
    evaluation_interface : Class
        Class that handles the objective measurement call from the supernet.
    param_count : int
        Number variables in the search space (e.g., OFA MobileNetV3 has 45)
    param_upperbound : array
        The upper int array that defines how many options each design variable has.
    '''

    def __init__(self, evaluation_interface, param_count, param_upperbound):
        super().__init__(n_var=param_count, n_obj=1, n_constr=0, xl=0, xu=param_upperbound, type_var=int)

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


class EvolutionaryMultiObjective(Problem):
    '''
    Interface between the user-defined evaluation interface and the SearchAlgoManager.

    Parameters
    ----------
    evaluation_interface : Class
        Class that handles the objective measurement call from the supernet.
    param_count : int
        Number variables in the search space (e.g., OFA MobileNetV3 has 45)
    param_upperbound : array
        The upper int array that defines how many options each design variable has.
    '''

    def __init__(self, evaluation_interface, param_count, param_upperbound):
        super().__init__(n_var=param_count, n_obj=2, n_constr=0, xl=0, xu=param_upperbound, type_var=int)

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


class EvolutionaryManyObjective(Problem):
    '''
    Interface between the user-defined evaluation interface and the SearchAlgoManager.

    Parameters
    ----------
    evaluation_interface : Class
        Class that handles the objective measurement call from the supernet.
    param_count : int
        Number variables in the search space (e.g., OFA MobileNetV3 has 45)
    param_upperbound : array
        The upper int array that defines how many options each design variable has.
    '''

    def __init__(self, evaluation_interface, param_count, param_upperbound):
        super().__init__(n_var=param_count, n_obj=3, n_constr=0, xl=0, xu=param_upperbound, type_var=int)

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
