
from dynast.search.evolutionary import (
    EvolutionaryManager,
    EvolutionaryManyObjective,
    EvolutionaryMultiObjective,
    EvolutionarySingleObjective,
)
from dynast.search.tactic.base import NASBaseConfig
from dynast.supernetwork.image_classification.bootstrapnas.bootstrapnas_encoding import BootstrapNASEncoding
from dynast.supernetwork.supernetwork_registry import *
from dynast.utils import log


class Evolutionary(NASBaseConfig):
    def __init__(
        self,
        supernet,
        optimization_metrics,
        measurements,
        num_evals,
        results_path,
        dataset_path: str = None,
        seed=42,
        population=50,
        batch_size=1,
        verbose=False,
        search_algo='nsga2',
        supernet_ckpt_path=None,
        test_fraction: float = 1.0,
        dataloader_workers: int = 4,
        device: str = 'cpu',
        **kwargs,
    ):
        super().__init__(
            dataset_path=dataset_path,
            supernet=supernet,
            optimization_metrics=optimization_metrics,
            measurements=measurements,
            num_evals=num_evals,
            results_path=results_path,
            seed=seed,
            population=population,
            batch_size=batch_size,
            verbose=verbose,
            search_algo=search_algo,
            supernet_ckpt_path=supernet_ckpt_path,
            device=device,
            test_fraction=test_fraction,
            dataloader_workers=dataloader_workers,
            **kwargs,
        )

    def search(self):
        self._init_search()

        # Following sets up the algorithm based on number of objectives
        # Could be refractored at the expense of readability
        if self.num_objectives == 1:
            problem = EvolutionarySingleObjective(
                evaluation_interface=self.validation_interface,
                param_count=self.supernet_manager.param_count,
                param_upperbound=self.supernet_manager.param_upperbound,
            )
            if self.search_algo == 'cmaes':
                search_manager = EvolutionaryManager(
                    algorithm='cmaes',
                    seed=self.seed,
                    n_obj=self.num_objectives,
                    verbose=self.verbose,
                )
                search_manager.configure_cmaes(num_evals=self.num_evals)
            else:
                search_manager = EvolutionaryManager(
                    algorithm='ga',
                    seed=self.seed,
                    n_obj=self.num_objectives,
                    verbose=self.verbose,
                )
                search_manager.configure_ga(population=self.population, num_evals=self.num_evals)
        elif self.num_objectives == 2:
            problem = EvolutionaryMultiObjective(
                evaluation_interface=self.validation_interface,
                param_count=self.supernet_manager.param_count,
                param_upperbound=self.supernet_manager.param_upperbound,
            )
            if self.search_algo == 'age':
                search_manager = EvolutionaryManager(
                    algorithm='age',
                    seed=self.seed,
                    n_obj=self.num_objectives,
                    verbose=self.verbose,
                )
                search_manager.configure_age(population=self.population, num_evals=self.num_evals)
            else:
                search_manager = EvolutionaryManager(
                    algorithm='nsga2',
                    seed=self.seed,
                    n_obj=self.num_objectives,
                    verbose=self.verbose,
                )
                search_manager.configure_nsga2(population=self.population, num_evals=self.num_evals)
        elif self.num_objectives == 3:
            problem = EvolutionaryManyObjective(
                evaluation_interface=self.validation_interface,
                param_count=self.supernet_manager.param_count,
                param_upperbound=self.supernet_manager.param_upperbound,
            )
            if self.search_algo == 'ctaea':
                search_manager = EvolutionaryManager(
                    algorithm='ctaea',
                    seed=self.seed,
                    n_obj=self.num_objectives,
                    verbose=self.verbose,
                )
                search_manager.configure_ctaea(num_evals=self.num_evals)
            elif self.search_algo == 'moead':
                search_manager = EvolutionaryManager(
                    algorithm='moead',
                    seed=self.seed,
                    n_obj=self.num_objectives,
                    verbose=self.verbose,
                )
                search_manager.configure_moead(num_evals=self.num_evals)
            else:
                search_manager = EvolutionaryManager(
                    algorithm='unsga3',
                    seed=self.seed,
                    n_obj=self.num_objectives,
                    verbose=self.verbose,
                )
                search_manager.configure_unsga3(population=self.population, num_evals=self.num_evals)
        else:
            log.error('Number of objectives not supported. Update optimization_metrics!')

        results = search_manager.run_search(problem)

        latest_population = results.pop.get('X')

        log.info("Validated model architectures in file: {}".format(self.results_path))

        output = list()
        for individual in latest_population:
            param_individual = self.supernet_manager.translate2param(individual)
            if 'bootstrapnas' in self.supernet:
                param_individual = BootstrapNASEncoding.convert_subnet_config_to_bootstrapnas(param_individual)
            output.append(param_individual)

        return output
