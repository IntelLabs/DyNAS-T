import pytest

from dynast.dynast_manager import DyNAS
from dynast.search.search_tactic import LINAS, Evolutionary, LINASDistributed, RandomSearch, RandomSearchDistributed


def test_dynas_not_enough_args_passed_exits():
    # Not all params passed
    with pytest.raises(SystemExit):
        DyNAS(supernet='ofa_resnet50')


def test_dynas_optimization_metrics_unsupported_number():
    with pytest.raises(SystemExit):
        DyNAS(
            supernet='ofa_resnet50',
            optimization_metrics=['m1', 'm2', 'm3', 'm4'],
            measurements=['m1', 'm2', 'm3'],
            search_tactic='linas',
            num_evals=1,
            results_path='test',
            dataset_path='test',
        )


def test_dynas_unsupported_search_tactic_exception():
    with pytest.raises(NotImplementedError):
        DyNAS(
            supernet='ofa_resnet50',
            optimization_metrics=['m1', 'm2', 'm3'],
            measurements=['m1', 'm2', 'm3'],
            search_tactic='new_and_unsupported_search_tactic',
            num_evals=1,
            results_path='test',
            dataset_path='test',
        )


def test_dynas_supported_search_tactics():
    assert isinstance(
        DyNAS(
            supernet='ofa_resnet50',
            optimization_metrics=['m1', 'm2', 'm3'],
            measurements=['m1', 'm2', 'm3'],
            search_tactic='linas',
            num_evals=1,
            results_path='test',
            dataset_path='test',
        ),
        LINAS,
    )

    assert isinstance(
        DyNAS(
            supernet='ofa_resnet50',
            optimization_metrics=['m1', 'm2', 'm3'],
            measurements=['m1', 'm2', 'm3'],
            search_tactic='evolutionary',
            num_evals=1,
            results_path='test',
            dataset_path='test',
        ),
        Evolutionary,
    )

    assert isinstance(
        DyNAS(
            supernet='ofa_resnet50',
            optimization_metrics=['m1', 'm2', 'm3'],
            measurements=['m1', 'm2', 'm3'],
            search_tactic='random',
            num_evals=1,
            results_path='test',
            dataset_path='test',
        ),
        RandomSearch,
    )

    assert isinstance(
        DyNAS(
            supernet='ofa_resnet50',
            optimization_metrics=['m1', 'm2', 'm3'],
            measurements=['m1', 'm2', 'm3'],
            search_tactic='linas_dist',
            num_evals=1,
            results_path='test',
            dataset_path='test',
        ),
        LINASDistributed,
    )

    assert isinstance(
        DyNAS(
            supernet='ofa_resnet50',
            optimization_metrics=['m1', 'm2', 'm3'],
            measurements=['m1', 'm2', 'm3'],
            search_tactic='random_dist',
            num_evals=1,
            results_path='test',
            dataset_path='test',
        ),
        RandomSearchDistributed,
    )


def test_dynas_unsupported_supernet():
    with pytest.raises(Exception):
        DyNAS(
            supernet='unsupported_supernet',
            optimization_metrics=['m1', 'm2', 'm3'],
            measurements=['m1', 'm2', 'm3'],
            search_tactic='linas',
            num_evals=1,
            results_path='test',
            dataset_path='test',
        )
