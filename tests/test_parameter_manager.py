from dynast.manager import ParameterManager

SUPERNET_PARAMETERS = {
    "d": {"count": 5, "vars": [0, 1]},
    "e": {"count": 12, "vars": [0.2, 0.25]},
    "w": {"count": 6, "vars": [0, 1, 2]},
}
TINY_SUPERNET_PARAMETERS = {
    "d": {"count": 1, "vars": [0, 1]},
    "e": {"count": 1, "vars": [0.2, 0.25]},
    "w": {"count": 1, "vars": [0, 1]},
}
SEED = 42


def test_random_sample():
    supernet_manager = ParameterManager(
        param_dict=SUPERNET_PARAMETERS,
        seed=SEED,
    )

    assert supernet_manager.random_sample() == [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 1, 0, 0]


def test_random_samples_size():
    supernet_manager = ParameterManager(
        param_dict=SUPERNET_PARAMETERS,
        seed=SEED,
    )

    result_confs = supernet_manager.random_samples(size=3)
    expected_confs = [
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 1, 0, 0],
        [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 2, 1, 2, 2, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 2, 2, 2, 0, 2, 2],
    ]

    assert len(expected_confs) == len(result_confs)
    assert expected_confs == result_confs


def test_random_samples_trials():
    supernet_manager = ParameterManager(
        param_dict=TINY_SUPERNET_PARAMETERS,
        seed=SEED,
    )

    result_confs = supernet_manager.random_samples(size=10, trial_limit=20)
    assert len(result_confs) == 8

    result_confs = supernet_manager.random_samples(size=10, trial_limit=1)
    assert len(result_confs) == 1


def test_translate2param():
    supernet_manager = ParameterManager(
        param_dict=TINY_SUPERNET_PARAMETERS,
        seed=SEED,
    )

    params = {
        "d": [0],  # 0
        "e": [0.25],  # 1
        "w": [1],  # 1
    }
    pymoo_vector = [0, 1, 1]

    assert params == supernet_manager.translate2param(pymoo_vector)


def test_translate2pymoo():
    supernet_manager = ParameterManager(
        param_dict=TINY_SUPERNET_PARAMETERS,
        seed=SEED,
    )

    params = {
        "d": [0],  # 0
        "e": [0.25],  # 1
        "w": [1],  # 1
    }
    pymoo_vector = [0, 1, 1]

    assert pymoo_vector == supernet_manager.translate2pymoo(params)
