from dynast.utils.distributed import WORLD_RANK, get_worker_results_path


def test_get_worker_results_path() -> None:
    results_path = 'dir/test.csv'
    worker_id = 1
    assert 'dir/test_1.csv' == get_worker_results_path(results_path, worker_id)

    results_path = 'test.csv'
    worker_id = 2
    assert 'test_2.csv' == get_worker_results_path(results_path, worker_id)
