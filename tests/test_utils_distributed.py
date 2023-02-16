import os
from unittest import mock

import pytest

from dynast.utils.distributed import get_distributed_vars, get_worker_results_path, is_main_process, is_worker_process


def test_get_worker_results_path() -> None:
    results_path = 'dir/test.csv'
    worker_id = 1
    assert 'dir/test_1.csv' == get_worker_results_path(results_path, worker_id)

    results_path = 'test.csv'
    worker_id = 2
    assert 'test_2.csv' == get_worker_results_path(results_path, worker_id)


def test_get_distributed_vars() -> None:
    with mock.patch.dict(os.environ, {"RANK": "2", "LOCAL_RANK": "1", "WORLD_SIZE": "2"}):
        local_rank, world_rank, world_size, dist_method = get_distributed_vars()
        assert local_rank == 1 and world_rank == 2 and world_size == 2 and dist_method == 'torchrun'

    with mock.patch.dict(
        os.environ, {"OMPI_COMM_WORLD_RANK": "2", "OMPI_COMM_WORLD_LOCAL_RANK": "1", "OMPI_COMM_WORLD_SIZE": "2"}
    ):
        local_rank, world_rank, world_size, dist_method = get_distributed_vars()
        assert local_rank == 1 and world_rank == 2 and world_size == 2 and dist_method == 'mpi'


def test_is_main_process() -> None:
    with mock.patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"}):
        assert is_main_process()

    with mock.patch.dict(os.environ, {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2"}):
        assert not is_main_process()

    with mock.patch.dict(
        os.environ, {"OMPI_COMM_WORLD_RANK": "0", "OMPI_COMM_WORLD_LOCAL_RANK": "0", "OMPI_COMM_WORLD_SIZE": "2"}
    ):
        assert is_main_process()

    with mock.patch.dict(
        os.environ, {"OMPI_COMM_WORLD_RANK": "1", "OMPI_COMM_WORLD_LOCAL_RANK": "1", "OMPI_COMM_WORLD_SIZE": "2"}
    ):
        assert not is_main_process()

    with pytest.raises(Exception):
        is_main_process()


def test_is_worker_process() -> None:
    with mock.patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"}):
        assert not is_worker_process()

    with mock.patch.dict(os.environ, {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2"}):
        assert is_worker_process()

    with mock.patch.dict(
        os.environ, {"OMPI_COMM_WORLD_RANK": "0", "OMPI_COMM_WORLD_LOCAL_RANK": "0", "OMPI_COMM_WORLD_SIZE": "2"}
    ):
        assert not is_worker_process()

    with mock.patch.dict(
        os.environ, {"OMPI_COMM_WORLD_RANK": "1", "OMPI_COMM_WORLD_LOCAL_RANK": "1", "OMPI_COMM_WORLD_SIZE": "2"}
    ):
        assert is_worker_process()

    with pytest.raises(Exception):
        is_worker_process()
