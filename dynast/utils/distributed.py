import os
from pathlib import PosixPath
from typing import Tuple

import torch.distributed as dist


def get_distributed_vars() -> Tuple[int, int, int, str]:
    # if torchrun was used
    local_rank = os.getenv('LOCAL_RANK')
    world_rank = os.getenv('RANK')
    world_size = os.getenv('WORLD_SIZE')
    dist_method = 'torchrun'

    # if mpi
    if not world_rank:
        local_rank = os.getenv('OMPI_COMM_WORLD_LOCAL_RANK')
        world_rank = os.getenv('OMPI_COMM_WORLD_RANK')
        world_size = os.getenv('OMPI_COMM_WORLD_SIZE')
        dist_method = 'mpi'

        if not world_rank:
            # at this point if local_rank is not set, then it's just a single process
            dist_method = None

    if dist_method is not None:
        local_rank, world_rank, world_size = int(local_rank), int(world_rank), int(world_size)
    dist_method = dist_method
    return local_rank, world_rank, world_size, dist_method


def init_processes(backend: str, world_rank: int, world_size: int) -> None:
    dist.init_process_group(backend, rank=world_rank, world_size=world_size)


def is_main_process() -> bool:
    _, world_rank, _, _ = get_distributed_vars()
    if world_rank is None:
        raise Exception('Not running in distributed mode.')
    return world_rank == 0


def is_worker_process() -> bool:
    _, world_rank, _, _ = get_distributed_vars()
    if world_rank is None:
        raise Exception('Not running in distributed mode.')
    return world_rank != 0


def get_worker_results_path(results_path: str, worker_id: int) -> str:
    path = PosixPath(results_path)
    results_path = str(path.parent / (path.stem + '_{}'.format(worker_id) + path.suffix))
    return results_path
