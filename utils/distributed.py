import torch
from torch import distributed as dist
from tqdm import tqdm


def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def d_print(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


def d_tqdm(rank, *args, **kwargs):
    if rank == 0:
        return tqdm(*args, **kwargs)
    else:
        return range(*args)
