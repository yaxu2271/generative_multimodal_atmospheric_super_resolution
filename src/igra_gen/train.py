import os

import warnings
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Sampler
from torchinfo import summary

from igra_gen.data.samplers import InfiniteSampler
from igra_gen.training.trainer import Trainer
from igra_gen.utils import io, stats

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    # Initialize the distributed process group only if not already initialized.
    torch.multiprocessing.set_start_method("spawn")
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    stats.init_multiprocessing(
        rank=io.get_rank(),
        sync_device=torch.device('cuda') if io.get_world_size() > 1 else None,
    )
    io.log0(f"Rank {io.get_rank()}, World Size {io.get_world_size()}")
    if io.get_rank() == 0:
        io.log0(OmegaConf.to_yaml(cfg))

    np.random.seed((cfg.seed * (torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1) + (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0)) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.system.torch.benchmark
        torch.backends.cudnn.allow_tf32 = cfg.system.torch.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = cfg.system.torch.allow_tf32
        torch.set_float32_matmul_precision(
            cfg.system.torch.set_float32_matmul_precision
        )
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            cfg.system.torch.amp_type == "float16"
        )

    io.log0("Loading dataset...")
    dataset: Dataset = instantiate(cfg.data.dataset, _convert_="object")
    dataset_sampler: Sampler = InfiniteSampler(
        dataset=dataset,
        rank=io.get_rank(),
        num_replicas=io.get_world_size(),
        shuffle=True,
        seed=cfg.seed,
    )
    dataloader = iter(
        DataLoader(
            dataset=dataset,
            sampler=dataset_sampler,
            batch_size=cfg.data.batch_size,
            pin_memory=True,
            num_workers=cfg.data.data_workers,
            prefetch_factor=(2 if cfg.data.data_workers > 0 else None),
            persistent_workers=False,
        )
    )

    io.log0("Constructing network...")
    if cfg.get("precond") is not None:
        net: torch.nn.Module = instantiate(
            cfg.precond,
            model=cfg.model,
            img_resolution=dataset.img_resolution,
            img_channels=dataset.n_channels,
            condition_channels=dataset.cond_channels,
            _recursive_=False,
            _convert_="object",
        )
    else:
        net: torch.nn.Module = instantiate(
            cfg.model,
            img_resolution=dataset.img_resolution,
            in_channels=dataset.n_channels,
            out_channels=dataset.out_channels,
            _convert_="object",
        )
    net.train().requires_grad_(True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if io.get_rank() == 0:
        summary(net, depth=3)

    io.log0("Constructing optimizer...")
    optimizer = instantiate(cfg.optimizer, net.parameters(), _convert_="object")
    io.log0("Constructing loss function...")
    loss_fn = instantiate(cfg.loss, _convert_="object")
    trainer: Trainer = instantiate(
        cfg.trainer,
        net=net,
        optimizer=optimizer,
        loss_fn=loss_fn,
        amp_type=cfg.system.torch.amp_type,
    )

    io.log0("Training...")
    trainer.train(dataloader)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    """
    local (1 gpu 1 node): python train.py
    """
    main()
