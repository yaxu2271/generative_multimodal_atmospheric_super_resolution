"""Weights & Biases logging utilities for training monitoring."""

import os
from typing import Optional, Dict, Any
import wandb
from . import io


_wandb_initialized = False
_wandb_run = None


def init_wandb(
    project: str = "igra_gen",
    name: Optional[str] = None,
    entity: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None,
    mode: str = "online",  # "online", "offline", or "disabled"
    save_code: bool = True,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
):
    """
    Initialize Weights & Biases logging.

    Args:
        project: W&B project name
        name: Run name (auto-generated if None)
        entity: W&B entity/username
        config: Configuration dictionary to log
        tags: List of tags for the run
        notes: Notes for the run
        mode: "online", "offline", or "disabled"
        save_code: Whether to save code to W&B
        group: Group name for organizing runs
        job_type: Job type (e.g., "train", "eval")
    """
    global _wandb_initialized, _wandb_run

    if _wandb_initialized:
        io.log0("W&B already initialized", mode="warning")
        return

    try:
        # Set environment variables for W&B
        os.environ["WANDB_MODE"] = mode

        # Initialize W&B
        _wandb_run = wandb.init(
            project=project,
            name=name,
            entity=entity,
            config=config,
            tags=tags,
            notes=notes,
            save_code=save_code,
            group=group,
            job_type=job_type,
        )

        _wandb_initialized = True
        if io.get_rank() == 0:
            io.log0(f"W&B initialized: {_wandb_run.url}")

    except Exception as e:
        io.log0(f"Failed to initialize W&B: {e}", mode="warning")
        _wandb_initialized = False


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    """
    Log metrics to W&B.

    Args:
        metrics: Dictionary of metrics to log
        step: Training step (optional)
    """
    if not _wandb_initialized or _wandb_run is None:
        return

    try:
        if step is not None:
            _wandb_run.log(metrics, step=step)
        else:
            _wandb_run.log(metrics)
    except Exception as e:
        io.log0(f"Failed to log metrics to W&B: {e}", mode="warning")


def log_model_checkpoint(path: str, name: str):
    """
    Log a model checkpoint as a W&B artifact.

    Args:
        path: Path to the checkpoint file
        name: Name for the artifact
    """
    if not _wandb_initialized or _wandb_run is None:
        return

    try:
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(path)
        _wandb_run.log_artifact(artifact)
    except Exception as e:
        io.log0(f"Failed to log checkpoint to W&B: {e}", mode="warning")


def finish_wandb():
    """Finish the W&B run."""
    global _wandb_initialized, _wandb_run

    if _wandb_initialized and _wandb_run is not None:
        try:
            _wandb_run.finish()
        except Exception as e:
            io.log0(f"Failed to finish W&B run: {e}", mode="warning")

    _wandb_initialized = False
    _wandb_run = None


def is_initialized():
    """Check if W&B is initialized."""
    return _wandb_initialized


def get_run():
    """Get the current W&B run object."""
    return _wandb_run