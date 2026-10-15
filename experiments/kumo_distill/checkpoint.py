"""Deterministic checkpoint and experiment metadata helpers."""

import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from experiments.kumo_distill.config import DistillConfig


def capture_rng(task_generator: torch.Generator) -> dict[str, Any]:
    """Capture global and task-stream random number states."""
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
        "task": task_generator.get_state(),
    }


def restore_rng(
    state: dict[str, Any], task_generator: torch.Generator
) -> None:
    """Restore global and task-stream random number states."""
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    task_generator.set_state(state["task"])


def save_checkpoint(
    path: Path,
    *,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    config: DistillConfig,
    task_generator: torch.Generator,
    metrics: dict[str, float],
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """Atomically save all state needed for exact continuation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "student": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "config": asdict(config),
        "rng": capture_rng(task_generator),
        "metrics": metrics,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    *,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    task_generator: torch.Generator,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, dict[str, float]]:
    """Restore training and RNG state, returning the completed step count."""
    payload = torch.load(path, map_location=device, weights_only=True)
    student.load_state_dict(payload["student"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    restore_rng(payload["rng"], task_generator)
    return int(payload["step"]), dict(payload["metrics"])


def write_metadata(
    path: Path,
    config: DistillConfig,
    *,
    device: torch.device,
    precision: str,
    episodes_seen: int,
    query_examples_seen: int,
) -> None:
    """Write human-readable provenance alongside binary checkpoints."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    hardware = platform.processor() or platform.machine()
    if device.type == "cuda":
        hardware = torch.cuda.get_device_name(device)
    metadata = {
        "git_commit": commit,
        "teacher_checkpoint_revision": config.revision,
        "student_checkpoint_revision": config.revision,
        "task_generator": "RandomSCMTaskSource",
        "task_generator_config": {
            "context_rows": config.context_rows,
            "query_rows": config.query_rows,
            "num_features": config.num_features,
            "min_classes": config.min_classes,
            "max_classes": config.max_classes,
            "noise_std": config.noise_std,
        },
        "seed": config.seed,
        "episodes_seen": episodes_seen,
        "query_examples_seen": query_examples_seen,
        "hardware": hardware,
        "device": str(device),
        "precision": precision,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n")
