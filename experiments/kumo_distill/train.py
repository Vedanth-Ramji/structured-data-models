"""Train Small Kumo Tabular from a frozen Large teacher."""

import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from experiments.kumo_distill.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    write_metadata,
)
from experiments.kumo_distill.config import DistillConfig
from experiments.kumo_distill.evaluate import (
    evaluate_model,
    make_validation_episodes,
    teacher_predictions,
)
from experiments.kumo_distill.losses import (
    DistillationLoss,
    classification_distillation_loss,
    regression_distillation_loss,
)
from experiments.kumo_distill.models import build_model_pair
from experiments.kumo_distill.tasks import EpisodeBatch, RandomSCMTaskSource


def resolve_precision(
    precision: str, device: torch.device
) -> tuple[torch.dtype | None, str]:
    """Resolve automatic mixed precision without changing parameter storage."""
    if precision == "float32" or device.type == "cpu":
        return None, "float32"
    if precision == "bfloat16":
        return torch.bfloat16, precision
    if precision == "float16":
        return torch.float16, precision
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bfloat16"
    return torch.float16, "float16"


def _autocast(device: torch.device, dtype: torch.dtype | None) -> Any:
    if dtype is None:
        return nullcontext()
    return torch.autocast(device.type, dtype=dtype)


def forward_losses(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    batch: EpisodeBatch,
    *,
    task: str,
    alpha: float,
    temperature: float,
    autocast_dtype: torch.dtype | None = None,
) -> tuple[DistillationLoss, Tensor, Tensor]:
    """Run aligned raw forwards and compute the selected output objective."""
    device = batch.x.device
    teacher.eval()
    with torch.no_grad(), _autocast(device, autocast_dtype):
        teacher_output = teacher(
            x=batch.x,
            y=batch.y_context,
            categorical_mask=batch.categorical_mask,
        )
    with _autocast(device, autocast_dtype):
        student_output = student(
            x=batch.x,
            y=batch.y_context,
            categorical_mask=batch.categorical_mask,
        )
    if teacher_output.shape != student_output.shape:
        raise ValueError("teacher and student raw output shapes do not align")

    if task == "classification":
        if batch.num_classes is None:
            raise ValueError("classification batch has no class count")
        teacher_output = teacher_output[..., : batch.num_classes]
        student_output = student_output[..., : batch.num_classes]
        losses = classification_distillation_loss(
            student_output,
            teacher_output,
            batch.y_query,
            alpha=alpha,
            temperature=temperature,
        )
    else:
        losses = regression_distillation_loss(
            student_output,
            teacher_output,
            batch.y_query,
            alpha=alpha,
        )
    return losses, student_output, teacher_output


def train_step(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    batch: EpisodeBatch,
    optimizer: torch.optim.Optimizer,
    *,
    task: str,
    alpha: float,
    temperature: float = 2.0,
    max_grad_norm: float = 1.0,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Perform one optimizer step and return structured scalar metrics."""
    student.train()
    optimizer.zero_grad(set_to_none=True)
    losses, student_output, teacher_output = forward_losses(
        teacher,
        student,
        batch,
        task=task,
        alpha=alpha,
        temperature=temperature,
        autocast_dtype=autocast_dtype,
    )
    losses.total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        student.parameters(), max_grad_norm
    )
    optimizer.step()
    return _metrics(
        losses, student_output, teacher_output, batch, task, grad_norm
    )


def _metrics(
    losses: DistillationLoss,
    student_output: Tensor,
    teacher_output: Tensor,
    batch: EpisodeBatch,
    task: str,
    grad_norm: Tensor | float,
) -> dict[str, float]:
    metrics = {
        "loss": float(losses.total.detach()),
        "hard_loss": float(losses.hard.detach()),
        "soft_loss": float(losses.soft.detach()),
        "grad_norm": float(grad_norm),
    }
    if task == "classification":
        student_prob = student_output.detach().float().softmax(dim=-1)
        teacher_class = teacher_output.detach().argmax(dim=-1)
        metrics["agreement"] = float(
            (student_prob.argmax(dim=-1) == teacher_class).float().mean()
        )
        metrics["entropy"] = float(
            (
                -(student_prob * student_prob.clamp_min(1e-9).log()).sum(-1)
            ).mean()
        )
        metrics["accuracy"] = float(
            (student_prob.argmax(dim=-1) == batch.y_query).float().mean()
        )
    else:
        student_quantiles = student_output.detach().float()
        metrics["agreement"] = float(
            F.smooth_l1_loss(
                student_quantiles, teacher_output.detach().float()
            )
        )
        metrics["quantile_spread"] = float(
            (student_quantiles[..., -1] - student_quantiles[..., 0]).mean()
        )
        metrics["quantile_crossing"] = float(
            (student_quantiles[..., :-1] - student_quantiles[..., 1:])
            .clamp_min(0)
            .mean()
        )
    return metrics


def make_scheduler(
    optimizer: torch.optim.Optimizer, steps: int, warmup_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create linear warmup followed by cosine decay."""

    def scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def run(config: DistillConfig) -> None:
    """Execute a reproducible synthetic distillation run."""
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype, precision_name = resolve_precision(
        config.precision, device
    )
    pair = build_model_pair(
        config.task,
        config.teacher_size,
        config.student_size,
        device,
    )
    optimizer = torch.optim.AdamW(
        pair.student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = make_scheduler(optimizer, config.steps, config.warmup_steps)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and autocast_dtype == torch.float16,
    )
    task_generator = torch.Generator().manual_seed(config.seed)
    source = RandomSCMTaskSource()
    start_step = 0
    latest_metrics: dict[str, float] = {}
    if config.resume is not None:
        start_step, latest_metrics = load_checkpoint(
            Path(config.resume),
            student=pair.student,
            optimizer=optimizer,
            scheduler=scheduler,
            task_generator=task_generator,
            device=device,
            scaler=scaler,
        )

    config.output_path.mkdir(parents=True, exist_ok=True)
    (config.output_path / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n"
    )
    log_path = config.output_path / "metrics.jsonl"
    episodes_seen = int(latest_metrics.get("episodes_seen", 0))
    query_examples_seen = int(latest_metrics.get("query_examples_seen", 0))
    interval_start = time.perf_counter()
    interval_examples = 0
    validation_episodes = make_validation_episodes(
        task=config.task,
        count=config.validation_episodes,
        seed=config.validation_seed,
        batch_size=config.batch_size,
        context_rows=config.context_rows[0],
        query_rows=config.query_rows[0],
        num_features=config.num_features[0],
        num_classes=config.max_classes,
    )
    validation_teacher = teacher_predictions(
        pair.teacher, validation_episodes, device, autocast_dtype
    )

    for step_index in range(start_step, config.steps):
        pair.student.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, float] = {}
        for _ in range(config.gradient_accumulation):
            shape = config.sample_shape(task_generator)
            batch = source.sample(shape, task_generator).to(device)
            losses, student_output, teacher_output = forward_losses(
                pair.teacher,
                pair.student,
                batch,
                task=config.task,
                alpha=config.resolved_alpha,
                temperature=config.temperature,
                autocast_dtype=autocast_dtype,
            )
            scaler.scale(
                losses.total / config.gradient_accumulation
            ).backward()
            current = _metrics(
                losses,
                student_output,
                teacher_output,
                batch,
                config.task,
                0.0,
            )
            for name, value in current.items():
                accumulated[name] = accumulated.get(name, 0.0) + (
                    value / config.gradient_accumulation
                )
            count = batch.batch_size * batch.query_rows
            episodes_seen += batch.batch_size
            query_examples_seen += count
            interval_examples += count

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            pair.student.parameters(), config.max_grad_norm
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        completed_step = step_index + 1
        accumulated["grad_norm"] = float(grad_norm)
        accumulated["learning_rate"] = scheduler.get_last_lr()[0]
        latest_metrics = accumulated

        if completed_step % config.validation_interval == 0:
            validation = evaluate_model(
                pair.student,
                validation_episodes,
                validation_teacher,
                task=config.task,
                device=device,
                autocast_dtype=autocast_dtype,
            )
            latest_metrics.update(
                {
                    f"validation_{name}": value
                    for name, value in validation.items()
                }
            )

        if completed_step % config.log_interval == 0 or completed_step == 1:
            elapsed = max(time.perf_counter() - interval_start, 1e-9)
            latest_metrics["examples_per_second"] = interval_examples / elapsed
            latest_metrics["step"] = float(completed_step)
            if device.type == "cuda":
                latest_metrics["gpu_memory_allocated"] = float(
                    torch.cuda.memory_allocated(device)
                )
                latest_metrics["gpu_memory_reserved"] = float(
                    torch.cuda.memory_reserved(device)
                )
            with log_path.open("a") as stream:
                stream.write(json.dumps(latest_metrics) + "\n")
            print(json.dumps(latest_metrics), flush=True)
            interval_start = time.perf_counter()
            interval_examples = 0

        if (
            completed_step % config.checkpoint_interval == 0
            or completed_step == config.steps
        ):
            latest_metrics["episodes_seen"] = float(episodes_seen)
            latest_metrics["query_examples_seen"] = float(query_examples_seen)
            save_checkpoint(
                config.output_path / f"checkpoint-{completed_step}.pt",
                student=pair.student,
                optimizer=optimizer,
                scheduler=scheduler,
                step=completed_step,
                config=config,
                task_generator=task_generator,
                metrics=latest_metrics,
                scaler=scaler,
            )
            save_checkpoint(
                config.output_path / "latest.pt",
                student=pair.student,
                optimizer=optimizer,
                scheduler=scheduler,
                step=completed_step,
                config=config,
                task_generator=task_generator,
                metrics=latest_metrics,
                scaler=scaler,
            )

    write_metadata(
        config.output_path / "metadata.json",
        config,
        device=device,
        precision=precision_name,
        episodes_seen=episodes_seen,
        query_examples_seen=query_examples_seen,
    )


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def parse_args() -> DistillConfig:
    """Parse the command-line interface into a validated configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("classification", "regression"), required=True
    )
    parser.add_argument("--teacher-size", default="large")
    parser.add_argument("--student-size", default="small")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument(
        "--context-rows", type=_parse_int_tuple, default=(64, 128, 256)
    )
    parser.add_argument(
        "--query-rows", type=_parse_int_tuple, default=(32, 64, 128)
    )
    parser.add_argument(
        "--num-features", type=_parse_int_tuple, default=(4, 8, 16, 32)
    )
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--validation-episodes", type=int, default=4)
    parser.add_argument("--validation-seed", type=int, default=10_000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--fixed-shape", action="store_true")
    return DistillConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
