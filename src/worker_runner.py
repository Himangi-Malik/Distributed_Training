from gradient_sync import parameter_server, ring, tree
from models import ann_model, cnn_model, rnn_model

import torch
import time
from pathlib import Path

from metrics.metrics_recorder import RankMetrics


def get_algo_module(algo_tag: str):
    return {
        "ring": ring,
        "tree": tree,
        "parameter_server": parameter_server,
    }[algo_tag]


def get_model_module(model_tag: str):
    return {
        "ann": ann_model,
        "cnn": cnn_model,
        "rnn": rnn_model,
    }[model_tag]


def _compute_grad_norm(grad_tensor: torch.Tensor) -> float:
    """Compute L2 norm of gradient tensor."""
    if isinstance(grad_tensor, torch.Tensor):
        return torch.norm(grad_tensor).item()
    return 0.0


def run_worker(config):
    algo_module = get_algo_module(config["algo"])
    model_module = get_model_module(config["model"])

    rank = config["rank"]
    world_size = config["world_size"]

    epochs = int(config.get("epochs", 1))
    steps_per_epoch = int(config.get("steps_per_epoch", 1))

    output_dir = Path(
        config.get("benchmark_results_dir", "benchmark_results")
    )

    metrics = RankMetrics(rank, world_size, output_dir)

    print(
        f"[rank {rank}] start "
        f"algo={config['algo']} "
        f"model={config['model']} "
        f"epochs={epochs} "
        f"steps={steps_per_epoch}",
        flush=True,
    )

    comm_ctx = None

    setup_start = time.perf_counter()

    try:
        # ---------------- Setup ----------------
        comm_ctx = algo_module.setup(config)
        state = model_module.build_model(config)

        setup_time = time.perf_counter() - setup_start

        print(
            f"[rank {rank}] setup complete "
            f"({setup_time:.4f}s)",
            flush=True,
        )

        # ---------------- Training Loop ----------------
        for epoch in range(epochs):
            for step in range(steps_per_epoch):

                step_config = {
                    **config,
                    "current_epoch": epoch,
                    "step": step,
                }

                # ---------- Compute ----------
                start = time.perf_counter()

                local_grad = model_module.train_step(
                    state,
                    step_config,
                )

                compute_time = time.perf_counter() - start

                loss = local_grad.get("loss", 0.0)

                # ---------- Synchronization ----------
                start = time.perf_counter()

                synced_grad = algo_module.average(
                    local_grad,
                    comm_ctx,
                    config,
                )

                sync_time = time.perf_counter() - start

                if isinstance(synced_grad, dict):
                    grad_tensor = synced_grad["gradients"]
                else:
                    grad_tensor = synced_grad

                grad_norm = _compute_grad_norm(grad_tensor)

                grad_size_bytes = (
                    grad_tensor.numel()
                    * grad_tensor.element_size()
                )

                # ---------- Optimizer ----------
                start = time.perf_counter()

                model_module.apply_synced_gradients(
                    state,
                    grad_tensor,
                )

                optim_time = time.perf_counter() - start

                metrics.record_step(
                    epoch=epoch,
                    step=step,
                    compute_time=compute_time,
                    sync_time=sync_time,
                    optim_time=optim_time,
                    grad_norm=grad_norm,
                    loss=float(loss)
                    if isinstance(loss, torch.Tensor)
                    else loss,
                    bytes_xferred=grad_size_bytes,
                )

                print(
                    f"[rank {rank}] "
                    f"epoch={epoch} "
                    f"step={step} "
                    f"compute={compute_time:.4f}s "
                    f"sync={sync_time:.4f}s "
                    f"optim={optim_time:.4f}s "
                    f"total={(compute_time + sync_time + optim_time):.4f}s "
                    f"loss={loss:.6f} "
                    f"grad_norm={grad_norm:.6f}",
                    flush=True,
                )

    finally:
        if comm_ctx is not None:
            algo_module.teardown(comm_ctx)

        metrics_path = metrics.save()
        stats = metrics.get_statistics()

        print(
            f"[rank {rank}] teardown complete",
            flush=True,
        )

        print(
            f"[rank {rank}] metrics: {metrics_path}",
            flush=True,
        )

        print(
            f"[rank {rank}] "
            f"compute={stats['compute'].get('mean', 0):.4f}s "
            f"(p95={stats['compute'].get('p95', 0):.4f}s) | "
            f"sync={stats['sync'].get('mean', 0):.4f}s "
            f"(p95={stats['sync'].get('p95', 0):.4f}s) | "
            f"iter={stats['iter'].get('mean', 0):.4f}s "
            f"(p95={stats['iter'].get('p95', 0):.4f}s)",
            flush=True,
        )