from gradient_sync import parameter_server, ring, tree
from models import ann_model, cnn_model, rnn_model
import torch
import time
from pathlib import Path
from timing_context import TimingTracker, timer
from metrics_recorder import RankMetrics


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


def _summarize_grad(grad_obj: dict) -> str:
    grad_tensor = grad_obj.get("gradients")
    if isinstance(grad_tensor, torch.Tensor):
        flat = grad_tensor.detach().flatten()
        sample = flat[: min(4, flat.numel())].tolist()
        return f"shape={tuple(grad_tensor.shape)} dtype={grad_tensor.dtype} sample={sample}"
    return f"type={type(grad_tensor).__name__}"


def _compute_grad_norm(grad_tensor: torch.Tensor) -> float:
    """Compute L2 norm of gradient tensor."""
    if isinstance(grad_tensor, torch.Tensor):
        return torch.norm(grad_tensor).item()
    return 0.0


def run_worker(config):
    algo_module = get_algo_module(config["algo"])
    model_module = get_model_module(config["model"])
    epochs = int(config.get("epochs", 1))
    steps_per_epoch = int(config.get("steps_per_epoch", 1))
    rank = config["rank"]
    world_size = config["world_size"]
    
    # Initialize metrics recording
    output_dir = Path(config.get("benchmark_results_dir", "benchmark_results"))
    metrics = RankMetrics(rank, world_size, output_dir)

    print(f"[rank {rank}] start mode={config['mode']} algo={config['algo']} epochs={epochs} steps_per_epoch={steps_per_epoch}", flush=True)

    comm_ctx = None
    
    # Track setup time (excluded from measurements)
    setup_start = time.perf_counter()

    try:
        # Step 1: communication setup first.
        comm_ctx = algo_module.setup(config)

        # Step 2: build model.
        try:
            model = model_module.build_model(config)
        except NotImplementedError as error:
            print(f"[rank {rank}] warning: build_model placeholder hit: {error}", flush=True)
            model = {"model_tag": config["model"], "placeholder": True}

        setup_time = time.perf_counter() - setup_start
        print(f"[rank {rank}] SETUP_COMPLETE setup_time={setup_time:.4f}s", flush=True)

        # Step 3: run training loop for specified number of epochs.
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                # Pass current epoch and step to model so synthetic data can be rank/epoch/step-specific
                epoch_config = {**config, "current_epoch": epoch, "step": step}
                
                # COMPUTE: Forward + Backward (train_step)
                compute_start = time.perf_counter()
                try:
                    local_grad = model_module.train_step(model, epoch_config)
                except NotImplementedError as error:
                    print(f"[rank {rank}] epoch {epoch} step {step} warning: train_step placeholder hit: {error}", flush=True)
                    local_grad = {"rank": rank, "gradients": torch.tensor([0.0], dtype=torch.float32)}
                compute_time = time.perf_counter() - compute_start
                
                # Extract loss from local gradients (not synchronized)
                loss = local_grad.get("loss", 0.0)

                # SYNC: Gradient Synchronization (algo_module.average)
                sync_start = time.perf_counter()
                try:
                    synced_grad = algo_module.average(local_grad, comm_ctx, config)
                except NotImplementedError as error:
                    print(f"[rank {rank}] epoch {epoch} step {step} warning: average placeholder hit: {error}", flush=True)
                    synced_grad = local_grad
                sync_time = time.perf_counter() - sync_start
                
                # Extract gradient info from synchronized gradients (all ranks should have identical values)
                grad_tensor = synced_grad.get("gradients", torch.tensor([0.0]))
                grad_norm = _compute_grad_norm(grad_tensor)
                grad_size_bytes = grad_tensor.numel() * 4  # float32

                # OPTIM: Apply Synchronized Gradients (optimizer step)
                optim_start = time.perf_counter()
                if hasattr(model_module, "apply_synced_gradients"):
                    try:
                        averaged = synced_grad.get("gradients") if isinstance(synced_grad, dict) else synced_grad
                        model_module.apply_synced_gradients(model, averaged)
                    except Exception as e:
                        print(f"[rank {rank}] epoch {epoch} step {step} apply_synced_gradients failed: {e}", flush=True)
                optim_time = time.perf_counter() - optim_start

                # Record metrics for this step
                metrics.record_step(
                    epoch=epoch,
                    step=step,
                    compute_time=compute_time,
                    sync_time=sync_time,
                    optim_time=optim_time,
                    grad_norm=grad_norm,
                    loss=float(loss) if isinstance(loss, torch.Tensor) else loss,
                    bytes_xferred=grad_size_bytes,
                )

                # Log output
                print(f"[rank {rank}] epoch {epoch} step {step} compute={compute_time:.4f}s sync={sync_time:.4f}s optim={optim_time:.4f}s total={(compute_time+sync_time+optim_time):.4f}s grad_norm={grad_norm:.6f} loss={loss:.6f}", flush=True)
    finally:
        # Step 4: teardown last.
        algo_module.teardown(comm_ctx)
        print(f"[rank {rank}] teardown done", flush=True)
        
        # Step 5: Save metrics to file
        metrics_path = metrics.save()
        stats = metrics.get_statistics()
        print(f"[rank {rank}] metrics saved to {metrics_path}", flush=True)
        print(f"[rank {rank}] compute: mean={stats['compute'].get('mean', 0):.4f}s p95={stats['compute'].get('p95', 0):.4f}s", flush=True)
        print(f"[rank {rank}] sync: mean={stats['sync'].get('mean', 0):.4f}s p95={stats['sync'].get('p95', 0):.4f}s", flush=True)
        print(f"[rank {rank}] iter: mean={stats['iter'].get('mean', 0):.4f}s p95={stats['iter'].get('p95', 0):.4f}s", flush=True)
