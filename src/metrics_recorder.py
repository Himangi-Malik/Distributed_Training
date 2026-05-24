"""Per-rank metrics collection and aggregation."""

import json
import torch
from typing import Dict, List, Optional
from pathlib import Path


class RankMetrics:
    """Collect per-step metrics for a single rank."""
    
    def __init__(self, rank: int, world_size: int, output_dir: Path = Path("benchmark_results")):
        self.rank = rank
        self.world_size = world_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Per-step metrics
        self.compute_times: List[float] = []
        self.sync_times: List[float] = []
        self.optim_times: List[float] = []
        self.iter_times: List[float] = []
        self.grad_norms: List[float] = []
        self.losses: List[float] = []
        self.bytes_transferred: List[int] = []
        self.epochs: List[int] = []
        self.steps: List[int] = []
    
    def record_step(
        self,
        epoch: int,
        step: int,
        compute_time: float,
        sync_time: float,
        optim_time: float,
        grad_norm: float,
        loss: float,
        bytes_xferred: int = 0,
    ) -> None:
        """Record metrics for a single training step."""
        self.epochs.append(epoch)
        self.steps.append(step)
        self.compute_times.append(compute_time)
        self.sync_times.append(sync_time)
        self.optim_times.append(optim_time)
        self.iter_times.append(compute_time + sync_time + optim_time)
        self.grad_norms.append(grad_norm)
        self.losses.append(loss)
        self.bytes_transferred.append(bytes_xferred)
    
    def compute_stats(self, values: List[float]) -> Dict[str, float]:
        """Compute statistics for a list of values."""
        if not values:
            return {}
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        
        mean = sum(sorted_vals) / n
        variance = sum((v - mean) ** 2 for v in sorted_vals) / n
        std = variance ** 0.5
        
        # Percentiles
        p95_idx = max(0, int(0.95 * (n - 1)))
        p99_idx = max(0, int(0.99 * (n - 1)))
        
        return {
            "count": n,
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "mean": mean,
            "std": std,
            "p95": sorted_vals[p95_idx],
            "p99": sorted_vals[p99_idx],
        }
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {
            "compute": self.compute_stats(self.compute_times),
            "sync": self.compute_stats(self.sync_times),
            "optim": self.compute_stats(self.optim_times),
            "iter": self.compute_stats(self.iter_times),
            "grad_norm": self.compute_stats(self.grad_norms),
            "loss": self.compute_stats(self.losses),
        }
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "num_steps": len(self.iter_times),
            "statistics": self.get_statistics(),
            "trajectories": {
                "compute_times": self.compute_times,
                "sync_times": self.sync_times,
                "optim_times": self.optim_times,
                "iter_times": self.iter_times,
                "grad_norms": self.grad_norms,
                "losses": self.losses,
                "bytes_transferred": self.bytes_transferred,
                "epochs": self.epochs,
                "steps": self.steps,
            }
        }
    
    def save(self, filename: Optional[str] = None) -> Path:
        """Save metrics to JSON file."""
        if filename is None:
            filename = f"metrics_rank{self.rank}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> "RankMetrics":
        """Load metrics from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        metrics = cls(data["rank"], data["world_size"])
        
        traj = data["trajectories"]
        for i in range(len(traj["iter_times"])):
            metrics.record_step(
                epoch=traj["epochs"][i],
                step=traj["steps"][i],
                compute_time=traj["compute_times"][i],
                sync_time=traj["sync_times"][i],
                optim_time=traj["optim_times"][i],
                grad_norm=traj["grad_norms"][i],
                loss=traj["losses"][i],
                bytes_xferred=traj["bytes_transferred"][i],
            )
        
        return metrics


class AggregatedMetrics:
    """Aggregate metrics across all ranks."""
    
    def __init__(self, world_size: int):
        self.world_size = world_size
        self.rank_metrics: Dict[int, RankMetrics] = {}
    
    def add_rank(self, rank_metrics: RankMetrics) -> None:
        """Add metrics from a rank."""
        self.rank_metrics[rank_metrics.rank] = rank_metrics
    
    def aggregate_stats(self) -> Dict:
        """Compute aggregated statistics across all ranks."""
        if not self.rank_metrics:
            return {}
        
        metrics_dict = {}
        
        for phase in ["compute", "sync", "optim", "iter"]:
            all_values = []
            for rank_metrics in self.rank_metrics.values():
                all_values.extend(getattr(rank_metrics, f"{phase}_times", []))
            
            if all_values:
                sorted_vals = sorted(all_values)
                n = len(sorted_vals)
                mean = sum(sorted_vals) / n
                variance = sum((v - mean) ** 2 for v in sorted_vals) / n
                std = variance ** 0.5
                p95_idx = max(0, int(0.95 * (n - 1)))
                p99_idx = max(0, int(0.99 * (n - 1)))
                
                metrics_dict[phase] = {
                    "mean": mean,
                    "std": std,
                    "min": sorted_vals[0],
                    "max": sorted_vals[-1],
                    "p95": sorted_vals[p95_idx],
                    "p99": sorted_vals[p99_idx],
                }
        
        # Throughput: samples/sec = (batch_size * world_size) / avg_iter_time
        if "iter" in metrics_dict and metrics_dict["iter"]["mean"] > 0:
            # Will be filled by caller with batch_size info
            metrics_dict["throughput_info"] = {
                "avg_iter_time_sec": metrics_dict["iter"]["mean"]
            }
        
        # Gradient norms
        all_grad_norms = []
        for rank_metrics in self.rank_metrics.values():
            all_grad_norms.extend(rank_metrics.grad_norms)
        
        if all_grad_norms:
            sorted_norms = sorted(all_grad_norms)
            n = len(sorted_norms)
            metrics_dict["grad_norm"] = {
                "mean": sum(sorted_norms) / n,
                "min": sorted_norms[0],
                "max": sorted_norms[-1],
            }
        
        # Losses
        all_losses = []
        for rank_metrics in self.rank_metrics.values():
            all_losses.extend(rank_metrics.losses)
        
        if all_losses:
            sorted_losses = sorted(all_losses)
            n = len(sorted_losses)
            metrics_dict["loss"] = {
                "mean": sum(sorted_losses) / n,
                "min": sorted_losses[0],
                "max": sorted_losses[-1],
                "trajectory": all_losses,
            }
        
        return metrics_dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "world_size": self.world_size,
            "num_ranks": len(self.rank_metrics),
            "aggregated": self.aggregate_stats(),
            "per_rank": {rank: metrics.to_dict() for rank, metrics in self.rank_metrics.items()}
        }
    
    def save(self, filepath: Path) -> Path:
        """Save aggregated metrics to JSON."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return filepath
