from statistics import mean, median, stdev
from pathlib import Path
import json
import numpy as np

from .rank_metrics import RankMetrics

class SummaryGenerator:
    """
    Computes experiment-level statistics from one or more RankMetrics objects.
    """

    def __init__(
        self,
        rank_metrics: list[RankMetrics],
    ) -> None:

        if not rank_metrics:
            raise ValueError(
                "SummaryGenerator requires at least one RankMetrics object."
            )

        self.rank_metrics = rank_metrics

    def _recorded_steps(
        self,
        rank: RankMetrics,
    ) -> list:

        return [
            step
            for step in rank.steps
            if not step.is_warmup
        ]

    def _collect(
        self,
        attribute: str,
    ) -> list[float]:

        """
        Collect one attribute from every recorded step.
        """

        values = []

        for rank in self.rank_metrics:
            for step in self._recorded_steps(rank):
                values.append(
                    getattr(
                        step,
                        attribute,
                    )
                )

        return values

    def _collect_iteration_maxima(self) -> list[float]:
        """
        Collect per-step distributed iteration maxima across ranks.
        """

        per_step_values: dict[tuple[int, int], list[float]] = {}

        for rank in self.rank_metrics:
            for step in self._recorded_steps(rank):
                key = (step.epoch, step.step)
                per_step_values.setdefault(key, []).append(step.iteration_time)

        expected_ranks = len(self.rank_metrics)

        return [
            max(values)
            for _, values in sorted(per_step_values.items())
            if len(values) == expected_ranks
        ]
    
    def _compute_statistics(
        self,
        values: list[float],
    ) -> dict:

        """
        Compute descriptive statistics for a list of values.
        """

        if not values:
            return {}

        values = sorted(values)

        return {
            "mean": mean(values),
            "median": median(values),
            "std": (
                stdev(values)
                if len(values) > 1
                else 0.0
            ),
            "min": values[0],
            "max": values[-1],
            "p95": float(np.percentile(values, 95)),
        }
    
    def _compute_totals(
        self,
        values: list[float],
    ) -> dict:

        """
        Compute totals for cumulative metrics.
        """

        if not values:
            return {}

        return {
            "total": sum(values),
            "mean": mean(values),
        }
    def compute(self) -> dict:
        """
        Compute experiment-level statistics across all ranks.
        """

        # --------------------------------------------------
        # Timing Metrics
        # --------------------------------------------------

        compute = self._compute_statistics(
            self._collect("compute_time")
        )

        sync = self._compute_statistics(
            self._collect("sync_time")
        )

        optimizer = self._compute_statistics(
            self._collect("optim_time")
        )

        iteration = self._compute_statistics(
            self._collect_iteration_maxima()
        )

        # --------------------------------------------------
        # Communication Metrics
        # --------------------------------------------------

        bytes_sent = self._compute_totals(
            self._collect("bytes_sent")
        )

        bytes_received = self._compute_totals(
            self._collect("bytes_received")
        )

        # --------------------------------------------------
        # Model Metrics
        # --------------------------------------------------
        reference_rank = self.rank_metrics[0]

        losses = [
            step.loss
            for step in self._recorded_steps(reference_rank)
        ]

        loss = (
            {
                "initial": losses[0],
                "final": losses[-1],
                "mean": mean(losses),
            }
            if losses
            else {}
        )

        grad_norm = self._compute_statistics(
            self._collect("grad_norm")
        )

        # --------------------------------------------------
        # Derived Metrics
        # --------------------------------------------------

        mean_iteration = iteration.get("mean", 0.0)
        mean_compute = compute.get("mean", 0.0)
        mean_sync = sync.get("mean", 0.0)
        mean_optimizer = optimizer.get("mean", 0.0)

        if mean_iteration:
            fractions = {
                "compute_fraction": mean_compute / mean_iteration,
                "sync_fraction": mean_sync / mean_iteration,
                "optimizer_fraction": mean_optimizer / mean_iteration,
            }
        else:
            fractions = {
                "compute_fraction": 0.0,
                "sync_fraction": 0.0,
                "optimizer_fraction": 0.0,
            }

        # --------------------------------------------------
        # Final Summary
        # --------------------------------------------------

        return {
            "timing": {
                "compute": compute,
                "sync": sync,
                "optimizer": optimizer,
                "iteration": iteration,
            },
            "communication": {
                "bytes_sent": bytes_sent,
                "bytes_received": bytes_received,
            },
            "model": {
                "loss": loss,
                "grad_norm": grad_norm,
            },
            "derived": fractions,
        }
    def save_summary(
        self,
        output_path: Path,
    ) -> Path:
        """
        Save the experiment summary to disk.
        """

        output_path = Path(output_path)

        summary = self.compute()

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file_handle:

            json.dump(
                summary,
                file_handle,
                indent=4,
            )

        return output_path 