"""Generate publication-ready tables from benchmark results."""

from pathlib import Path
from collections import defaultdict
from statistics import mean
import pandas as pd

from .experiment_results import (
    ExperimentResult,
    load_experiment_results,
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "paper_results" / "tables"

def _mean(values):
    return round(mean(values), 2)

def _timing(experiment: ExperimentResult, phase: str) -> float:
    """Return the mean timing for a phase in milliseconds."""
    return experiment.summary["timing"][phase]["mean"] * 1000

def _format_algorithm(name: str) -> str:
    """Format algorithm names for paper tables."""
    return name.replace("_", " ").title()

def _group_by_algorithm(results: list[ExperimentResult]):
    grouped = defaultdict(list)

    for result in results:
        grouped[result.config["algo"]].append(result)

    return grouped

def _format_values(values) -> str:
    """Format a collection of values for display in a paper table."""
    return ", ".join(str(value) for value in sorted(values))


def table_1_experimental_configuration(
    results: list[ExperimentResult],
) -> pd.DataFrame:
    """
    Generate Table 1: Experimental Configuration.
    """

    if not results:
        return pd.DataFrame(columns=["Parameter", "Value"])

    first_config = results[0].config

    algorithms = {
        result.config["algo"]
        for result in results
    }

    models = {
        result.config["model"]
        for result in results
    }

    world_sizes = {
        result.config["world_size"]
        for result in results
    }

    batch_sizes = {
        result.config["batch_size"]
        for result in results
    }

    rows = [
        ("Dataset", first_config["dataset"]),
        ("Models", _format_values(models)),
        ("Communication Algorithms", _format_values(algorithms)),
        ("World Sizes", _format_values(world_sizes)),
        ("Batch Sizes", _format_values(batch_sizes)),
        ("Epochs", first_config["epochs"]),
        ("Steps per Epoch", first_config["steps_per_epoch"]),
        ("Optimizer", "SGD"),
        ("Learning Rate", first_config["lr"]),
        ("Total Configurations", len(results)),
    ]

    return pd.DataFrame(rows, columns=["Parameter", "Value"])


def table_2_timing_breakdown(
    results: list[ExperimentResult],
) -> pd.DataFrame:
    """
    Generate Table 2: Timing Breakdown.
    """

    rows = []
    grouped = _group_by_algorithm(results)

    for algorithm in sorted(grouped):

        experiments = grouped[algorithm]

        rows.append(
            {
                "Algorithm": _format_algorithm(algorithm),
                "Compute (ms)": _mean(
                    _timing(experiment, "compute")
                    for experiment in experiments
                ),
                "Sync (ms)": _mean(
                    _timing(experiment, "sync")
                    for experiment in experiments
                ),
                "Optimizer (ms)": _mean(
                    _timing(experiment, "optimizer")
                    for experiment in experiments
                ),
                "Iteration (ms)": _mean(
                    _timing(experiment, "iteration")
                    for experiment in experiments
                ),
            }
        )

    return pd.DataFrame(rows)


def table_3_communication_stability(
    results: list[ExperimentResult],
) -> pd.DataFrame:
    """
    Generate Table 4: Communication Stability.
    """

    rows = []
    grouped = _group_by_algorithm(results)

    for algorithm in sorted(grouped):

        experiments = grouped[algorithm]

        rows.append(
            {
                "Algorithm": _format_algorithm(algorithm),
                "Mean Sync (ms)": _mean(
                    _timing(experiment, "sync")
                    for experiment in experiments
                ),
                "Median Sync (ms)": _mean(
                    experiment.summary["timing"]["sync"]["median"] * 1000
                    for experiment in experiments
                ),
                "Std Sync (ms)": _mean(
                    experiment.summary["timing"]["sync"]["std"] * 1000
                    for experiment in experiments
                ),
                "P95 Sync (ms)": _mean(
                    experiment.summary["timing"]["sync"]["p95"] * 1000
                    for experiment in experiments
                ),
            }
        )

    return pd.DataFrame(rows)

def table_4_training_validation(
    results: list[ExperimentResult],
) -> pd.DataFrame:
    """
    Generate Table 4: Training Validation.
    """

    rows = []
    grouped = _group_by_algorithm(results)

    for algorithm in sorted(grouped):

        experiments = grouped[algorithm]

        rows.append(
            {
                "Algorithm": _format_algorithm(algorithm),
                "Initial Loss": _mean(
                    experiment.summary["model"]["loss"]["initial"]
                    for experiment in experiments
                ),
                "Final Loss": _mean(
                    experiment.summary["model"]["loss"]["final"]
                    for experiment in experiments
                ),
                "Mean Grad Norm": _mean(
                    experiment.summary["model"]["grad_norm"]["mean"]
                    for experiment in experiments
                ),
            }
        )

    return pd.DataFrame(rows)

def generate_all_tables():
    results = load_experiment_results()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table_1_experimental_configuration(results).to_csv(
        OUTPUT_DIR / "table_1_experimental_configuration.csv",
        index=False,
    )

    table_2_timing_breakdown(results).to_csv(
        OUTPUT_DIR / "table_2_timing_breakdown.csv",
        index=False,
    )

    table_3_communication_stability(results).to_csv(
        OUTPUT_DIR / "table_3_communication_stability.csv",
        index=False,
    )

    table_4_training_validation(results).to_csv(
        OUTPUT_DIR / "table_4_training_validation.csv",
        index=False,
    )
    