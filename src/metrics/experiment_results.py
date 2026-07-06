"""Load completed benchmark experiment results to pass onto tables and plots scripts."""

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(slots=True)
class ExperimentResult:
    """Represents one completed benchmark experiment."""

    config: dict
    summary: dict


def load_experiment_results(results_dir: Path = Path("benchmark_results")) -> list[ExperimentResult]:
    """
    Load all completed benchmark experiments.

    Each experiment directory must contain:
        - config.json
        - summary.json

    Returns:
        A list of ExperimentResult objects.
    """

    experiments: list[ExperimentResult] = []

    if not results_dir.exists():
        return experiments

    for experiment_dir in sorted(results_dir.glob("experiment_*")):
        config_file = experiment_dir / "config.json"
        summary_file = experiment_dir / "summary.json"

        if not config_file.exists() or not summary_file.exists():
            continue

        with config_file.open("r") as f:
            config = json.load(f)

        with summary_file.open("r") as f:
            summary = json.load(f)

        experiments.append(
            ExperimentResult(
                config=config,
                summary=summary,
            )
        )

    return experiments