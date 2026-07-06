"""Generate publication-ready figures from benchmark results."""

from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from statistics import mean


from .experiment_results import (
    ExperimentResult,
    load_experiment_results,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "paper_results" / "figures"

MODELS = [
    "ann",
    "cnn",
    "rnn",
]

BATCH_SIZES = [
    16,
    32,
    64,
    128,
]

WORLD_SIZES = [
    1,
    2,
    4,
    8,
    16,
]

ALGORITHMS = [
    "ring",
    "tree",
    "parameter_server",
]

def _sync_time(experiment: ExperimentResult) -> float:
    """Return mean synchronization time in milliseconds."""
    return experiment.summary["timing"]["sync"]["mean"] * 1000


def _build_lookup(results: list[ExperimentResult]):
    """Build a lookup table for fast experiment retrieval."""

    lookup = {}

    for result in results:
        key = (
            result.config["algo"],
            result.config["model"],
            result.config["batch_size"],
            result.config["world_size"],
        )
        lookup[key] = result

    return lookup


def _format_algorithm(name: str) -> str:
    return name.replace("_", " ").title()

def _sync_fraction(experiment: ExperimentResult) -> float:
    """Return synchronization fraction as a percentage."""
    return experiment.summary["derived"]["sync_fraction"] * 100

def _group_by_algorithm_and_world_size(
    results: list[ExperimentResult],
    metric_fn,
):
    """
    Group a metric by (algorithm, world size).

    Returns
    -------
    {
        (algorithm, world_size): [metric values]
    }
    """

    grouped = defaultdict(list)

    for experiment in results:
        key = (
            experiment.config["algo"],
            experiment.config["world_size"],
        )

        grouped[key].append(metric_fn(experiment))

    return grouped

def _throughput(experiment: ExperimentResult) -> float:
    """Return training throughput in samples per second."""

    batch_size = experiment.config["batch_size"]
    world_size = experiment.config["world_size"]

    iteration_time = experiment.summary["timing"]["iteration"]["mean"]

    return (batch_size * world_size) / iteration_time

def _speedup(experiment: ExperimentResult, baseline_lookup) -> float:
    """Return speedup relative to the single-worker baseline."""

    key = (
        experiment.config["algo"],
        experiment.config["model"],
        experiment.config["batch_size"],
    )

    baseline = baseline_lookup.get(key)

    if baseline is None:
        return None

    baseline_time = baseline.summary["timing"]["iteration"]["mean"]
    current_time = experiment.summary["timing"]["iteration"]["mean"]

    return baseline_time / current_time

def _parallel_efficiency(
    experiment: ExperimentResult,
    baseline_lookup,
) -> float:
    """Return parallel efficiency as a percentage."""

    speedup = _speedup(
        experiment,
        baseline_lookup,
    )

    if speedup is None:
        return None

    return (speedup / experiment.config["world_size"]) * 100


def figure_1_sync_scaling(
    results: list[ExperimentResult],
):
    """
    Figure 1:
    Mean synchronization time vs. world size for every model and batch size.
    """

    lookup = _build_lookup(results)

    fig, axes = plt.subplots(
        nrows=len(MODELS),
        ncols=len(BATCH_SIZES),
        figsize=(18, 10),
        sharex=True,
        sharey=True,
    )

    for row, model in enumerate(MODELS):

        for col, batch_size in enumerate(BATCH_SIZES):

            ax = axes[row][col]

            for algorithm in ALGORITHMS:

                world_sizes = []
                sync_times = []

                for world_size in WORLD_SIZES:

                    experiment = lookup.get(
                        (
                            algorithm,
                            model,
                            batch_size,
                            world_size,
                        )
                    )

                    if experiment is None:
                        continue

                    world_sizes.append(world_size)
                    sync_times.append(
                        _sync_time(experiment)
                    )

                if world_sizes:
                    ax.plot(
                        world_sizes,
                        sync_times,
                        marker="o",
                        linewidth=2,
                        label=_format_algorithm(algorithm),
                    )

            # If nothing was plotted for this subplot
            if not ax.has_data():
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="gray",
                )

            ax.set_title(f"Batch Size = {batch_size}")

            if col == 0:
                ax.set_ylabel(
                    f"{model.upper()}\n\nSync Time (ms)"
                )

            if row == len(MODELS) - 1:
                ax.set_xlabel("World Size")

            ax.set_xticks(WORLD_SIZES)
            ax.grid(True, alpha=0.3)

    # Shared legend (only if something was plotted)
    handles, labels = axes[0][0].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(ALGORITHMS),
            frameon=False,
        )

    fig.suptitle(
        "Figure 1. Mean Synchronization Time vs. World Size",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        OUTPUT_DIR / "figure_1_sync_scaling.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

def figure_2_sync_fraction(
    results: list[ExperimentResult],
):
    """
    Figure 2:
    Average synchronization fraction vs. world size.
    """

    grouped = _group_by_algorithm_and_world_size(
        results,
        _sync_fraction,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for algorithm in ALGORITHMS:

        world_sizes = []
        sync_fractions = []

        for world_size in WORLD_SIZES:

            values = grouped.get(
                (algorithm, world_size),
                [],
            )

            if not values:
                continue

            world_sizes.append(world_size)
            sync_fractions.append(mean(values))

        if world_sizes:
            ax.plot(
                world_sizes,
                sync_fractions,
                marker="o",
                linewidth=2,
                label=_format_algorithm(algorithm),
            )

    ax.set_title("Figure 2. Synchronization Fraction vs. World Size")

    ax.set_xlabel("World Size")
    ax.set_ylabel("Synchronization Fraction (%)")

    ax.set_xticks(WORLD_SIZES)

    ax.grid(True, alpha=0.3)

    ax.legend(frameon=False)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        OUTPUT_DIR / "figure_2_sync_fraction.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

def figure_3_throughput(
    results: list[ExperimentResult],
):
    """
    Figure 3:
    Average training throughput vs. world size.
    """

    grouped = _group_by_algorithm_and_world_size(
        results,
        _throughput,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for algorithm in ALGORITHMS:

        world_sizes = []
        throughputs = []

        for world_size in WORLD_SIZES:

            values = grouped.get(
                (algorithm, world_size),
                [],
            )

            if not values:
                continue

            world_sizes.append(world_size)
            throughputs.append(mean(values))

        if world_sizes:
            ax.plot(
                world_sizes,
                throughputs,
                marker="o",
                linewidth=2,
                label=_format_algorithm(algorithm),
            )

    ax.set_title("Figure 3. Throughput vs. World Size")

    ax.set_xlabel("World Size")
    ax.set_ylabel("Throughput (samples/sec)")

    ax.set_xticks(WORLD_SIZES)

    ax.grid(True, alpha=0.3)

    ax.legend(frameon=False)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        OUTPUT_DIR / "figure_3_throughput.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

def figure_4_speedup(
    results: list[ExperimentResult],
):
    """
    Figure 4:
    Average speedup vs. world size.
    """

    baseline_lookup = {}

    for experiment in results:

        if experiment.config["world_size"] != 1:
            continue

        key = (
            experiment.config["algo"],
            experiment.config["model"],
            experiment.config["batch_size"],
        )

        baseline_lookup[key] = experiment

    grouped = _group_by_algorithm_and_world_size(
        [
            experiment
            for experiment in results
            if experiment.config["world_size"] >= 1
        ],
        lambda experiment: _speedup(
            experiment,
            baseline_lookup,
        ),
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for algorithm in ALGORITHMS:

        world_sizes = []
        speedups = []

        for world_size in WORLD_SIZES:

            values = [
                value
                for value in grouped.get(
                    (algorithm, world_size),
                    [],
                )
                if value is not None
            ]

            if not values:
                continue

            world_sizes.append(world_size)
            speedups.append(mean(values))

        if world_sizes:
            ax.plot(
                world_sizes,
                speedups,
                marker="o",
                linewidth=2,
                label=_format_algorithm(algorithm),
            )

    # Ideal linear speedup
    ax.plot(
        WORLD_SIZES,
        WORLD_SIZES,
        "--",
        linewidth=1.5,
        color="black",
        label="Ideal",
    )

    ax.set_title("Figure 4. Speedup vs. World Size")

    ax.set_xlabel("World Size")
    ax.set_ylabel("Speedup")

    ax.set_xticks(WORLD_SIZES)

    ax.grid(True, alpha=0.3)

    ax.legend(frameon=False)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        OUTPUT_DIR / "figure_4_speedup.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

def figure_5_parallel_efficiency(
    results: list[ExperimentResult],
):
    """
    Figure 5:
    Average parallel efficiency vs. world size.
    """

    baseline_lookup = {}

    for experiment in results:

        if experiment.config["world_size"] != 1:
            continue

        key = (
            experiment.config["algo"],
            experiment.config["model"],
            experiment.config["batch_size"],
        )

        baseline_lookup[key] = experiment

    grouped = _group_by_algorithm_and_world_size(
        results,
        lambda experiment: _parallel_efficiency(
            experiment,
            baseline_lookup,
        ),
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for algorithm in ALGORITHMS:

        world_sizes = []
        efficiencies = []

        for world_size in WORLD_SIZES:

            values = [
                value
                for value in grouped.get(
                    (algorithm, world_size),
                    [],
                )
                if value is not None
            ]

            if not values:
                continue

            world_sizes.append(world_size)
            efficiencies.append(mean(values))

        if world_sizes:
            ax.plot(
                world_sizes,
                efficiencies,
                marker="o",
                linewidth=2,
                label=_format_algorithm(algorithm),
            )

    # Ideal efficiency
    ax.axhline(
        y=100,
        linestyle="--",
        linewidth=1.5,
        color="black",
        label="Ideal",
    )

    ax.set_title("Figure 5. Parallel Efficiency vs. World Size")

    ax.set_xlabel("World Size")
    ax.set_ylabel("Parallel Efficiency (%)")

    ax.set_xticks(WORLD_SIZES)

    ax.grid(True, alpha=0.3)

    ax.legend(frameon=False)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        OUTPUT_DIR / "figure_5_parallel_efficiency.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

def generate_all_figures():
    results = load_experiment_results()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure_1_sync_scaling(results)
    figure_2_sync_fraction(results)
    figure_3_throughput(results)
    figure_4_speedup(results)
    figure_5_parallel_efficiency(results)