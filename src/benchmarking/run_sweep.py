import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

from .benchmark_sweep import generate_sweep_configs, config_to_dirname
from .run_single_benchmark import run_single_benchmark


def run_full_sweep(
    base_config_path: Path,
    output_base_dir: Path,
    algos: Optional[list] = None,
    models: Optional[list] = None,
    batch_sizes: Optional[list] = None,
    epochs: Optional[list] = None,
    world_sizes: Optional[list] = None,
    num_configs: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """
    Run full sweep with given parameters.
    
    Args:
        base_config_path: Path to base config.json
        output_base_dir: Base directory for all sweep results
        algos, models, batch_sizes, epochs, world_sizes: Parameter ranges
        num_configs: Run only first N configs (for testing)
        verbose: Print detailed output
    
    Returns:
        Summary dict with stats
    """
    # Load base config
    with open(base_config_path) as f:
        base_config = json.load(f)
    
    # Generate all configurations
    all_configs = generate_sweep_configs(algos, models, batch_sizes, epochs, world_sizes)
    
    if num_configs:
        configs_to_run = all_configs[:num_configs]
        print(f"Running {num_configs} of {len(all_configs)} total configs (test mode)")
    else:
        configs_to_run = all_configs
        print(f"Running {len(all_configs)} total configs")
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    sweep_dir = output_base_dir / f"sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {sweep_dir}\n")
    
    # Track results
    results = {
        "total": len(configs_to_run),
        "completed": 0,
        "failed": 0,
        "failed_configs": [],
        "start_time": datetime.now().isoformat(),
    }
    
    # Run each config
    for idx, config in enumerate(configs_to_run, 1):
        config_dir = sweep_dir / config_to_dirname(config)
        
        try:
            print(f"[{idx}/{len(configs_to_run)}] Running {config_to_dirname(config)}...", end=" ", flush=True)
            
            success = run_single_benchmark(config, base_config, config_dir, verbose=verbose)
            
            if success:
                print("✓")
                results["completed"] += 1
            else:
                print("✗")
                results["failed"] += 1
                results["failed_configs"].append(config_to_dirname(config))
        
        except Exception as e:
            print(f"✗ (Exception: {e})")
            results["failed"] += 1
            results["failed_configs"].append(config_to_dirname(config))
    
    results["end_time"] = datetime.now().isoformat()
    
    # Save results summary
    results_file = sweep_dir / "sweep_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Sweep Complete!")
    print(f"  Total: {results['total']}")
    print(f"  Completed: {results['completed']}")
    print(f"  Failed: {results['failed']}")
    if results["failed_configs"]:
        print(f"  Failed configs:")
        for cfg in results["failed_configs"][:5]:
            print(f"    - {cfg}")
        if len(results["failed_configs"]) > 5:
            print(f"    ... and {len(results['failed_configs']) - 5} more")
    print(f"\nOutput: {sweep_dir}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark sweep with multiple parameter combinations")
    parser.add_argument("--base-config", default="src/config.json", help="Path to base config.json")
    parser.add_argument("--output-dir", default="benchmark_results", help="Base output directory")
    parser.add_argument("--num-configs", type=int, default=None, help="Run only first N configs (for testing)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    
    # Parameter ranges
    parser.add_argument("--algos", nargs="+", default=["ring", "tree", "parameter_server"])
    parser.add_argument("--models", nargs="+", default=["ann", "cnn", "rnn"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 8, 32, 64, 128])
    parser.add_argument("--epochs", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    
    args = parser.parse_args()
    
    base_config_path = Path(args.base_config)
    output_dir = Path(args.output_dir)
    
    if not base_config_path.exists():
        print(f"ERROR: Base config not found: {base_config_path}")
        sys.exit(1)
    
    run_full_sweep(
        base_config_path,
        output_dir,
        algos=args.algos,
        models=args.models,
        batch_sizes=args.batch_sizes,
        epochs=args.epochs,
        world_sizes=args.world_sizes,
        num_configs=args.num_configs,
        verbose=args.verbose,
    )