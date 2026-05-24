import json
import csv
from pathlib import Path
from typing import List, Dict, Any


def aggregate_sweep_results(sweep_dir: Path) -> List[Dict[str, Any]]:
    """
    Aggregate metrics from all config subdirectories.
    
    Returns:
        List of dicts: [{algo, model, batch_size, epochs, world_size, metrics...}, ...]
    """
    results = []
    
    # Iterate through all config directories
    for config_dir in sorted(sweep_dir.iterdir()):
        if not config_dir.is_dir() or config_dir.name.startswith("sweep_"):
            continue
        
        # Parse config dir name
        parts = config_dir.name.split("_")
        config_dict = {}
        for part in parts:
            if "=" in part:
                key, val = part.split("=", 1)
                if key == "bs":
                    config_dict["batch_size"] = int(val)
                elif key == "ep":
                    config_dict["epochs"] = int(val)
                elif key == "ws":
                    config_dict["world_size"] = int(val)
                else:
                    config_dict[key] = val
        
        # Aggregate metrics from all ranks
        metrics_files = list(config_dir.glob("metrics_rank*.json"))
        
        if not metrics_files:
            print(f"Warning: No metrics found in {config_dir}")
            continue
        
        # Aggregate across ranks (simple average)
        aggregated = {
            "compute_time_mean": [],
            "sync_time_mean": [],
            "optim_time_mean": [],
            "iter_time_mean": [],
            "loss_mean": [],
            "grad_norm_mean": [],
        }
        
        for mf in metrics_files:
            with open(mf) as f:
                data = json.load(f)
                if "statistics" in data:
                    stats = data["statistics"]
                    aggregated["compute_time_mean"].append(stats.get("compute", {}).get("mean", 0))
                    aggregated["sync_time_mean"].append(stats.get("sync", {}).get("mean", 0))
                    aggregated["optim_time_mean"].append(stats.get("optim", {}).get("mean", 0))
                    aggregated["iter_time_mean"].append(stats.get("iter", {}).get("mean", 0))
                    aggregated["loss_mean"].append(stats.get("loss", {}).get("mean", 0))
                    aggregated["grad_norm_mean"].append(stats.get("grad_norm", {}).get("mean", 0))
        
        # Average across ranks
        for key in aggregated:
            if aggregated[key]:
                aggregated[key] = sum(aggregated[key]) / len(aggregated[key])
            else:
                aggregated[key] = None
        
        # Combine config + metrics
        result = {**config_dict, **aggregated}
        results.append(result)
    
    return results


def save_to_csv(results: List[Dict[str, Any]], output_path: Path):
    """Save aggregated results to CSV."""
    if not results:
        print("No results to save")
        return
    
    # Get all keys
    keys = list(results[0].keys())
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Saved {len(results)} rows to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate sweep results into CSV")
    parser.add_argument("sweep_dir", help="Sweep output directory (contains config subdirectories)")
    parser.add_argument("--output", default=None, help="Output CSV file (default: sweep_dir/aggregated.csv)")
    
    args = parser.parse_args()
    
    sweep_dir = Path(args.sweep_dir)
    output_path = Path(args.output) if args.output else sweep_dir / "aggregated.csv"
    
    print(f"Aggregating results from {sweep_dir}...")
    results = aggregate_sweep_results(sweep_dir)
    print(f"Found {len(results)} configurations")
    
    save_to_csv(results, output_path)
    
    # Print sample
    print(f"\nSample results (first 5):")
    for res in results[:5]:
        print(f"  {res['algo']} {res['model']} bs={res['batch_size']} ep={res['epochs']} ws={res['world_size']}: "
              f"iter_time={res.get('iter_time_mean', 'N/A'):.3f}s sync_time={res.get('sync_time_mean', 'N/A'):.3f}s")
