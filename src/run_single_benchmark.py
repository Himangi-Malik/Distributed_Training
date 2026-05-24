import json
import subprocess
import sys
import os
import shutil
from pathlib import Path
from typing import Dict, Any


def create_config_json(base_config: Dict[str, Any], sweep_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge base config with sweep parameters.
    Sweep parameters override base config values.
    Dynamically creates ip_list to match world_size (same IP, different ports).
    """
    merged = base_config.copy()
    world_size = sweep_config["world_size"]
    
    # Get base IP from config (use first IP or default to localhost)
    base_ip = base_config.get("ip_list", ["127.0.0.1"])[0]
    
    # Create ip_list matching world_size
    merged.update({
        "algo": sweep_config["algo"],
        "model": sweep_config["model"],
        "batch_size": sweep_config["batch_size"],
        "epochs": sweep_config["epochs"],
        "world_size": world_size,
        "ip_list": [base_ip] * world_size,  # Same IP repeated for each rank
    })
    return merged


def run_single_benchmark(
    config: Dict[str, Any],
    base_config: Dict[str, Any],
    output_dir: Path,
    verbose: bool = False,
) -> bool:
    """
    Run a single benchmark with given config using distributed mode.
    Spawns all ranks in PARALLEL so they can connect to each other.
    
    Args:
        config: Sweep config (algo, model, batch_size, epochs, world_size)
        base_config: Base config loaded from config.json
        output_dir: Directory to store metrics and config
        verbose: Print output from benchmark
    
    Returns:
        True if successful, False otherwise
    """
    full_config = create_config_json(base_config, config)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output (for reference)
    config_ref_path = output_dir / "config.json"
    with open(config_ref_path, "w") as f:
        json.dump(full_config, f, indent=2)
    
    # Backup original config.json
    original_config_path = Path("src/config.json")
    backup_config_path = Path("src/config.json.backup")
    
    if original_config_path.exists():
        shutil.copy(original_config_path, backup_config_path)
    
    try:
        # Write temp config to src/config.json (where main_comm_runner.py looks for it)
        with open(original_config_path, "w") as f:
            json.dump(full_config, f, indent=2)
        
        # Run ALL ranks in PARALLEL (not sequential)
        # This allows them to connect to each other via socket topology
        world_size = config["world_size"]
        processes = []
        
        # Start all rank processes concurrently
        for rank in range(world_size):
            cmd = [
                sys.executable,
                "main_comm_runner.py",
                "--rank", str(rank),
            ]
            
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd="src",
                    stdout=subprocess.PIPE if not verbose else None,
                    stderr=subprocess.PIPE if not verbose else None,
                    text=True,
                )
                processes.append((rank, proc))
                if verbose:
                    print(f"  Rank {rank} process spawned (PID: {proc.pid})")
            except Exception as e:
                print(f"ERROR: Failed to spawn rank {rank}: {e}")
                # Kill any already-spawned processes
                for _, p in processes:
                    try:
                        p.terminate()
                    except:
                        pass
                return False
        
        # Wait for all rank processes to complete
        failed_ranks = []
        for rank, proc in processes:
            try:
                proc.wait(timeout=3600)  # 1 hour timeout
                if proc.returncode != 0:
                    failed_ranks.append(rank)
                    if verbose:
                        print(f"  Rank {rank} failed with code {proc.returncode}")
                else:
                    if verbose:
                        print(f"  Rank {rank} completed successfully")
            except subprocess.TimeoutExpired:
                failed_ranks.append(rank)
                proc.kill()
                if verbose:
                    print(f"  Rank {rank} timed out (killed)")
            except Exception as e:
                failed_ranks.append(rank)
                if verbose:
                    print(f"  Rank {rank} error: {e}")
        
        # Copy metrics from benchmark_results/ to output_dir (they're in src/benchmark_results/)
        src_metrics_dir = Path("src/benchmark_results")
        if src_metrics_dir.exists():
            for metrics_file in src_metrics_dir.glob("metrics_rank*.json"):
                dest_file = output_dir / metrics_file.name
                shutil.copy(metrics_file, dest_file)
                if verbose:
                    print(f"  Copied {metrics_file.name} to {output_dir}")
        
        # Check if all ranks succeeded
        if failed_ranks:
            print(f"ERROR: Ranks {failed_ranks} failed for config {config}")
            return False
        
        return True
    
    finally:
        # Restore original config.json
        if backup_config_path.exists():
            shutil.move(backup_config_path, original_config_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True, help="Sweep config as JSON string or file")
    parser.add_argument("--base-config", required=True, help="Path to base config.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for metrics")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Load configs
    with open(args.base_config) as f:
        base_config = json.load(f)
    
    if args.config_json.startswith("{"):
        sweep_config = json.loads(args.config_json)
    else:
        with open(args.config_json) as f:
            sweep_config = json.load(f)
    
    output_dir = Path(args.output_dir)
    
    success = run_single_benchmark(sweep_config, base_config, output_dir, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True, help="Sweep config as JSON string or file")
    parser.add_argument("--base-config", required=True, help="Path to base config.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for metrics")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Load configs
    with open(args.base_config) as f:
        base_config = json.load(f)
    
    if args.config_json.startswith("{"):
        sweep_config = json.loads(args.config_json)
    else:
        with open(args.config_json) as f:
            sweep_config = json.load(f)
    
    output_dir = Path(args.output_dir)
    
    success = run_single_benchmark(sweep_config, base_config, output_dir, verbose=args.verbose)
    sys.exit(0 if success else 1)
