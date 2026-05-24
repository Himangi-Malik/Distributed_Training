import itertools
import json
from typing import List, Dict, Any

def generate_sweep_configs(
    algos: List[str] = None,
    models: List[str] = None,
    batch_sizes: List[int] = None,
    epochs: List[int] = None,
    world_sizes: List[int] = None,
) -> List[Dict[str, Any]]:
    """
    Generate all combinations of benchmark parameters.
    
    Returns:
        List of config dicts: {"algo": ..., "model": ..., "batch_size": ..., "epochs": ..., "world_size": ...}
    """
    algos = algos or ["ring", "tree", "parameter_server"]
    models = models or ["ann", "cnn", "rnn"]
    batch_sizes = batch_sizes or [2, 8, 32, 64, 128]
    epochs = epochs or [1, 5, 10, 20, 50]
    world_sizes = world_sizes or [1, 2, 4, 8]
    
    configs = []
    for algo, model, bs, ep, ws in itertools.product(
        algos, models, batch_sizes, epochs, world_sizes
    ):
        configs.append({
            "algo": algo,
            "model": model,
            "batch_size": bs,
            "epochs": ep,
            "world_size": ws,
        })
    
    return configs


def config_to_dirname(config: Dict[str, Any]) -> str:
    """Convert config dict to directory name."""
    return (
        f"algo={config['algo']}_"
        f"model={config['model']}_"
        f"bs={config['batch_size']}_"
        f"ep={config['epochs']}_"
        f"ws={config['world_size']}"
    )


def create_config_json(base_config: Dict[str, Any], sweep_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge base config with sweep parameters.
    Sweep parameters override base config values.
    Dynamically creates ip_list to match world_size (same IP, different ports).
    Uses DISTRIBUTED mode (all ranks spawn in parallel via socket connections).
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


if __name__ == "__main__":
    configs = generate_sweep_configs()
    print(f"Total configurations: {len(configs)}")
    print(f"Sample configs:")
    for cfg in configs[:5]:
        print(f"  {config_to_dirname(cfg)}")
