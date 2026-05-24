import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/plot_results.py <aggregated.csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    if not Path(csv_path).exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    output_dir = Path(csv_path).parent / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Generating plots...\n")
    
    # Plot 1: Algorithm Comparison
    print("  [1] Algorithm Comparison")
    fig, ax = plt.subplots(figsize=(10, 6))
    algo_stats = df.groupby('algo')['iter_time_mean'].agg(['mean', 'std'])
    algo_stats['mean'].plot(kind='bar', ax=ax, yerr=algo_stats['std'], capsize=5, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_title('Algorithm Comparison: Mean Iteration Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Iteration Time (seconds)', fontsize=12)
    ax.set_xlabel('Algorithm', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_algorithm_comparison.png", dpi=300)
    plt.close()
    
    # Plot 2: Sync Time vs World Size
    print("  [2] Sync Time vs World Size")
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in sorted(df['algo'].unique()):
        algo_df = df[df['algo'] == algo].groupby('world_size')['sync_time_mean'].mean().sort_index()
        ax.plot(algo_df.index, algo_df.values, marker='o', linewidth=2, markersize=8, label=algo)
    ax.set_title('Synchronization Time vs World Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sync Time (seconds)', fontsize=12)
    ax.set_xlabel('World Size', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_sync_time_vs_world_size.png", dpi=300)
    plt.close()
    
    # Plot 3: Iteration Time vs Batch Size
    print("  [3] Iteration Time vs Batch Size")
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in sorted(df['algo'].unique()):
        algo_df = df[df['algo'] == algo].groupby('batch_size')['iter_time_mean'].mean().sort_index()
        ax.plot(algo_df.index, algo_df.values, marker='s', linewidth=2, markersize=8, label=algo)
    ax.set_title('Iteration Time vs Batch Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Iteration Time (seconds)', fontsize=12)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_iter_time_vs_batch_size.png", dpi=300)
    plt.close()
    
    # Plot 4: Model Comparison
    print("  [4] Model Comparison")
    fig, ax = plt.subplots(figsize=(10, 6))
    model_stats = df.groupby('model')['iter_time_mean'].agg(['mean', 'std'])
    model_stats['mean'].plot(kind='bar', ax=ax, yerr=model_stats['std'], capsize=5, color=['#95E1D3', '#F38181', '#AA96DA'])
    ax.set_title('Model Comparison: Mean Iteration Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Iteration Time (seconds)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_model_comparison.png", dpi=300)
    plt.close()
    
    # Plot 5: Loss by Epochs
    print("  [5] Loss vs Epochs")
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in sorted(df['algo'].unique()):
        algo_df = df[df['algo'] == algo].groupby('epochs')['loss_mean'].mean().sort_index()
        ax.plot(algo_df.index, algo_df.values, marker='^', linewidth=2, markersize=8, label=algo)
    ax.set_title('Loss vs Training Epochs', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_loss_vs_epochs.png", dpi=300)
    plt.close()
    
    # Plot 6: Heatmap
    print("  [6] Heatmap: Algo vs World Size")
    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap_data = df.pivot_table(values='iter_time_mean', index='algo', columns='world_size', aggfunc='mean')
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Iter Time (s)'})
    ax.set_title('Mean Iteration Time: Algorithm vs World Size', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/06_heatmap_algo_vs_world_size.png", dpi=300)
    plt.close()
    
    print(f"\n✓ Generated 6 plots in: {output_dir}/")
    print("  Files: 01_algorithm_comparison.png through 06_heatmap_algo_vs_world_size.png")

if __name__ == "__main__":
    main()
