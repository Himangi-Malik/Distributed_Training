import pandas as pd
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/read_results.py <aggregated.csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    if not Path(csv_path).exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*100}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*100}\n")
    
    print(f"Total configurations: {len(df)}")
    print(f"Algorithms: {sorted(df['algo'].unique().tolist())}")
    print(f"Models: {sorted(df['model'].unique().tolist())}")
    print(f"World sizes: {sorted(df['world_size'].unique().tolist())}\n")
    
    # By Algorithm
    print(f"\n{'='*100}")
    print("RESULTS BY ALGORITHM")
    print(f"{'='*100}\n")
    
    for algo in sorted(df['algo'].unique()):
        algo_df = df[df['algo'] == algo]
        print(f"\n{algo.upper()}")
        print(f"  Compute time (mean):  {algo_df['compute_time_mean'].mean():.4f}s ± {algo_df['compute_time_mean'].std():.4f}s")
        print(f"  Sync time (mean):     {algo_df['sync_time_mean'].mean():.4f}s ± {algo_df['sync_time_mean'].std():.4f}s")
        print(f"  Iter time (mean):     {algo_df['iter_time_mean'].mean():.4f}s ± {algo_df['iter_time_mean'].std():.4f}s")
        print(f"  Loss (mean):          {algo_df['loss_mean'].mean():.6f}")
        print(f"  Grad norm (mean):     {algo_df['grad_norm_mean'].mean():.6f}")
    
    # Sample rows
    print(f"\n{'='*100}")
    print("SAMPLE DATA (first 20 rows)")
    print(f"{'='*100}\n")
    
    cols = ['algo', 'model', 'batch_size', 'epochs', 'world_size', 'compute_time_mean', 'sync_time_mean', 'iter_time_mean', 'loss_mean']
    print(df[cols].head(20).to_string(index=False))

if __name__ == "__main__":
    main()
