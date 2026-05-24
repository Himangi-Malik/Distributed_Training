"""Context managers and utilities for precise timing measurements."""

import time
from contextlib import contextmanager
from typing import Dict, List, Optional


class TimingTracker:
    """Track timing for multiple phases with statistical analysis."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
    
    def record(self, phase: str, elapsed: float) -> None:
        """Record a timing measurement for a phase."""
        if phase not in self.timings:
            self.timings[phase] = []
        self.timings[phase].append(elapsed)
    
    def get_stats(self, phase: str) -> Dict[str, float]:
        """Compute statistics for a phase."""
        if phase not in self.timings or not self.timings[phase]:
            return {}
        
        times = sorted(self.timings[phase])
        n = len(times)
        
        mean = sum(times) / n
        variance = sum((t - mean) ** 2 for t in times) / n
        std = variance ** 0.5
        
        # Percentiles
        p95_idx = max(0, int(0.95 * (n - 1)))
        p99_idx = max(0, int(0.99 * (n - 1)))
        
        return {
            "count": n,
            "min": times[0],
            "max": times[-1],
            "mean": mean,
            "std": std,
            "p95": times[p95_idx],
            "p99": times[p99_idx],
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all tracked phases."""
        return {phase: self.get_stats(phase) for phase in self.timings}


class PrecisionTimer:
    """Context manager for precise timing with perf_counter."""
    
    def __init__(self, tracker: TimingTracker, phase: str):
        self.tracker = tracker
        self.phase = phase
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time
        self.tracker.record(self.phase, self.elapsed)
        return False


@contextmanager
def timer(tracker: TimingTracker, phase: str):
    """
    Context manager for timing a code block.
    
    Usage:
        tracker = TimingTracker()
        with timer(tracker, "compute"):
            # code to time
    """
    t = PrecisionTimer(tracker, phase)
    with t:
        yield t
