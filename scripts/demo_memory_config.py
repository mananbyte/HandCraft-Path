#!/usr/bin/env python3
"""
Demo script showing dynamic memory configuration in action.

This script demonstrates how the memory detection and batch size computation
works without actually processing images.

Usage:
    python scripts/demo_memory_config.py
    python scripts/demo_memory_config.py --max-memory 8
    python scripts/demo_memory_config.py --memory-percent 60
    python scripts/demo_memory_config.py --memory-backend cpu
"""

import argparse
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.memory_config import MemoryConfig


def main():
    parser = argparse.ArgumentParser(
        description="Demo dynamic memory configuration for PanNuke pipeline"
    )
    parser.add_argument(
        "--max-memory", type=float, default=None,
        help="Max memory to use (GB)"
    )
    parser.add_argument(
        "--memory-percent", type=float, default=None,
        help="Use percentage of available memory"
    )
    parser.add_argument(
        "--memory-backend", choices=["auto", "cpu", "gpu"], default="auto",
        help="Backend selection"
    )
    parser.add_argument(
        "--safety-margin", type=float, default=25,
        help="Safety margin %"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*72)
    print("PanNuke Memory Configuration Demo")
    print("="*72 + "\n")
    
    # Create memory config with user options
    memory_config = MemoryConfig(
        max_memory_gb=args.max_memory,
        memory_percent=args.memory_percent,
        safety_margin_pct=args.safety_margin,
        force_backend=args.memory_backend,
        verbose=True,
    )
    
    # Compute batch size
    batch_size = memory_config.compute_batch_size()
    
    # Print detailed summary
    memory_config.print_summary()
    
    # Print usage examples
    print("="*72)
    print("USAGE EXAMPLES FOR YOUR SYSTEM")
    print("="*72 + "\n")
    
    if memory_config.backend == 'gpu':
        example_backend = "GPU"
        example_desc = "Fast GPU processing"
    else:
        example_backend = "CPU"
        example_desc = "CPU-only processing"
    
    print("1. Default (auto-optimized):")
    print(f"   $ python scripts/build_dataset.py --label-mode binary --folds 1,2")
    print(f"   → Will use batch size {batch_size} on {example_backend}\n")
    
    print("2. Safe mode (leave room for other processes):")
    safe_limit = max(2, int(memory_config.usable_memory_gb * 0.8 / (1 - 0.25)))
    print(f"   $ python scripts/build_dataset.py --label-mode binary --folds 1,2 --max-memory {safe_limit}")
    safe_batch = int((safe_limit * 1000 * 0.75) / 60)
    print(f"   → Will use batch size ~{safe_batch}\n")
    
    print("3. Aggressive mode (use more resources, faster):")
    if memory_config.backend == 'gpu':
        aggressive_limit = int(memory_config.usable_memory_gb * 1.3 / (1 - 0.25))
        print(f"   $ python scripts/build_dataset.py --label-mode binary --folds 1,2 --max-memory {aggressive_limit}")
        aggressive_batch = int((aggressive_limit * 1000 * 0.75) / 60)
        print(f"   → Will use batch size ~{aggressive_batch}\n")
    else:
        print("   → Not applicable on CPU-only systems\n")
    
    print("4. Multi-tenant mode (share resources 60/40):")
    shared_limit = int(memory_config.usable_memory_gb * 0.6 / (1 - 0.25))
    print(f"   $ python scripts/build_dataset.py --label-mode binary --folds 1,2 --max-memory {shared_limit}")
    shared_batch = int((shared_limit * 1000 * 0.75) / 60)
    print(f"   → Will use batch size ~{shared_batch}\n")
    
    # Performance estimation
    print("="*72)
    print("ESTIMATED PERFORMANCE")
    print("="*72 + "\n")
    
    # Assume ~4,768 images for Fold 1+2
    total_images = 4768
    images_per_sec = 0.5  # Conservative: 0.5 img/sec on average
    
    total_time_min = (total_images / (batch_size * images_per_sec)) * batch_size / 100
    hours = int(total_time_min / 60)
    minutes = int(total_time_min % 60)
    
    print(f"For Fold 1+2 binary (~{total_images} images):")
    print(f"  Batch size: {batch_size} images")
    print(f"  Estimated time: {hours}h {minutes}m (Pass 2 only)")
    print(f"  Memory usage: ~{memory_config.usable_memory_gb:.1f} GB")
    print()
    
    print("="*72 + "\n")


if __name__ == "__main__":
    main()
