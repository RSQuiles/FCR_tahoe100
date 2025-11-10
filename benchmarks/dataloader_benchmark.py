"""
DataLoader benchmark script for the FCR project.
Measures batch time and throughput for a DataLoader configuration.

Usage examples:
  python fcr/benchmarks/dataloader_benchmark.py --data-path /path/to/data.h5ad --batch-size 256 --num-workers 8 --num-batches 200

The script will import repo code by adding the repo root to sys.path, so run from anywhere.
"""
import os
import sys
import time
import argparse

# Add repo root to path (two levels up from this script: fcr/benchmarks -> repo_root)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from torch.utils.data import DataLoader

# Import dataset loader and collate from repo
try:
    from fcr.dataset.dataset import load_dataset_train_test
    from fcr.utils.data_utils import data_collate
except Exception as e:
    print("Failed to import project modules:", e)
    print("Make sure you run this script from within the workspace or set PYTHONPATH to the repo root.")
    raise


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', type=str, required=True, help='Path to the h5ad dataset (or dataset used by load_dataset_train_test)')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--persistent-workers', action='store_true')
    p.add_argument('--pin-memory', action='store_true')
    p.add_argument('--prefetch-factor', type=int, default=2)
    p.add_argument('--num-batches', type=int, default=200, help='Number of batches to measure (after warmup)')
    p.add_argument('--warmup-batches', type=int, default=10, help='Warmup batches to skip from timing')
    p.add_argument('--batch-print', type=int, default=50, help='Print progress every N batches')
    p.add_argument('--split-key', type=str, default='split', help='split_key argument passed to load_dataset_train_test')
    p.add_argument('--covariate-keys', type=str, default='cell_name')
    p.add_argument('--perturbation-key', type=str, default='Agg_Treatment')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print('Loading dataset (this may still be I/O heavy)...')
    # Call the loader similarly to train.prepare in the repo. Some arguments are intentionally minimal.
    datasets = load_dataset_train_test(
        args.data_path,
        perturbation_input='ohe',
        covariate_keys=args.covariate_keys,
        perturbation_key=args.perturbation_key,
        split_key=args.split_key,
        sample_cf=False,
        control_name="DMSO_TF",
        embedded_dose=None,
        args={"batch_size": args.batch_size}
    )

    dataset = datasets['train']
    print(f'Dataset loaded. num_samples={len(dataset)}. num_outcomes={getattr(dataset, "num_outcomes", "?")}.')

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=None,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers>0 else 2,
        collate_fn=(lambda batch: data_collate(batch, nb_dims=1))
    )

    print('DataLoader created:')
    print(f'  batch_size={args.batch_size}, num_workers={args.num_workers}, persistent_workers={args.persistent_workers}, pin_memory={args.pin_memory}, prefetch_factor={args.prefetch_factor}')

    # Warmup
    it = iter(loader)
    print(f'Warming up for {args.warmup_batches} batches...')
    for i in range(args.warmup_batches):
        try:
            _ = next(it)
        except StopIteration:
            break

    # Timed run
    print(f'Starting timed run for {args.num_batches} batches...')
    t0 = time.time()
    batch_count = 0
    samples_count = 0
    start = time.time()
    while batch_count < args.num_batches:
        try:
            batch = next(it)
        except StopIteration:
            # restart iterator
            it = iter(loader)
            try:
                batch = next(it)
            except StopIteration:
                break
        batch_count += 1
        # estimate number of samples in batch (handles tuple outputs)
        if isinstance(batch, (list, tuple)):
            first = batch[0]
            try:
                bsize = first.size(0)
            except Exception:
                # fallback
                bsize = args.batch_size
        else:
            try:
                bsize = batch.size(0)
            except Exception:
                bsize = args.batch_size
        samples_count += bsize
        if (batch_count % args.batch_print) == 0:
            elapsed = time.time() - t0
            print(f'  batches={batch_count}/{args.num_batches} elapsed={elapsed:.2f}s avg_batch={elapsed/batch_count:.3f}s samples/sec={samples_count/elapsed:.1f}')

    elapsed = time.time() - t0
    avg_batch = elapsed / max(1, batch_count)
    samples_per_sec = samples_count / elapsed if elapsed>0 else 0.0

    print('\nBenchmark results:')
    print(f'  total_batches_measured={batch_count}')
    print(f'  total_samples={samples_count}')
    print(f'  elapsed_sec={elapsed:.2f}')
    print(f'  avg_sec_per_batch={avg_batch:.4f}')
    print(f'  samples_per_sec={samples_per_sec:.1f}')

    print('\nNext diagnostics:')
    print(' - If samples_per_sec is low and CPU usage is low, investigate storage (I/O) and h5 chunking; try copying dataset to local SSD and re-run.')
    print(' - If CPU usage is high, inspect Dataset.__getitem__ and collate_fn for Python-level overheads.')
    print(' - Try changing num_workers and persistent_workers to see effect.')


if __name__ == '__main__':
    main()
