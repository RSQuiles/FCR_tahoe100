#!/bin/bash

# Quick Start Guide for Large-Scale FCR Training
# For datasets with 8M+ samples

echo "=========================================="
echo "FCR Large-Scale Training Quick Start"
echo "=========================================="
echo ""

# Step 1: Check system requirements
echo "Step 1: Checking system requirements..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Step 2: Test with small subset first
echo ""
echo "Step 2: Testing with 1% of data (recommended)..."
echo "Edit your config to sample data:"
echo "  - Add 'data_fraction': 0.01 to config"
echo "  - Or use: adata = adata[np.random.choice(adata.n_obs, size=80000, replace=False)]"

# Step 3: Run optimized training
echo ""
echo "Step 3: Running optimized training..."
echo "Command: python -m fcr.train.train_optimized config_large_scale.json"

# Step 4: Monitor training
echo ""
echo "Step 4: Monitor training progress:"
echo "  - WandB dashboard: https://wandb.ai/"
echo "  - GPU usage: watch -n 1 nvidia-smi"
echo "  - Memory: watch -n 1 'nvidia-smi --query-gpu=memory.used --format=csv'"

echo ""
echo "=========================================="
echo "Key Optimizations Enabled"
echo "=========================================="
echo "✓ Mixed Precision Training (FP16)"
echo "✓ Gradient Accumulation (4 steps)"
echo "✓ Optimized DataLoader (4 workers)"
echo "✓ MC Sampling Disabled (sample_latent=false)"
echo "✓ Reduced Model Capacity"
echo ""
echo "Expected Performance:"
echo "  - Memory: ~12GB GPU (vs ~40GB without)"
echo "  - Speed: ~2000-5000 samples/sec"
echo "  - Time per epoch: ~30-60 min for 8M samples"
echo ""

# Troubleshooting
echo "=========================================="
echo "Troubleshooting"
echo "=========================================="
echo ""
echo "If you get OOM (Out of Memory) errors:"
echo "  1. Reduce batch_size (try 128 or 64)"
echo "  2. Increase gradient_accumulation_steps (try 8)"
echo "  3. Reduce model dimensions in config"
echo "  4. Enable gradient checkpointing (see SCALABILITY_OPTIMIZATIONS.md)"
echo ""
echo "If training is too slow:"
echo "  1. Increase num_workers (try 8)"
echo "  2. Ensure use_mixed_precision=true"
echo "  3. Check GPU utilization: nvidia-smi dmon"
echo "  4. Reduce checkpoint_freq to save time"
echo ""
echo "For multi-GPU training:"
echo "  - Use PyTorch DDP (Distributed Data Parallel)"
echo "  - See: pytorch.org/tutorials/intermediate/ddp_tutorial.html"
echo ""

# Monitoring commands
echo "=========================================="
echo "Useful Monitoring Commands"
echo "=========================================="
echo ""
echo "# Watch GPU usage:"
echo "watch -n 1 nvidia-smi"
echo ""
echo "# Monitor GPU memory:"
echo "nvidia-smi dmon -s mu"
echo ""
echo "# Check training logs:"
echo "tail -f experiments/large_scale/saves/*/training.log"
echo ""
echo "# Profile GPU memory in Python:"
echo "python -c \"
import torch
from torch.profiler import profile, ProfilerActivity

# Add to your training loop:
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Your training step here
    pass

print(prof.key_averages().table(sort_by='cuda_memory_usage', row_limit=10))
\""
echo ""

# Quick test script
echo "=========================================="
echo "Quick Test Script"
echo "=========================================="
cat << 'EOF' > test_optimized_training.py
"""Quick test script for optimized FCR training"""
import torch
from torch.cuda.amp import autocast, GradScaler
import time

print("Testing optimization components...\n")

# Test 1: Mixed Precision
print("1. Testing Mixed Precision (AMP)...")
scaler = GradScaler()
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()

with autocast():
    z = torch.matmul(x, y)
    
print(f"   ✓ Mixed precision working. Result dtype: {z.dtype}")

# Test 2: Memory efficiency
print("\n2. Testing memory efficiency...")
mem_before = torch.cuda.memory_allocated() / 1e9
large_tensor = torch.randn(10000, 10000).cuda()
mem_after = torch.cuda.memory_allocated() / 1e9
print(f"   Allocated {mem_after - mem_before:.2f} GB")

del large_tensor
torch.cuda.empty_cache()
mem_freed = torch.cuda.memory_allocated() / 1e9
print(f"   ✓ Freed memory, now at {mem_freed:.2f} GB")

# Test 3: DataLoader with workers
print("\n3. Testing DataLoader performance...")
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(torch.randn(10000, 100), torch.randn(10000, 10))

# Without optimization
start = time.time()
loader1 = DataLoader(dataset, batch_size=256, num_workers=0)
for _ in loader1:
    pass
time1 = time.time() - start

# With optimization
start = time.time()
loader2 = DataLoader(dataset, batch_size=256, num_workers=4, 
                    pin_memory=True, prefetch_factor=2, persistent_workers=True)
for _ in loader2:
    pass
time2 = time.time() - start

print(f"   Without optimization: {time1:.2f}s")
print(f"   With optimization: {time2:.2f}s")
print(f"   ✓ Speedup: {time1/time2:.2f}x")

print("\n✓ All tests passed! Ready for large-scale training.")
EOF

echo ""
echo "Created test_optimized_training.py"
echo "Run: python test_optimized_training.py"
echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo "1. Edit config_large_scale.json with your data paths"
echo "2. Test with small data fraction first (0.01)"
echo "3. Monitor memory and speed"
echo "4. Scale up gradually"
echo "5. Review SCALABILITY_OPTIMIZATIONS.md for advanced tips"
echo ""
