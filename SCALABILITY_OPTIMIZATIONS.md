# FCR Scalability Optimizations for Large Datasets (8M+ samples)

## Current Bottlenecks Identified:

1. **MC Sampling Memory Explosion**: `mc_sample_size=30` creates 30x memory overhead
2. **No Mixed Precision Training**: FP32 uses 2x memory vs FP16
3. **No Gradient Checkpointing**: Deep networks store all activations
4. **Inefficient DataLoader**: No prefetching, pinned memory, or workers
5. **Large Embedding Tables**: All embeddings loaded at once
6. **No Gradient Accumulation**: Forces large batch sizes
7. **Inefficient Loss Computation**: Repeating tensors creates copies

## Priority Optimizations (Implement in Order):

### 1. **CRITICAL: Enable Mixed Precision Training**
**Impact**: 50% memory reduction, 2-3x speedup
**Implementation**: Add to training loop

```python
# In train/train.py
from torch.cuda.amp import autocast, GradScaler

# Add to train() function:
scaler = GradScaler()

# Wrap forward/backward:
with autocast():
    minibatch_training_stats = model.update(
        experiment, treatment, control, covariates, adv_training
    )
    
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. **CRITICAL: Reduce MC Sample Size**
**Impact**: 30x memory reduction if reduced from 30 to 1
**Implementation**: Set in config

```json
{
    "sample_latent": false,  // This sets mc_sample_size to 1
}
```

Or use adaptive sampling:
```python
# Only use MC sampling during evaluation, not training
mc_sample_size = 1 if training else 5
```

### 3. **HIGH: Enable Gradient Accumulation**
**Impact**: Allows smaller batch sizes, reducing OOM errors
**Implementation**:

```python
# In train/train.py, modify training loop:
accumulation_steps = 4  # Effective batch_size = batch_size * accumulation_steps

for i, data in enumerate(datasets["loader_tr"]):
    minibatch_training_stats = model.update(
        experiment, treatment, control, covariates, adv_training
    )
    
    if (i + 1) % accumulation_steps == 0:
        model.optimizer_autoencoder.step()
        model.optimizer_autoencoder.zero_grad()
        if adv_training:
            model.optimizer_discriminator.step()
            model.optimizer_discriminator.zero_grad()
```

### 4. **HIGH: Optimize DataLoader**
**Impact**: Faster data loading, better GPU utilization

```python
# In prepare() function:
datasets.update({
    "loader_tr": torch.utils.data.DataLoader(
        datasets["train"],
        batch_size=args["batch_size"],
        shuffle=True,
        collate_fn=(lambda batch: data_collate(batch, nb_dims=1)),
        num_workers=4,              # NEW: Parallel data loading
        pin_memory=True,            # NEW: Faster GPU transfer
        prefetch_factor=2,          # NEW: Prefetch batches
        persistent_workers=True,    # NEW: Keep workers alive
    )
})
```

### 5. **MEDIUM: Add Gradient Checkpointing**
**Impact**: Trade compute for memory (30-40% memory reduction)
**Implementation**:

```python
# In model/model.py, add to encoders/decoders:
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(self, x):
    return checkpoint(self.original_forward, x, use_reentrant=False)
```

### 6. **MEDIUM: Optimize Sample Operations**
**Current Problem**: `outcomes.repeat(self.mc_sample_size, ...)` creates full copies

```python
# In model.py, replace repeat operations:
# BEFORE:
outcomes.repeat(self.mc_sample_size, *[1]*(outcomes.dim()-1))

# AFTER: Use expand (creates view, not copy)
outcomes.expand(self.mc_sample_size, -1, -1)  # Adjust dims as needed
```

### 7. **MEDIUM: Use Sparse Embeddings**
**Impact**: Memory reduction for large vocabulary

```python
# In model/model.py:
self.outcomes_embeddings = nn.Embedding(
    num_embeddings=self.num_outcomes,
    embedding_dim=self.hparams["outcome_emb_dim"],
    sparse=True  # NEW: Enable sparse gradients
)
```

### 8. **LOW: Enable Compilation (PyTorch 2.0+)**
**Impact**: 20-30% speedup

```python
# In model/model.py, after initialization:
if hasattr(torch, 'compile'):
    self.encoder_ZX = torch.compile(self.encoder_ZX)
    self.encoder_ZT = torch.compile(self.encoder_ZT)
    self.encoder_ZXT = torch.compile(self.encoder_ZXT)
    self.decoder = torch.compile(self.decoder)
```

## Recommended Configuration for 8M Samples:

```json
{
    "batch_size": 512,           // Smaller batch with grad accumulation
    "sample_latent": false,       // Disable MC sampling during training
    "max_epochs": 50,             // Fewer epochs needed with large data
    "checkpoint_freq": 5,         // Less frequent evaluation
    
    "hparams": {
        "ZX_dim": 32,             // Reduce latent dims
        "ZT_dim": 32,
        "ZXT_dim": 32,
        "outcome_emb_dim": 128,   // Reduce embedding size
        "encoder_width": 64,      // Narrower networks
        "encoder_depth": 2,       // Shallower networks
        "decoder_width": 64,
        "decoder_depth": 2,
        "discriminator_width": 32,
        "discriminator_depth": 1,
        "autoencoder_lr": 1e-3    // Higher LR for faster convergence
    }
}
```

## Expected Performance Improvements:

| Optimization | Memory Reduction | Speed Improvement |
|-------------|------------------|-------------------|
| Mixed Precision | 50% | 2-3x |
| MC Sample=1 | 95% (30→1) | 10x |
| Gradient Accumulation | Varies | Enables training |
| DataLoader Optimization | - | 30-50% |
| Gradient Checkpointing | 30-40% | -20% (slower) |
| Total (Combined) | **~80%** | **~5-10x** |

## Implementation Priority:

1. **Week 1**: Mixed Precision + MC Sample Reduction (biggest wins)
2. **Week 2**: Gradient Accumulation + DataLoader optimization
3. **Week 3**: Model architecture reductions + Gradient Checkpointing
4. **Week 4**: Advanced optimizations (compilation, sparse embeddings)

## Monitoring:

```python
# Add to training loop:
import torch.cuda as cuda

print(f"GPU Memory Allocated: {cuda.memory_allocated()/1e9:.2f} GB")
print(f"GPU Memory Cached: {cuda.memory_reserved()/1e9:.2f} GB")
print(f"Samples/sec: {batch_size * accumulation_steps / batch_time:.1f}")
```

## Testing Strategy:

1. Start with 1% of data (80K samples)
2. Verify training converges
3. Scale to 10% (800K samples)
4. Profile memory/speed
5. Scale to full dataset

