# GPU Memory Optimization Guide for GRPO Training

## Understanding the Problem

### Root Cause Analysis
The OOM error during GRPO training stems from **sequence length** being the primary memory bottleneck:

**Memory Calculation for Logprobs:**
```
Tensor Shape: [batch_size, seq_length, vocab_size]
Example: [4, 6144, 151936] = ~15GB per batch just for logits!
```

During GRPO training, the model:
1. Processes packed sequences
2. Calculates logprobs for all tokens (creates large intermediate tensors)
3. Computes entropy: `(-torch.exp(log_probs_full) * log_probs_full).sum()`
4. This operation is where OOM typically occurs

### Memory Budget Breakdown (24GB GPU)

| Component | Memory | Notes |
|-----------|--------|-------|
| Base Model (0.6B fp16) | ~1.2 GB | Loaded once |
| LoRA Adapters (r=16) | ~0.5 GB | Trainable parameters |
| Optimizer States | ~1.5 GB | Adam optimizer |
| Activations (batch=1, seq=2048) | ~4 GB | With gradient checkpointing |
| Gradients | ~1.5 GB | For backpropagation |
| KV Cache | ~3 GB | For inference/rollouts |
| Logits & Intermediates | ~5 GB | Main bottleneck (was 15GB) |
| VLLM Engine | ~2 GB | Framework overhead |
| System/Fragmentation | ~2 GB | Safety buffer |
| **Total** | **~21 GB** | **Fits in 24GB ✓** |

---

## Memory Optimization Techniques

### 1. Sequence Length Reduction ⭐ (Highest Impact)
**Problem**: Long sequences create massive intermediate tensors
**Solution**: Reduce max sequence length

```yaml
init:
  max_seq_length: 2048  # Default is often 6144
engine:
  max_model_len: 2048   # Must match max_seq_length
```

**Impact**:
- Memory reduction: ~67% for sequence-related tensors
- Trade-off: May truncate longer conversations
- Recommended values: 1024-2048 for RL training, 2048-4096 for SFT

**When to use**: Always start here - this is the most effective optimization

---

### 2. Batch Size Reduction with Gradient Accumulation
**Problem**: Large batches multiply memory requirements
**Solution**: Use batch=1 with gradient accumulation

```yaml
trainer:
  per_device_train_batch_size: 1  # Reduce from 4
  gradient_accumulation_steps: 4  # Maintain effective batch size
```

**Impact**:
- Memory reduction: 75% peak memory
- Trade-off: Slower training (more gradient accumulation steps)
- Maintains training dynamics (same effective batch size)

**When to use**: When sequence length reduction isn't enough

---

### 3. LoRA Rank Reduction
**Problem**: Higher rank = more trainable parameters = more memory
**Solution**: Reduce LoRA rank

```yaml
peft:
  r: 16              # Reduce from 32
  lora_alpha: 32     # Typically 2x the rank
```

**Impact**:
- Memory reduction: 50% fewer trainable parameters
- Trade-off: Slightly reduced model capacity
- Research shows r=8-16 is often sufficient for <3B models

**When to use**: For smaller models (<3B parameters)

---

### 4. GPU Memory Utilization
**Problem**: VLLM allocates memory aggressively
**Solution**: Reduce memory utilization to leave headroom

```yaml
init:
  gpu_memory_utilization: 0.4  # Default is often 0.9
engine:
  gpu_memory_utilization: 0.4
```

**Impact**:
- Leaves more memory for training operations
- Trade-off: Smaller KV cache for inference
- Recommended: 0.3-0.5 for training, 0.7-0.9 for inference only

**When to use**: Always - provides safety margin for training

---

### 5. Gradient Checkpointing
**Problem**: Storing all activations for backprop uses memory
**Solution**: Recompute activations during backward pass

```yaml
peft:
  use_gradient_checkpointing: "unsloth"  # or "true"
```

**Impact**:
- Memory reduction: ~30-50% for activations
- Trade-off: ~20-30% slower training (recomputation overhead)
- Unsloth implementation is optimized for speed

**When to use**: Almost always for training large models

---

### 6. Model Quantization (4-bit/8-bit)
**Problem**: Full precision model weights use significant memory
**Solution**: Load base model in 4-bit or 8-bit

```yaml
init:
  load_in_4bit: true   # or load_in_8bit: true
```

**Impact**:
- Memory reduction: 75% (4-bit) or 50% (8-bit) for base model
- Trade-off: Slight quality degradation, slower inference
- LoRA adapters remain in fp16/bf16

**When to use**: For larger models (>3B) or as last resort

---

### 7. PyTorch Memory Fragmentation Fix
**Problem**: Memory fragmentation prevents allocation
**Solution**: Enable expandable segments

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

**Impact**:
- Better memory utilization
- No performance trade-off
- Helps with "out of memory" despite showing free memory

**When to use**: Always - no downside

---

### 8. Flash Attention 2
**Problem**: Standard attention is memory-intensive
**Solution**: Use Flash Attention 2 (if supported)

```bash
uv pip install flash-attn --no-build-isolation
```

**Impact**:
- Memory reduction: ~20-30% for attention operations
- Speed improvement: ~2-3x faster attention
- Automatically used if installed

**When to use**: If your GPU supports it (Ampere/Ada/Hopper)

---

### 9. Reduce Trajectory Groups
**Problem**: Storing many trajectories uses memory
**Solution**: Reduce trajectories per group

```yaml
training:
  trajectories_per_group: 16  # Reduce from 32
  groups_per_step: 2          # Reduce from 4
```

**Impact**:
- Memory reduction: Proportional to reduction
- Trade-off: Less diverse training signal per step
- May need more training steps

**When to use**: For GRPO/RL training specifically

---

### 10. Mixed Precision Training
**Problem**: FP32 uses 2x memory of FP16
**Solution**: Use BF16 or FP16 training

```yaml
trainer:
  bf16: true  # or fp16: true
```

**Impact**:
- Memory reduction: 50% for weights/gradients
- Usually enabled by default in modern frameworks
- BF16 preferred for stability

**When to use**: Always (usually default)

---

## Optimization Strategy

### Step-by-Step Approach

1. **Start Conservative** (Apply these first):
   - ✅ Reduce `max_seq_length` to 2048
   - ✅ Set `gpu_memory_utilization` to 0.4
   - ✅ Enable gradient checkpointing
   - ✅ Enable PyTorch memory fragmentation fix

2. **If Still OOM** (Apply incrementally):
   - Reduce `per_device_train_batch_size` to 1 with gradient accumulation
   - Reduce `max_seq_length` to 1536 or 1024
   - Reduce LoRA rank to 8-12

3. **Last Resort** (Significant trade-offs):
   - Enable 4-bit quantization
   - Reduce trajectory groups significantly
   - Use smaller base model

### Quick Reference Table

| Optimization | Memory Saved | Speed Impact | Quality Impact | Difficulty |
|--------------|--------------|--------------|----------------|------------|
| Sequence Length ↓ | ⭐⭐⭐ High | ✓ None | ⚠️ May truncate | Easy |
| Batch Size ↓ + Grad Accum | ⭐⭐⭐ High | ⚠️ Slower | ✓ None | Easy |
| LoRA Rank ↓ | ⭐⭐ Medium | ✓ Faster | ⚠️ Slight | Easy |
| GPU Memory Util ↓ | ⭐ Low | ✓ None | ✓ None | Easy |
| Gradient Checkpointing | ⭐⭐ Medium | ⚠️ Slower | ✓ None | Easy |
| 4-bit Quantization | ⭐⭐⭐ High | ⚠️ Slower | ⚠️ Slight | Medium |
| Flash Attention | ⭐⭐ Medium | ✓ Faster! | ✓ None | Medium |
| PyTorch Mem Fix | ⭐ Low | ✓ None | ✓ None | Easy |

---

## Monitoring and Debugging

### Real-time Monitoring
```bash
# Watch GPU memory usage
watch -n 1 nvidia-smi

# Detailed PyTorch memory stats
python -c "import torch; torch.cuda.empty_cache(); print(torch.cuda.memory_summary())"
```

### Pre-flight Check
```bash
# Run configuration test
uv run python test_memory_config.py
```

### Common OOM Patterns

**OOM during first training step**:
- → Reduce `max_seq_length` or enable 4-bit quantization

**OOM during logprob calculation**:
- → Reduce `per_device_train_batch_size` to 1

**OOM during inference/rollout**:
- → Reduce `gpu_memory_utilization` or concurrent rollouts

**OOM with "free memory available" message**:
- → Enable PyTorch memory fragmentation fix

---

## Model-Specific Recommendations

### For 0.5-1B Models (e.g., Qwen3-0.6B)
```yaml
max_seq_length: 2048
per_device_train_batch_size: 1-2
lora_rank: 16
gpu_memory_utilization: 0.4-0.5
```

### For 1-3B Models (e.g., Qwen3-1.7B)
```yaml
max_seq_length: 1536-2048
per_device_train_batch_size: 1
lora_rank: 16-32
gpu_memory_utilization: 0.3-0.4
load_in_4bit: optional
```

### For 3-7B Models (e.g., Qwen3-4B)
```yaml
max_seq_length: 1024-1536
per_device_train_batch_size: 1
lora_rank: 8-16
gpu_memory_utilization: 0.3
load_in_4bit: recommended
```

---

## References and Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth) - Optimized training
- [VLLM Memory Guide](https://docs.vllm.ai/en/latest/models/engine_args.html) - Engine configuration
- [PyTorch CUDA Memory](https://pytorch.org/docs/stable/notes/cuda.html#memory-management) - Memory management
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - 4-bit quantization + LoRA
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Low-rank adaptation
- [Flash Attention](https://arxiv.org/abs/2205.14135) - Efficient attention
