# ART Unsloth Gradient Accumulation Deadlock

## Problem

The ART unsloth training loop **deadlocks** when `gradient_accumulation_steps > 1` due to a mismatch between input/output queue expectations.

## Root Cause

The service assumes a **1:1 input-to-result ratio**, but gradient accumulation requires **N:1**:

- **Service**: Puts 1 input → waits for 1 result
- **Trainer** (with `gradient_accumulation_steps=N`): Needs N inputs → produces 1 result

```
gradient_accumulation_steps = 1 (Works):
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Service │────▶│ Trainer │────▶│ Service │
│ input#1 │     │ step+log│     │ result  │
└─────────┘     └─────────┘     └─────────┘

gradient_accumulation_steps = 4 (Deadlock):
┌─────────┐     ┌─────────┐
│ Service │────▶│ Trainer │
│ input#1 │     │ substep │ (no log, needs 3 more inputs)
│  wait   │         │
│   ⏳    │         │
│   💀    │◀────────┘ (waits for input#2, queue empty)
└─────────┘
```

## Technical Details

### When Results Are Added to Queue

Results are only added via the patched `trainer.log()` method, called from:

1. `on_step_end()` callback (after full optimizer step)
2. Only when gradient accumulation completes (`do_sync_step=True`)
3. **NOT** during `on_substep_end()` (gradient accumulation substeps)

**Code locations:**
- Queue put: `train.py:253` → `results_queue.put_nowait(logs)`
- Called from: `transformers/trainer.py:2664` → `_maybe_log_save_evaluate()`
- After: Optimizer step completes all gradient accumulation substeps

### Service Loop Expectation

From `service.py:107-199`:

```python
for offset in range(0, packed_tensors["tokens"].shape[0]):
    # Put 1 input
    self.state.inputs_queue.put_nowait(TrainInputs(...))

    # Immediately wait for 1 result
    done, _ = await asyncio.wait([
        asyncio.create_task(self.results_queue.get()),  # Expects 1:1 mapping
        self._train_task,
    ], return_when=asyncio.FIRST_COMPLETED)
```

## Solutions

### Option 1: Keep `gradient_accumulation_steps = 1` (Current)

**Recommended** - No code changes needed.

```yaml
trainer:
  gradient_accumulation_steps: 1  # Required
  per_device_train_batch_size: 2  # Increase this for larger batches
```

### Option 2: Batch Inputs in Service

Modify `service.py` to queue N inputs before waiting:

```python
gradient_accum_steps = self.config.get("trainer", {}).get("gradient_accumulation_steps", 1)

for offset in range(0, packed_tensors["tokens"].shape[0]):
    # Queue all inputs for gradient accumulation
    for _ in range(gradient_accum_steps):
        self.state.inputs_queue.put_nowait(TrainInputs(...))

    # Wait for aggregated result
    result = await self.results_queue.get()
    yield result
```

### Option 3: Log Every Substep (Not Recommended)

Modify trainer to call `log()` during `on_substep_end()`, but this breaks gradient accumulation semantics.

## Configuration Check

Current safe configuration in `config/grpo/Qwen/Qwen3-0.6B/config.yaml`:

```yaml
trainer:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 1  # ✅ Must stay at 1
  num_generations: 2
```

**Warning:** Changing `gradient_accumulation_steps` to any value > 1 will cause immediate deadlock.
