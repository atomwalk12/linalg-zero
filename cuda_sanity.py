# cuda_sanity.py
import multiprocessing as mp

import torch


def worker():
    torch.cuda.set_device(0)
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("child OK:", torch.cuda.get_device_name(0), y.shape)


if __name__ == "__main__":
    print("cuda_available:", torch.cuda.is_available())
    p = mp.get_context("spawn").Process(target=worker)
    p.start()
    p.join()
    print("rc", p.exitcode)


# root@69a898e4a95b:/workspace/linalg-zero# CUDA_VISIBLE_DEVICES=0 uv run python cuda_sanity.py
# /workspace/linalg-zero/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
#   import pynvml  # type: ignore[import]
# cuda_available: True
# /workspace/linalg-zero/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
#   import pynvml  # type: ignore[import]
# Process SpawnProcess-1:
# Traceback (most recent call last):
#   File "/usr/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap
#     self.run()
#   File "/usr/lib/python3.12/multiprocessing/process.py", line 108, in run
#     self._target(*self._args, **self._kwargs)
#   File "/workspace/linalg-zero/cuda_sanity.py", line 5, in worker
#     torch.cuda.set_device(0)
#   File "/workspace/linalg-zero/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py", line 569, in set_device
#     torch._C._cuda_setDevice(device)
# torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
# CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
# For debugging consider passing CUDA_LAUNCH_BLOCKING=1
# Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

# rc 1
