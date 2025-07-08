# linalg-zero

[![Release](https://img.shields.io/github/v/release/atomwalk12/linalg-zero)](https://img.shields.io/github/v/release/atomwalk12/linalg-zero)
[![Build status](https://img.shields.io/github/actions/workflow/status/atomwalk12/linalg-zero/main.yml?branch=main)](https://github.com/atomwalk12/linalg-zero/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/atomwalk12/linalg-zero/branch/main/graph/badge.svg)](https://codecov.io/gh/atomwalk12/linalg-zero)
[![Commit activity](https://img.shields.io/github/commit-activity/m/atomwalk12/linalg-zero)](https://img.shields.io/github/commit-activity/m/atomwalk12/linalg-zero)
[![License](https://img.shields.io/github/license/atomwalk12/linalg-zero)](https://img.shields.io/github/license/atomwalk12/linalg-zero)

This repository offers tools for generating a linear algebra problem dataset and training an open-source base model, aiming to explore its potential for emergent reasoning as inspired by the Deepseek-R1 paper.

- **Documentation** <https://atomwalk12.github.io/linalg-zero/>

## Installation

### PyTorch Configuration

This project is configured with PyTorch defaults that work for most users:
- **Linux**: CUDA 12.8 builds (for GPU acceleration)
- **macOS/Windows**: CPU builds

#### For Different CUDA Versions

If you need a different CUDA version, run the following commands:

```bash
# To automatically detect and install dependencies:
UV_TORCH_BACKEND=auto uv sync

# Alternatively, to install Pytorch with a specific CUDA version:
nvidia-smi                      # check your CUDA version
UV_TORCH_BACKEND=cu121 uv sync  # for CUDA 12.1
UV_TORCH_BACKEND=cu124 uv sync  # for CUDA 12.4
UV_TORCH_BACKEND=cpu uv sync    # for CPU-only
```

For the available CUDA versions see the [official documentation](https://pytorch.org/get-started/locally/).
