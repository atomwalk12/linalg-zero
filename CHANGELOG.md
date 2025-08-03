# CHANGELOG


## v0.2.0 (2025-08-03)

### Bug Fixes

- Add logging utilities for files and stdout
  ([`d2692e6`](https://github.com/atomwalk12/linalg-zero/commit/d2692e68bbedabb5d110582737b3d995e26d64a2))

- Be granular about exceptions being thrown
  ([`4c88f9f`](https://github.com/atomwalk12/linalg-zero/commit/4c88f9fc8a14effb11b49ce90fb323122e0c4916))

Co-authored-by: gemini-code-assist[bot] <176961590+gemini-code-assist[bot]@users.noreply.github.com>

- Ensure an exception is raised if the subprocess returns a non-zero exit code
  ([`4695ab9`](https://github.com/atomwalk12/linalg-zero/commit/4695ab9c9afa4b56ec4167d99777b6b230d01ac7))

Co-authored-by: gemini-code-assist[bot] <176961590+gemini-code-assist[bot]@users.noreply.github.com>

- Finetune the number of Llama-cpp GPU layers offloaded to the GPU
  ([`a37fc13`](https://github.com/atomwalk12/linalg-zero/commit/a37fc130ab9b04be0ecccfe1278c6e1fe68d3350))

- **dependencies**: Pin llama-cpp-python version to 0.3.13), update distilabel dependencies and add
  lock file
  ([`ef18d53`](https://github.com/atomwalk12/linalg-zero/commit/ef18d53a9fda569c6148ab8130557ce689d7c7ba))

- **distillation**: Add script to push debugging dataset to huggingface
  ([`6d6c91a`](https://github.com/atomwalk12/linalg-zero/commit/6d6c91ac275aaaee1d145bbb2d99296da823c617))

- **inference**: Add hf_pretrained_model_name_or_path and remove redundant installation in launch
  script
  ([`75be580`](https://github.com/atomwalk12/linalg-zero/commit/75be58003bf3fd321deb185d117c7fc43ba11a57))

### Build System

- Remove python 3.9 support
  ([`e7421e6`](https://github.com/atomwalk12/linalg-zero/commit/e7421e611029b15f21ac9dcef2c4e14d97e99f9b))

### Features

- Add configuration parameters, tasks for running distributed training and pin dependencies
  ([`180fa5f`](https://github.com/atomwalk12/linalg-zero/commit/180fa5f4ccabf0f45d5c02371b80d0774a9cace5))

- Add dataset generator utility
  ([`279b0bd`](https://github.com/atomwalk12/linalg-zero/commit/279b0bdcf867b586c3b201afe425da1e93f2bfea))

- Add workflow to generate a new dataset
  ([`e0ba35d`](https://github.com/atomwalk12/linalg-zero/commit/e0ba35d7792309627dcca34293d11cb758ac3def))

- Implement question generation factories for arithmetic and linear algebra
  ([`f20fb2c`](https://github.com/atomwalk12/linalg-zero/commit/f20fb2c97d825d059eb6612d1f7cd4504c41f74c))

- make use of a global registry to keep track of the various problem definitions

- **distillation**: Add centralised control for launching the inference server
  ([`23a4078`](https://github.com/atomwalk12/linalg-zero/commit/23a4078dc59d1706989850f32d67786db32191ec))

- **distillation**: Add filter to easily track and discard incorrect results
  ([`d991534`](https://github.com/atomwalk12/linalg-zero/commit/d991534a1f6eee0c691520f7d0a53bbdd0fbb341))

- **distillation**: Add generation pipeline
  ([`d18317f`](https://github.com/atomwalk12/linalg-zero/commit/d18317f32e6e03dbb04e124560c58c731af57ba9))

- **distillation**: Add local script for llama.cpp inference server
  ([`bf1c3e1`](https://github.com/atomwalk12/linalg-zero/commit/bf1c3e18ff1ff5cf39b47b8df3319898b2f41c3a))

- **distillation**: Add planning and tool selection
  ([`6b9e3e2`](https://github.com/atomwalk12/linalg-zero/commit/6b9e3e2e952267ce9cd5796bcb0d448a48f98a3e))

- **distillation**: Add result synthesiser
  ([`017a47c`](https://github.com/atomwalk12/linalg-zero/commit/017a47ca3204fb0a27c21532ba1e9fd480dae7a6))

- **distillation**: Add the argilla components for simpler result inspection
  ([`20b7649`](https://github.com/atomwalk12/linalg-zero/commit/20b7649f88621cd5d5fea11624ce4e5fde8441de))

- **distillation**: Add the planner component
  ([`ed5175a`](https://github.com/atomwalk12/linalg-zero/commit/ed5175aca518ed6d593ae863e00f6e84a02f6f4e))

- **distillation**: Add verl dependency for GRPO
  ([`9d7ccb6`](https://github.com/atomwalk12/linalg-zero/commit/9d7ccb65c8488760e566cda5aabb345ef41a78f6))

- **distillation**: Code execution component
  ([`24072e7`](https://github.com/atomwalk12/linalg-zero/commit/24072e75338875335d1efd226df6b72339f0c3c4))

- **distillation**: Customise the chat generation pipeline to preserve input/output results
  ([`9973a10`](https://github.com/atomwalk12/linalg-zero/commit/9973a1009b1400a3e0cc0e0562b1464ffbd291b8))

- **distillation**: Demonstrate the tool selection component and update planner to use
  ChatGeneration
  ([`8fe4e0a`](https://github.com/atomwalk12/linalg-zero/commit/8fe4e0a840dfa665b80a89b2a62e3f07fb5806d5))

- **distillation**: Implement function calling pipeline using Llama-cpp
  ([`f8776ce`](https://github.com/atomwalk12/linalg-zero/commit/f8776cedf82002f1c08a18f9252a929d70991dd1))

- **distillation**: Improve launch script to download models from the hf-hub and tune configuration
  parameters
  ([`6fc9503`](https://github.com/atomwalk12/linalg-zero/commit/6fc950382725f00268b4b9b671c90abed1f3ab02))

- **distillation**: Integrate all related components to generate new data (completed pipeline)
  ([`c8532a4`](https://github.com/atomwalk12/linalg-zero/commit/c8532a4d5a6da3bfb7ce82e6de10f81f033ab826))

- **distillation**: Integrate math-verify for formal evaluation of the output
  ([`a2c3757`](https://github.com/atomwalk12/linalg-zero/commit/a2c3757e216d462ed9915a179e517382c68639bd))

- **sft**: Add additional callbacks (i.e. evaluation, push revision to hub, early stopping)
  ([`90e7f35`](https://github.com/atomwalk12/linalg-zero/commit/90e7f3532e0ca502f0edda571e5b33889cb41b89))

- **sft**: Complete evaluation with the ability to resume training, log results via wandb, create
  model cards and save to the huggingface hub
  ([`a7d86a3`](https://github.com/atomwalk12/linalg-zero/commit/a7d86a3e5c59ab527e9d3ebc5779f4d35065b2e7))

### Testing

- Add dataset configuration file
  ([`5372677`](https://github.com/atomwalk12/linalg-zero/commit/5372677debfcb91b17c0bbc5af665bd19872c3ba))

- Check default registry configuration
  ([`47f190a`](https://github.com/atomwalk12/linalg-zero/commit/47f190a94e045650b3c1cffee3d4888d26b16164))


## v0.1.0 (2025-07-08)

### Bug Fixes

- Add container fix for Docker
  ([`bf3884b`](https://github.com/atomwalk12/linalg-zero/commit/bf3884b10d03f6dfa253944c2497297bc91d32d2))

- Improve release pipeline to include semantic releases
  ([`888738c`](https://github.com/atomwalk12/linalg-zero/commit/888738c4b914c65e001025ca3ba4473c1712f235))

### Features

- Add additional tasks
  ([`c036f31`](https://github.com/atomwalk12/linalg-zero/commit/c036f31ba63af43b205e7a73464935f98f602773))

- Add codeql support
  ([`9b337c6`](https://github.com/atomwalk12/linalg-zero/commit/9b337c6eb7892943e2e6e1d004ec8611f868e0bd))

- Add gemini style guide
  ([`0f65ed1`](https://github.com/atomwalk12/linalg-zero/commit/0f65ed1847f72f900cf3d37e0766e1d06269c451))

- Set up semantic release
  ([`a97ecd3`](https://github.com/atomwalk12/linalg-zero/commit/a97ecd319d69c348b1a3a760c00ad6678a7b7339))

- Update mkdocs theme
  ([`8dd0a40`](https://github.com/atomwalk12/linalg-zero/commit/8dd0a40dd627c64ce868d557b8c889c7c6a24d0d))
