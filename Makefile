.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 uv pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
	@uv sync
	@uv run pre-commit install

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
ifeq ($(CI),true)
	@echo "🔍 CI detected: Running ruff in check mode"
	@uv run ruff check .
	@uv run ruff format --check .
	@SKIP=ruff,ruff-format uv run pre-commit run -a
else
	@uv run pre-commit run -a
endif
	@echo "🚀 Static type checking: Running mypy"
	@uv run mypy
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@uv run deptry .

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml

.PHONY: coverage-site
coverage-site: ## Generate coverage report in HTML format
	@echo "🚀 Generating coverage report in HTML format"
	@uv run coverage html

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: publish
publish: ## Publish a release to PyPI.
	@echo "🚀 Publishing."
	@uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@echo "🚀 Testing documentation build"
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@echo "🚀 Building and serving documentation"
	@uv run mkdocs serve

.PHONY: semantic-release
semantic-release: ## Test semantic release
	@echo "🚀 Testing semantic release"
	@uv run semantic-release -vv --noop version --print

.PHONY: gh-deploy
gh-deploy: ## Deploy the documentation to GitHub Pages
	@echo "🚀 Deploying documentation to GitHub Pages"
	@uv run mkdocs gh-deploy --force

# Set a default value if MODEL_URL is not provided
MODEL_URL ?= https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-GGUF/resolve/main/Llama-3.3-70B-Instruct-Q4_K_M.gguf
CPU_OFFLOAD_LAYERS ?= 30 # This variable may need to be tuned, depending on the model.

.PHONY: distillation-server
distillation-server: ## Start the llama.cpp server
	@echo "🚀 Starting llama.cpp server"
	@sh linalg_zero/distillation/llama-cpp/local.sh $(MODEL_URL) ${CPU_OFFLOAD_LAYERS}

.PHONY: distil
distil: ## Run the distillation pipeline
	@echo "🚀 Running distillation pipeline"
	@uv run python linalg_zero/distil_gen.py --config linalg_zero/config/distillation/debug.yaml

.PHONY: distil-fc
distil-fc: ## Run the distillation pipeline
	@echo "🚀 Running distillation pipeline"
	@uv run python linalg_zero/distil_fc.py --config linalg_zero/config/distillation/debug.yaml

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
