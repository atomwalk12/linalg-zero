.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
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

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
