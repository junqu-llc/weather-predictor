#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up Python environment for Weather Predictor...${NC}"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
uv venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies using uv
echo -e "${BLUE}Installing dependencies...${NC}"
uv pip install -r requirements.txt

# Install additional development tools
echo -e "${BLUE}Installing development tools...${NC}"
uv pip install -e .

# Set up pre-commit hooks
echo -e "${BLUE}Setting up pre-commit hooks...${NC}"
cat > .pre-commit-config.yaml << EOL
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
    -   id: black

-   repo: https://github.com/pycqa/isort
    rev: 5.13.0
    hooks:
    -   id: isort

-   repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.8
    hooks:
    -   id: ruff
        args: [--fix]
EOL

# Install pre-commit
uv pip install pre-commit
pre-commit install

# Create pyproject.toml for tool configuration
echo -e "${BLUE}Creating pyproject.toml...${NC}"
cat > pyproject.toml << EOL
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "weather_predictor"
version = "0.1.0"
description = "Weather prediction using LSTM and NWS API"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}

[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88

[tool.ruff]
line-length = 88
target-version = "py38"
select = ["E", "F", "B", "I", "UP"]
ignore = []

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
check_untyped_defs = true

[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --cov=src --cov-report=term-missing"
testpaths = ["tests"]
EOL

# Create setup.py
echo -e "${BLUE}Creating setup.py...${NC}"
cat > setup.py << EOL
from setuptools import setup, find_packages

setup(
    name="weather_predictor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if not line.startswith("#") and line.strip()
    ],
)
EOL

echo -e "${GREEN}Environment setup complete!${NC}"
echo -e "${BLUE}To activate the environment, run:${NC}"
echo -e "    source .venv/bin/activate" 