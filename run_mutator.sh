#!/bin/bash

# Mutator CLI Wrapper Script
# This script activates the virtual environment and runs the mutator CLI

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment and run mutator
source "$SCRIPT_DIR/.venv/bin/activate" && python -m mutator.cli "$@" 