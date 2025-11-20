#!/bin/bash

# run_evaluations.sh
# Script to run evaluations with proper environment setup

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the project root (one level up from scripts directory)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment variables from .env file
ENV_FILE="$PROJECT_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from $ENV_FILE"
    set -a  # automatically export all variables
    source "$ENV_FILE"
    set +a
else
    echo "Warning: .env file not found at $ENV_FILE"
fi

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Change to project root directory
cd "$PROJECT_ROOT"

# Print some info for debugging
echo "Project root: $PROJECT_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo "Current directory: $(pwd)"
echo ""
echo "Running evaluations..."
echo "----------------------------------------"

# Run the Python script
python3.12 notebooks/models/evaluations.py "$@"

# Capture exit code
EXIT_CODE=$?

# Exit with the same code as the Python script
exit $EXIT_CODE
